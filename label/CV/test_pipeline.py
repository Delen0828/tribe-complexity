"""Behavioral checks for sampling, native masks, triage and count semantics."""
import csv
import random
import tempfile
import unittest
from unittest.mock import patch, Mock
from pathlib import Path
import numpy as np
from common import CV, ROOT, encode_mask, decode_mask, deduplicate, mask_iou, read_json, write_json, validate_preparation, digest
from backends import candidate
from render import render_figure, read_decisions
from carry_review import carry
from run import process_figure


class MaskTests(unittest.TestCase):
    def test_holes_and_native_offset(self):
        mask=np.zeros((13,17),bool);mask[3:10,5:12]=True;mask[5:8,7:10]=False
        rle=encode_mask(mask)
        self.assertEqual(rle['origin'],[5,3]);self.assertEqual(rle['size'],[7,7])
        np.testing.assert_array_equal(decode_mask(rle),mask[3:10,5:12])
        emitted=candidate(mask,.9,[100,200],'test')
        self.assertEqual(emitted['rle']['origin'],[105,203]);self.assertEqual(emitted['area'],40)

    def test_empty_and_single_pixel(self):
        self.assertEqual(decode_mask(encode_mask(np.zeros((2,2),bool))).size,0)
        r=encode_mask(np.ones((1,1),bool));self.assertEqual(r['counts'],[0,1]);self.assertTrue(decode_mask(r)[0,0])

    def test_dedup_preserves_overlapping_boxes(self):
        a=np.eye(9,dtype=bool);b=np.fliplr(a)
        candidates=[candidate(a,.9,[0,0],'test')|{'candidate_id':'a'},candidate(a,.8,[0,0],'test')|{'candidate_id':'duplicate'},candidate(b,.85,[0,0],'test')|{'candidate_id':'b'}]
        deduplicate(candidates,.9)
        self.assertEqual(candidates[1]['duplicate_of'],'a');self.assertNotIn('duplicate_of',candidates[2])
        self.assertLess(mask_iou(candidates[0]['rle'],candidates[2]['rle']),.1)

    def test_corrupt_rle_rejected(self):
        with self.assertRaises(ValueError):decode_mask({'origin':[0,0],'size':[3,3],'counts':[5]})


class SampleTests(unittest.TestCase):
    def test_sample_matches_frozen_training_draw(self):
        selection=read_json(CV/'selection.json');labels={r['image_id']:r for r in read_json(ROOT/'label/output/labels.json')}
        with (ROOT/'label/stats/results/split.csv').open() as f:train={r['image_id'] for r in csv.DictReader(f) if r['split']=='train'}
        rng=random.Random(0);expected=[]
        for band in range(10):
            pool=sorted([r for i,r in labels.items() if i in train and min(9,int(r['mean_human_complexity_rating']//10))==band],key=lambda r:r['index'])
            expected.extend(r['image_id'] for r in rng.sample(pool,15))
        self.assertEqual([r['image_id'] for r in selection['records']],expected)
        self.assertEqual(len(set(expected)),150)

    def test_all_preparation_geometry(self):
        for row in read_json(CV/'selection.json')['records']:
            validate_preparation(read_json(CV/'preparation/figures'/f"{row['image_id']}.json"),row)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='cv-test-')
        self.directory=Path(self.temp.name)
        self.row=next(r for r in read_json(CV/'selection.json')['records'] if r['image_id']=='2225.png')
        mask=np.ones((3,4),bool)
        self.c=candidate(mask,.9,[20,30],'test')|{'candidate_id':'p01-00001','panel_id':'p01','families':['bar_segment']}
        self.raw={'figure_id':self.row['image_id'],'candidate_set_id':'set-a','preparation_sha256':'prep','status':'segmented','panels':[{'panel_id':'p01','status':'segmented','applicable':True,'mark_families':['bar_segment'],'preparation_status':'prepared','notes':''}],'candidates':[self.c],'runtime_seconds':1,'errors':[]}

    def tearDown(self):self.temp.cleanup()

    def decision(self,state):
        write_json(self.directory/'candidates'/f"{self.row['image_id']}.json",self.raw)
        write_json(self.directory/'triage'/f"{self.row['image_id']}.json",{'candidate_set_id':'set-a','reviewed_by':'test','decisions':{'p01-00001':{'decision':state,'family':'bar_segment','reason':'test behavioral case'}}})

    def test_stale_triage_is_rejected(self):
        self.decision('retain');self.raw['candidate_set_id']='set-b'
        with self.assertRaisesRegex(ValueError,'Stale'):read_decisions(self.directory,self.row,self.raw)

    def test_count_states(self):
        # Asset rendering requires its run to be within the static project.
        inside=CV/'cache/tests';inside.mkdir(parents=True,exist_ok=True)
        self.temp.cleanup();self.temp=tempfile.TemporaryDirectory(dir=inside);self.directory=Path(self.temp.name)
        self.decision('retain');result=render_figure(self.row,self.directory)
        self.assertEqual(result['count'],1);self.assertEqual(result['status'],'complete')
        self.decision('reject');result=render_figure(self.row,self.directory)
        self.assertEqual(result['count'],0);self.assertEqual(result['status'],'complete')
        self.decision('ambiguous');result=render_figure(self.row,self.directory)
        self.assertEqual(result['count'],0);self.assertEqual(result['status'],'partial');self.assertEqual(result['ambiguous_count'],1)
        self.raw.update(status='not_applicable',candidates=[])
        self.raw['panels'][0].update(status='not_applicable',applicable=False)
        write_json(self.directory/'candidates'/f"{self.row['image_id']}.json",self.raw)
        write_json(self.directory/'triage'/f"{self.row['image_id']}.json",{'candidate_set_id':'set-a','reviewed_by':'test','decisions':{}})
        result=render_figure(self.row,self.directory);self.assertIsNone(result['count']);self.assertEqual(result['status'],'not_applicable')


class ReviewTransferTests(unittest.TestCase):
    def test_only_identical_masks_transfer_and_source_changes_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            base=candidate(np.ones((3,3),bool),.9,[0,0],'test')|{'candidate_id':'p01-00001','panel_id':'p01','families':['point']}
            second=candidate(np.ones((3,3),bool),.9,[10,10],'test')|{'candidate_id':'p01-00002','panel_id':'p01','families':['point']}
            old={'figure_id':'unit.png','candidate_set_id':'old','source_sha256':'source-a','candidates':[base,second]}
            changed=dict(second);mask=np.ones((3,3),bool);mask[0,0]=False
            changed['rle']=encode_mask(mask);changed['rle']['origin']=[10,10]
            current=old|{'candidate_set_id':'new','candidates':[base,changed]}
            write_json(directory/'candidate_history/unit.png.old.json',old)
            write_json(directory/'candidates/unit.png.json',current)
            previous=directory/'previous.json';write_json(previous,{'figure_id':'unit.png','candidate_set_id':'old','decisions':{c['candidate_id']:{'decision':'retain','family':'point','reason':'Visually reviewed'} for c in old['candidates']}})
            with patch('carry_review.active_run',return_value=directory):
                result=carry(str(previous),{'p01'})
                self.assertEqual(set(result['decisions']),{'p01-00001'})
                self.assertEqual(result['pending_candidate_ids'],['p01-00002'])
                current['source_sha256']='source-b';write_json(directory/'candidates/unit.png.json',current)
                with self.assertRaisesRegex(ValueError,'Source image changed'):carry(str(previous),{'p01'})

    def test_resume_skips_valid_receipt_and_changed_preparation_recomputes(self):
        from types import SimpleNamespace
        row=next(r for r in read_json(CV/'selection.json')['records'] if r['image_id']=='2225.png')
        prep=read_json(CV/'preparation/figures/2225.png.json')
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);path=directory/'candidates/2225.png.json'
            previous={'preparation_sha256':digest(prep),'errors':[],'status':'segmented','cached_marker':'unchanged'}
            write_json(path,previous)
            backend=SimpleNamespace(segment=Mock(return_value=iter(())))
            self.assertEqual(process_figure(row,prep,backend,directory,read_json(CV/'config.json')),previous)
            backend.segment.assert_not_called()
            changed=prep|{'notes':prep.get('notes','')+' Updated preparation.'}
            result=process_figure(row,changed,backend,directory,read_json(CV/'config.json'))
            self.assertEqual(result['preparation_sha256'],digest(changed))
            self.assertEqual(result['status'],'segmented')
            self.assertTrue(backend.segment.called)


if __name__=='__main__':unittest.main()
