"""Transfer visual decisions only for byte-identical masks on declared same units."""
import argparse
from common import CV, read_json, write_json, digest
from run import active_run


def carry(previous_path, panels):
    directory=active_run();previous=read_json(previous_path)
    figure_id=previous.get('figure_id')
    if not figure_id:
        raise ValueError('Previous review needs figure_id')
    current=read_json(directory/'candidates'/f'{figure_id}.json')
    old_path=directory/'candidate_history'/f"{figure_id}.{previous['candidate_set_id']}.json"
    old=read_json(old_path)
    if old['source_sha256']!=current['source_sha256']:
        raise ValueError('Source image changed; previous visual decisions cannot transfer')
    def identity(c):
        return digest({k:c[k] for k in ['candidate_id','panel_id','rle','families']})
    old_candidates={c['candidate_id']:c for c in old['candidates']}
    decisions={};pending=[]
    for c in current['candidates']:
        if c.get('duplicate_of') or c.get('automatic_rejection'):continue
        cid=c['candidate_id'];old_candidate=old_candidates.get(cid)
        if c['panel_id'] in panels and cid in previous['decisions'] and old_candidate and identity(c)==identity(old_candidate):
            decisions[cid]=previous['decisions'][cid]
        else:pending.append(cid)
    result={'figure_id':figure_id,'candidate_set_id':current['candidate_set_id'],'reviewed_by':previous.get('reviewed_by','model_assisted'),'reviewed_sheets':previous.get('reviewed_sheets',[]),'quality_issues':previous.get('quality_issues',[]),'notes':previous.get('notes',''),'decisions':decisions,'inherited_review':{'previous_candidate_set_id':previous['candidate_set_id'],'previous_review':str(previous_path),'unchanged_unit_panels':sorted(panels),'verification':'Identical source SHA, candidate ID, panel ID, mask RLE and target-family list','transferred_count':len(decisions)},'pending_candidate_ids':pending}
    target=directory/('triage/parts' if pending else 'triage')/f'{figure_id}.json'
    write_json(target,result)
    print(f'{figure_id}: carried {len(decisions)} identical masks, {len(pending)} require visual review; {target}',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous_review');parser.add_argument('--panels',required=True,help='Panel IDs whose counting units/context are unchanged, comma separated')
    args=parser.parse_args();carry(args.previous_review,set(args.panels.split(',')))
