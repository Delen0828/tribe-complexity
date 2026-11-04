"""Archive and regenerate figures whose visual preparation was corrected."""
import argparse
import shutil
from types import SimpleNamespace
from common import CV, read_json, write_json, digest, now
from run import active_run, process_figure, load_preparation
from backends import create_backend
from render import sheets


def reconcile(checkpoint=None):
    directory=active_run()
    metadata=read_json(directory/'run.json')
    config=metadata['config']
    records=read_json(CV/'selection.json')['records']
    corrections_path=CV/'results/preparation-corrections.json'
    audit=read_json(corrections_path) if corrections_path.exists() else {'corrections':[]}
    backend=None;corrected=[]
    for row in records:
        path=directory/'candidates'/f"{row['image_id']}.json"
        if not path.exists():continue
        raw=read_json(path);prep=load_preparation(row)
        if prep is None or digest(prep)==raw['preparation_sha256']:continue
        if backend is None:
            backend=create_backend(metadata['model'],config,checkpoint)
            if backend.provenance()['checkpoint_sha256'] != metadata['checkpoint_sha256']:
                raise ValueError('Checkpoint changed; create a new model run before reconciling')
        archive=directory/'candidate_history'/f"{row['image_id']}.{raw['candidate_set_id']}.json"
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():shutil.copy2(path,archive)
        triage=directory/'triage'/path.name
        if triage.exists():
            old=directory/'triage_history'/archive.name
            old.parent.mkdir(parents=True,exist_ok=True);triage.replace(old)
        change={'figure_id':row['image_id'],'old_preparation_sha256':raw['preparation_sha256'],'new_preparation_sha256':digest(prep),'old_candidate_set_id':raw['candidate_set_id'],'reason':'Visual review corrected panel or exemplar placement','new_panel_notes':[p.get('notes','') for p in prep['panels']],'archived_raw':str(archive.relative_to(CV)),'recorded_at':now()}
        audit['corrections'].append(change);write_json(corrections_path,audit)
        process_figure(row,prep,backend,directory,config)
        sheets(SimpleNamespace(run_id=directory.name,ids=row['image_id']))
        corrected.append(row['image_id'])
    print('Reconciled:',corrected,flush=True)
    return corrected


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint')
    reconcile(parser.parse_args().checkpoint)
