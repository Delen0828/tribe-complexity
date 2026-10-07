"""Benchmark persistent GPU batching against an existing validated FP32 run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=ROOT/'outputs/thread_benchmark/thread_8')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/gpu_benchmark')
    args=parser.parse_args()
    source=args.reference.resolve(); base=args.output_dir.resolve()
    rows=json.loads((source/'inputs/manifest.json').read_text())
    if (source/'selection.json').exists():
        selection=json.loads((source/'selection.json').read_text())['rows']
        chosen={r['index'] for b in range(10) for r in [x for x in selection if x['bin']==b][:2]}
        rows=[r for r in rows if r['index'] in chosen]
    else:
        rows=rows[:20]
    if not rows:
        raise ValueError('No reference stimuli')
    base.mkdir(parents=True,exist_ok=True)
    results={}
    for precision,batch in [('fp32',1),('fp32',2),('fp32',4),('bf16',2)]:
        key=f'{precision}_batch_{batch}'
        out=base/key
        (out/'inputs').mkdir(parents=True,exist_ok=True)
        for row in rows:
            src=source/'inputs'/f"{row['index']}.png"
            assert hashlib.sha256(src.read_bytes()).hexdigest()==row['sha256']
            shutil.copyfile(src,out/'inputs'/src.name)
        (out/'inputs/manifest.json').write_text(json.dumps(rows))
        with (base/f'{key}.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'run.py'),'--device','cuda','--batch-size',str(batch),
                            '--precision',precision,'--thread','8','--inputs-dir',str(out/'inputs'),
                            '--outputs-dir',str(out),'--manifest',str(out/'inputs/manifest.json')],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        meta=json.loads((out/'run_metadata.json').read_text())
        assert meta['encoder_model_loads']==1
        assert meta['uncached_count']==len(rows)
        errors=[]; correlations=[]
        for row in rows:
            path=f"prediction_{row['index']}.npz"
            with np.load(out/path) as new, np.load(source/path) as old:
                a=new['predictions']; b=old['predictions']
                assert a.shape==b.shape==(3,20484) and np.isfinite(a).all()
                np.testing.assert_array_equal(new['times'],old['times'])
                if precision=='fp32':
                    np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
                errors.append(np.abs(a-b))
                correlations.append(float(np.corrcoef(a.ravel(),b.ravel())[0,1]))
        per_item=meta['processing_wall_seconds']/len(rows)
        results[key]=dict(count=len(rows),wall_seconds=meta['processing_wall_seconds'],
                          seconds_per_stimulus=per_item,full_5800_hours=(meta['model_load_seconds']+5800*per_item)/3600,
                          peak_allocated_gib=meta['peak_cuda_allocated_bytes']/2**30,
                          peak_reserved_gib=meta['peak_cuda_reserved_bytes']/2**30,
                          model_loads=meta['encoder_model_loads'],forward_batches=meta['encoder_forward_batches'],
                          maximum_absolute_error=float(np.max(errors)),mean_absolute_error=float(np.mean(errors)),
                          minimum_correlation=min(correlations))
        (base/'benchmark.json').write_text(json.dumps(results,indent=2)+'\n')
        print(key,json.dumps(results[key]),flush=True)


if __name__=='__main__':
    main()
