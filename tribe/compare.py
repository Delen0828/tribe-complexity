"""Render first-timestep cortical maps and directional/de-meaned contrasts."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import datasets, plotting
from PIL import Image


def colorbar(fig, cell, low, high, cmap, label):
    host = fig.add_subplot(cell)
    host.axis('off')
    cax = host.inset_axes([.05, .18, .25, .64])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(low, high), cmap=cmap),
                 cax=cax, label=label)


def brain(fig, cell, fs, values, hemi, view, limit, positive=False):
    ax = fig.add_subplot(cell, projection='3d')
    half = values[:10242] if hemi == 'left' else values[10242:]
    plotting.plot_surf_stat_map(
        fs[f'infl_{hemi}'], half, hemi=hemi, view=view,
        bg_map=fs[f'sulc_{hemi}'], cmap='hot' if positive else 'RdBu_r',
        vmin=0 if positive else -limit, vmax=limit,
        symmetric_cbar=not positive, colorbar=False, axes=ax, figure=fig,
        threshold=1e-12 if positive else None,
    )
    ax.set_title(f'{hemi.capitalize()} {view}', fontsize=10, pad=0)
    return ax


def main():
    metadata = json.loads((ROOT/'outputs/run_metadata.json').read_text())
    assert metadata['protocol'] == 'paper_3s_t0_bt709_v2'
    assert metadata['duration_seconds'] == 3
    assert metadata['reported_time_index'] == 0
    arrays = [np.load(ROOT/f'outputs/prediction_{i}.npz') for i in (751,4849)]
    preds = [a['predictions'] for a in arrays]
    for a, p in zip(arrays, preds):
        assert p.shape == (3,20484) and np.isfinite(p).all()
        np.testing.assert_array_equal(a['times'], [0,1,2])
    maps = [p[0].copy() for p in preds]
    delta = maps[0]-maps[1]
    demeaned = delta-delta.mean()
    np.savez_compressed(ROOT/'outputs/selected_maps.npz',
                        image_751=maps[0], image_4849=maps[1], difference=delta,
                        demeaned_difference=demeaned, selected_time_seconds=0.)
    metrics = {
        'selection': 'First prediction at t=0; no temporal averaging',
        'spatial_pearson_r': float(np.corrcoef(*maps)[0,1]),
        'mean_absolute_difference': float(np.abs(delta).mean()),
        'root_mean_square_difference': float(np.sqrt(np.mean(delta**2))),
        'mean_signed_difference_751_minus_4849': float(delta.mean()),
        'demeaned_mean_absolute_difference': float(np.abs(demeaned).mean()),
        'per_image': {str(i): {'shape': list(p.shape), 'selected_map_mean': float(m.mean()),
                              'selected_map_rms': float(np.sqrt(np.mean(m*m)))}
                      for i,p,m in zip((751,4849),preds,maps)},
    }
    (ROOT/'outputs/comparison.json').write_text(json.dumps(metrics,indent=2)+'\n')
    fs = datasets.fetch_surf_fsaverage(mesh='fsaverage5', data_dir=str(ROOT/'cache/nilearn'))
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11})
    limit = max(float(np.abs(m).max()) for m in maps)
    dlimit = float(np.abs(delta).max())
    fig = plt.figure(figsize=(16,14), facecolor='white')
    gs = fig.add_gridspec(4,5, width_ratios=[1,1,1,1,.13],
                          height_ratios=[1.4,1,1,1], hspace=.3, wspace=.03)
    fig.suptitle('MASSVIS | Predicted cortical responses', fontsize=22, x=.055, ha='left', y=.97)
    fig.text(.055,.935,'3-second silent static stimuli | First prediction only (t=0) | Population-level TRIBE v2',fontsize=12)
    for col,index in enumerate((751,4849)):
        ax = fig.add_subplot(gs[0,2*col:2*col+2])
        with Image.open(ROOT/f'inputs/{index}.png') as source:
            ax.imshow(source.convert('RGB'))
        ax.axis('off')
        ax.set_title(f'Original stimulus | Image {index}',fontsize=13,pad=10)
    views = [('left','lateral'),('left','medial'),('right','medial'),('right','lateral')]
    titles = ['Image 751 | t=0','Image 4849 | t=0','Signed contrast | 751 - 4849']
    for row,(values,title,lim) in enumerate(zip(maps+[delta],titles,[limit,limit,dlimit]),start=1):
        axes = [brain(fig,gs[row,col],fs,values,hemi,view,lim)
                for col,(hemi,view) in enumerate(views)]
        axes[0].text2D(-.03,1.14,title,transform=axes[0].transAxes,fontsize=13,weight='bold')
        colorbar(fig,gs[row,4],-lim,lim,'RdBu_r','Model output units')
    # fig.text(.055,.04,'fsaverage5: 20,484 vertices. Both image maps share a scale; the signed contrast has its own scale.\nNative hemodynamic compensation retained. Model predictions, not measured fMRI or complexity scores.',fontsize=10,linespacing=1.5)
    fig.subplots_adjust(top=.89,bottom=.08,left=.055,right=.92)
    fig.savefig(ROOT/'outputs/comparison.png',dpi=180,facecolor='white')

    # Follow Supplement C's directional dorsal presentation; also show Supplement B's demeaning check.
    fig2 = plt.figure(figsize=(14,9),facecolor='white')
    gs2 = fig2.add_gridspec(2,5,width_ratios=[1,1,1,1,.13],hspace=.34,wspace=.02)
    fig2.suptitle('Directional cortical contrasts | t=0',fontsize=22,x=.055,ha='left',y=.97)
    fig2.text(.055,.915,'Dorsal views | Positive and negative differences displayed separately',fontsize=12)
    shared = max(dlimit,float(np.abs(demeaned).max()))
    for row,(values,label) in enumerate([(delta,'Raw contrast'),(demeaned,'Whole-brain mean removed')]):
        panels = [np.maximum(values,0),np.maximum(-values,0)]
        for direction,values_positive in enumerate(panels):
            for col,hemi in enumerate(('left','right')):
                ax = brain(fig2,gs2[row,2*direction+col],fs,values_positive,hemi,'dorsal',shared,True)
                if col == 0:
                    title = '751 > 4849' if direction == 0 else '4849 > 751'
                    ax.text2D(0,1.17,f'{label}\n{title}',transform=ax.transAxes,fontsize=12,weight='bold')
        colorbar(fig2,gs2[row,4],0,shared,'hot','Positive contrast (model units)')
    # fig2.text(.055,.035,'One shared, data-derived scale across both directions and both rows; zero/opposite-sign vertices are uncolored.\nDemeaning subtracts the spatial mean of the signed contrast. One image pair: no group inference or significance threshold.',fontsize=10,linespacing=1.5)
    fig2.subplots_adjust(top=.80,bottom=.10,left=.055,right=.90)
    fig2.savefig(ROOT/'outputs/directional_contrasts.png',dpi=180,facecolor='white')
    with PdfPages(ROOT/'outputs/comparison.pdf') as pdf:
        pdf.savefig(fig,facecolor='white')
        pdf.savefig(fig2,facecolor='white')
    plt.close(fig); plt.close(fig2)
    (ROOT/'index.html').write_text('''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRIBE v2 | Cortical maps</title>
<style>body{margin:32px auto;padding:0 24px;max-width:1400px;font:16px/1.6 system-ui;color:#18262b;background:white}h1{line-height:1.15}img{width:100%;height:auto}a{color:#006980}p{max-width:950px}</style>
<h1>TRIBE v2 | MASSVIS 751 and 4849</h1>
<p>Three-second silent clips, native resolution with even-dimension cropping, first prediction at t=0.
Original stimuli, cortical maps, and contrasts. Values are model predictions, not measured fMRI or complexity ratings.</p>
<img src="outputs/comparison.png" alt="Original stimuli 751 and 4849, their first-prediction cortical maps, and signed difference">
<img src="outputs/directional_contrasts.png" alt="Dorsal positive and negative cortical contrasts, with a whole-brain demeaning sensitivity check">
<p><a href="outputs/comparison.pdf">Download neuroimaging report (PDF)</a> |
<a href="processing_comparison.md">Stimulus-processing comparison table</a> |
<a href="README.md">Protocol and reproduction</a></p></html>''')
    print(json.dumps(metrics,indent=2))


if __name__ == '__main__':
    main()
