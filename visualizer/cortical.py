"""Nilearn cortical rendering shared by the web visualizer and legacy exports."""
import os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1] / 'tribe'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache/matplotlib'))

from dataset import STANDARD_VIEWS, view_metadata


def brain(fig, cell, fs, values, hemi, view, limit, positive=False):
    from nilearn import plotting
    from nilearn.surface import PolyMesh, load_surf_data
    ax = fig.add_subplot(cell, projection='3d')
    if hemi == 'both':
        # PolyMesh keeps the inflated hemispheres side by side and preserves
        # the model's left-then-right vertex order in one inferior projection.
        mesh = PolyMesh(left=fs['infl_left'], right=fs['infl_right'])
        half = values
        background = np.concatenate([load_surf_data(fs[f'sulc_{h}'])
                                     for h in ('left', 'right')])
    else:
        mesh = fs[f'infl_{hemi}']
        half = values[:10242] if hemi == 'left' else values[10242:]
        background = fs[f'sulc_{hemi}']
    plotting.plot_surf_stat_map(
        mesh, half, hemi=hemi, view=view,
        bg_map=background, cmap='hot' if positive else 'RdBu_r',
        vmin=0 if positive else -limit, vmax=limit,
        symmetric_cbar=not positive, colorbar=False, axes=ax, figure=fig,
        threshold=1e-12 if positive else None,
    )
    if hemi == 'both' and view == 'ventral':
        # Keep anterior at the top, with hemispheres side by side. Looking
        # from below puts anatomical right on the left of the image.
        ax.view_init(elev=-90, azim=90)
        for x, label in ((.26, 'R'), (.74, 'L')):
            ax.text2D(x, .04, label, transform=ax.transAxes, ha='center', fontsize=8)
    title = 'Inferior / ventral · both hemispheres' if hemi == 'both' else f'{hemi.capitalize()} {view}'
    ax.set_title(title, fontsize=8 if hemi == 'both' else 10, pad=0)
    return ax



def render_groups(groups, output, views='all'):
    view_metadata(views)
    import matplotlib
    matplotlib.use('Agg')
    from nilearn import datasets
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    fs = datasets.fetch_surf_fsaverage(mesh='fsaverage5', data_dir=str(ROOT / 'cache/nilearn'))
    folder = output / 'brain_maps'
    folder.mkdir(exist_ok=True)
    limits = {kind: max(1e-12, max(float(np.abs(g[kind]).max()) for g in groups))
              for kind in ('mean', 'contrast')}
    layouts = []
    if views in ('four', 'all'):
        layouts.append(('', STANDARD_VIEWS))
    if views in ('inferior', 'all'):
        layouts.append(('_inferior', [('both', 'ventral')]))
    for number, group in enumerate(groups, 1):
        for kind in ('mean', 'contrast'):
            for suffix, panels in layouts:
                fig = plt.figure(figsize=(3 * len(panels), 3.4), facecolor='white')
                gs = fig.add_gridspec(1, len(panels), wspace=0)
                limit = limits[kind]
                for column, (hemi, view) in enumerate(panels):
                    brain(fig, gs[0, column], fs, group[kind], hemi, view, limit)
                fig.subplots_adjust(top=.92, bottom=.28, left=0, right=1)
                cax = fig.add_axes([.14, .15, .72, .03] if suffix else [.34, .15, .32, .03])
                bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-limit, limit), cmap='RdBu_r'),
                                   cax=cax, orientation='horizontal')
                bar.set_label('Predicted response (model units)', fontsize=8 if suffix else 10)
                name = f"{group['attribute']}_{group['value']}_{kind}{suffix}.png"
                fig.savefig(folder / name, dpi=110, facecolor='white')
                plt.close(fig)
                group[f'{kind}{suffix}_image'] = f'brain_maps/{name}'
        print(f'Rendered group {number}/{len(groups)}: {group["attribute"]} {group["label"]}', flush=True)
    return limits

