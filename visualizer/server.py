"""Serve the visualizer and render selected prediction aggregates with Nilearn."""
import argparse
import errno
from collections import OrderedDict
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import sys
import threading
import webbrowser
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tribe'))
sys.path.insert(0, str(ROOT))
from dataset import attribute_metadata, load_predictions, report_rows
from studies import study_variables
from visualizer.analysis import aggregate_xy, complexity_tertiles

STUDIES = {
    'kylie_bar': ('kylie', 'bar'), 'kylie_area': ('kylie', 'area'),
    'hayeong_color': ('hayeong', 'color'),
    'hayeong_hand_drawn': ('hayeong', 'hand_drawn'),
    'arunkumar_experiment_1': ('arunkumar', 'experiment_1'),
    'arunkumar_experiment_2': ('arunkumar', 'experiment_2'),
}
TITLES = {'massvis_sample': 'MassVis · Sample', 'massvis_all': 'MassVis · All',
          'kylie': "Kylie's study", 'hayeong': "Hayeong's study",
          'arunkumar': "Arunkumar's study · Both experiments"}


class Visualizer:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.outputs = self.root / 'tribe/outputs'
        self.snapshots = OrderedDict()
        self.grids = OrderedDict()
        self.lock = threading.RLock()
        self.plot_lock = threading.Lock()

    def folder(self, dataset, data=''):
        if data:
            path = (self.root / unquote(data).lstrip('/')).resolve()
            if path.name not in ('explorer.json', 'selection.json'):
                raise ValueError('Invalid report path')
            folder = path.parent
        else:
            if not dataset or '/' in dataset or '\\' in dataset or dataset in ('.', '..'):
                raise ValueError('Invalid dataset')
            folder = (self.outputs / STUDIES.get(dataset, (dataset,))[0]).resolve()
        if not folder.is_relative_to(self.root) or not folder.is_dir():
            raise FileNotFoundError('Report unavailable')
        return folder

    def catalog(self):
        items = []
        for name in ('massvis_sample', 'massvis_all', 'kylie', 'hayeong', 'arunkumar', *STUDIES):
            folder = self.outputs / STUDIES.get(name, (name,))[0]
            items.append(dict(id=name, label=TITLES.get(name, name.replace('_', ' ').title()),
                              available=(folder / 'selection.json').is_file()))
        return items

    def load(self, dataset, data=''):
        folder = self.folder(dataset, data)
        path = folder / 'selection.json'
        with self.lock:
            selection = json.loads(path.read_text())
            rows = report_rows(selection['rows'], folder, allow_partial=True)
            files = [path] + [folder / f'prediction_{r["index"]}{suffix}'
                             for r in rows for suffix in ('.npz', '.json')]
            fingerprint = hashlib.sha256(json.dumps([
                (str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files]).encode()).hexdigest()
            snapshot = self.snapshots.get(folder)
            if snapshot is None or snapshot['revision'] != fingerprint:
                maps, _ = load_predictions(rows, folder, selection['background_rgb'], selection.get('precision', 'fp32'))
                snapshot = dict(revision=fingerprint, rows=rows, maps=maps, selection=selection, folder=folder)
                self.snapshots[folder] = snapshot
            self.snapshots.move_to_end(folder)
            while len(self.snapshots) > 2:
                self.snapshots.popitem(last=False)
            selection = snapshot['selection']
            attributes = selection.get('attributes', attribute_metadata())
            title = TITLES.get(dataset) if not data else None
            title = title or selection.get('title') or folder.name
            selected_count = len(selection['rows'])
            if not data and dataset in STUDIES:
                parent, comparison_id = STUDIES[dataset]
                manifest = json.loads((self.root / f'tribe/studies/manifests/{parent}.json').read_text())
                comparison = next(c for c in manifest['comparisons'] if c['id'] == comparison_id)
                included = set(comparison['indices'])
                positions = [i for i, row in enumerate(snapshot['rows']) if row['index'] in included]
                rows = [snapshot['rows'][i] for i in positions]
                maps = snapshot['maps'][positions]
                attributes = comparison['attributes']
                title = comparison['title']
                selected_count = len(included)
            else:
                rows, maps = snapshot['rows'], snapshot['maps']
            if selection.get('dataset') == 'hayeong':
                included = {row['index'] for row in selection['rows'] if row['attributes']['experiment'] == 2}
                if not data and dataset in STUDIES:
                    included.intersection_update(comparison['indices'])
                positions = [i for i, row in enumerate(rows) if row['index'] in included]
                rows, maps = [rows[i] for i in positions], maps[positions]
                selected_count = len(included)
                attributes = [a for a in attributes if a['key'] != 'experiment']
            if not rows:
                raise FileNotFoundError('No completed predictions')
            rows, attributes = study_variables(selection.get('dataset'), rows, attributes)
            if selection.get('dataset', 'massvis') == 'massvis':
                rows, attributes = complexity_tertiles(rows, attributes, selection['rows'])
            public_rows = []
            prefix = '/' + folder.relative_to(self.root).as_posix()
            for row in rows:
                item = {k: v for k, v in row.items() if k not in ('source', 'sha256')}
                item['image'] = f'{prefix}/inputs/{row["index"]}.png'
                public_rows.append(item)
            revision = hashlib.sha256(json.dumps([
                fingerprint, attributes, [row['attributes'] for row in public_rows]], sort_keys=True).encode()).hexdigest()
            metadata = dict(title=title, dataset=selection.get('dataset', 'massvis'),
                            count=len(rows), selected_count=selected_count, complete=len(rows) == selected_count,
                            attributes=attributes, rows=public_rows, revision=revision,
                            selection=prefix + '/selection.json')
            return metadata, maps

    def analysis(self, dataset, data, x, y, statistic):
        metadata, maps = self.load(dataset, data)
        key = hashlib.sha256(json.dumps([metadata['revision'], dataset, data, x, y, statistic]).encode()).hexdigest()
        with self.lock:
            if key not in self.grids:
                grid = aggregate_xy(metadata['rows'], maps, metadata['attributes'], x, y, statistic)
                self.grids[key] = grid
            grid = self.grids[key]
            self.grids.move_to_end(key)
            while len(self.grids) > 4:
                self.grids.popitem(last=False)
        cells = [{k: v for k, v in cell.items() if k != 'values'} for cell in grid['cells']]
        return dict(id=key, revision=metadata['revision'], axes=grid['axes'], cells=cells, limit=grid['limit'])

    def render(self, grid_id, x, y, views, source=None):
        if views not in ('inferior', 'four', 'all'):
            raise ValueError('Invalid cortical view')
        digest = hashlib.sha256(json.dumps([grid_id, x, y, views, 'nilearn-v1']).encode()).hexdigest()
        cache = self.root / 'tribe/cache/visualizer' / f'{digest}.png'
        if cache.is_file():
            return cache.read_bytes()
        if grid_id not in self.grids and source:
            refreshed = self.analysis(source.get('dataset', 'massvis_sample'), source.get('data', ''),
                                      source.get('x_axis', ''), source.get('y_axis', ''), source.get('statistic', 'mean'))
            if refreshed['id'] != grid_id:
                raise ValueError('Predictions changed; refresh the grid')
        with self.lock:
            grid = self.grids.get(grid_id)
            if grid is None:
                raise ValueError('Analysis expired; refresh the grid')
            cell = next((c for c in grid['cells'] if str(c['x']) == x and str(c['y']) == y), None)
            if cell is None or not cell['count']:
                raise ValueError('Empty cell')
        with self.plot_lock:
            if not cache.is_file():
                # Nilearn/Matplotlib are loaded only when the browser requests a map.
                os.environ.setdefault('MPLCONFIGDIR', str(self.root / 'tribe/cache/matplotlib'))
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from nilearn import datasets
                from visualizer.cortical import brain
                fs = datasets.fetch_surf_fsaverage(mesh='fsaverage5', data_dir=str(self.root / 'tribe/cache/nilearn'))
                panels = ([('both', 'ventral')] if views == 'inferior' else
                          [('left', 'lateral'), ('left', 'medial'), ('right', 'lateral'), ('right', 'medial')]
                          + ([('both', 'ventral')] if views == 'all' else []))
                columns = 1 if views == 'inferior' else 2
                rows = (len(panels) + columns - 1) // columns
                fig = plt.figure(figsize=(3 * columns, 2.7 * rows), facecolor='white')
                try:
                    spec = fig.add_gridspec(rows, columns, hspace=0, wspace=0)
                    for i, (hemi, view) in enumerate(panels):
                        ax = brain(fig, spec[i // columns, i % columns], fs, cell['values'], hemi, view, grid['limit'])
                        if views == 'inferior':
                            ax.set_title('')
                    fig.subplots_adjust(left=0, right=1, top=.96, bottom=0)
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format='png', dpi=95, facecolor='white')
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache.with_suffix(f'.{os.getpid()}.tmp')
                    temporary.write_bytes(buffer.getvalue())
                    temporary.replace(cache)
                finally:
                    plt.close(fig)
            return cache.read_bytes()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, visualizer, **kwargs):
        self.visualizer = visualizer
        super().__init__(*args, directory=str(visualizer.root), **kwargs)

    def reply(self, value, status=200, mime='application/json'):
        body = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'private, max-age=86400' if mime == 'image/png' else 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def send_head(self):
        path = unquote(urlsplit(self.path).path)
        resolved = (self.visualizer.root / path.lstrip('/')).resolve()
        if any(p.startswith('.') for p in Path(path).parts) or not resolved.is_relative_to(self.visualizer.root):
            self.send_error(404)
            return None
        return super().send_head()

    def list_directory(self, path):
        self.send_error(404)
        return None

    def do_GET(self):
        url = urlsplit(self.path)
        if not url.path.startswith('/api/'):
            super().do_GET()
            return
        query = {key: values[0] for key, values in parse_qs(url.query, keep_blank_values=True).items()}
        try:
            if url.path == '/api/reports':
                self.reply(self.visualizer.catalog())
            elif url.path == '/api/report':
                self.reply(self.visualizer.load(query.get('dataset', 'massvis_sample'), query.get('data', ''))[0])
            elif url.path == '/api/analysis':
                self.reply(self.visualizer.analysis(query.get('dataset', 'massvis_sample'), query.get('data', ''),
                                                    query.get('x', ''), query.get('y', ''), query.get('statistic', 'mean')))
            elif url.path == '/api/map':
                self.reply(self.visualizer.render(query.get('id', ''), query.get('x', ''), query.get('y', ''),
                                                  query.get('views', 'inferior'), source=query), mime='image/png')
            else:
                self.reply(dict(error='Unknown endpoint'), 404)
        except (FileNotFoundError, KeyError, StopIteration):
            self.reply(dict(error='Predictions unavailable'), 404)
        except (ValueError, AssertionError) as error:
            self.reply(dict(error=str(error)), 400)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--open', action='store_true', dest='open_browser', help='Open the frontend in your browser')
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('--port must be between 0 and 65535 (0 selects a free port)')
    try:
        server = ThreadingHTTPServer((args.host, args.port), partial(Handler, visualizer=Visualizer()))
    except OSError as error:
        hint = ('Choose another port with ./start-visualizer.sh --port 8001.\n'
                if error.errno == errno.EADDRINUSE else '')
        parser.exit(1, f'Cannot start visualizer on {args.host}:{args.port}: {error}\n' + hint)
    host = '127.0.0.1' if args.host == '0.0.0.0' else args.host
    url = f'http://{host}:{server.server_port}/visualizer/'
    print(f'Visualizer: {url}', flush=True)
    print(f'API: http://{host}:{server.server_port}/api/reports\nPress Ctrl+C to stop.', flush=True)
    if args.open_browser:
        # Browser launch must not block the HTTP loop (e.g. a terminal browser).
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
