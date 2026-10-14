"""Keep old output URLs pointing to the shared visualizer."""
import argparse
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent


def redirect(path, target, preserve_query=False):
    script = ''
    if preserve_query:
        script = ('<script>const p = new URLSearchParams(location.search);'
                  'if(p.has("data")) p.set("data",new URL(p.get("data"),location.href).pathname);'
                  f'location.replace({json.dumps(target)} + "?" + p);</script>')
    path.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>TRIBE</title>'
                    + script + ('' if preserve_query else
                    f'<meta http-equiv="refresh" content="0;url={escape(target, quote=True)}">')
                    + f'<a href="{escape(target, quote=True)}">Open visualizer</a></html>')


def build_site(outputs):
    outputs = outputs.resolve()
    web = outputs / 'web'
    web.mkdir(parents=True, exist_ok=True)
    viewer = ROOT.parent / 'visualizer'
    home = Path(os.path.relpath(viewer, web)).as_posix() + '/'
    redirect(web / 'index.html', home)
    redirect(web / 'brain.html', home + 'brain.html', preserve_query=True)
    reports = []
    for folder in sorted(outputs.iterdir()):
        if not folder.is_dir() or folder == web:
            continue
        report_path = folder / 'explorer.json'
        selection_path = folder / 'selection.json'
        if not report_path.is_file() and not selection_path.is_file():
            continue
        report = json.loads((report_path if report_path.is_file() else selection_path).read_text())
        count = report.get('count', len(report.get('rows', [])))
        label = (report.get('title') or folder.name) + f' · {count:,}'
        if report.get('complete') is False:
            label += ' · partial'
        reports.append(dict(id=folder.name, label=label,
                            data='../' + quote(folder.name, safe='') + '/explorer.json'))
        # Absolute URL paths are used only for files inside the repository.
        source = selection_path if selection_path.is_file() else report_path
        data = '/' + source.relative_to(ROOT.parent).as_posix() if source.is_relative_to(ROOT.parent) else str(source)
        target = home + 'brain.html?data=' + quote(data, safe='')
        redirect(web / ('study-' + folder.name + '.html'), target)
        relative_viewer = Path(os.path.relpath(viewer, folder)).as_posix() + '/brain.html'
        redirect(folder / 'index.html', relative_viewer + '?data=' + quote(data, safe=''))
        for asset in ('brain.js', 'brain.css', 'style.css'):
            (folder / asset).unlink(missing_ok=True)
    (web / 'reports.json').write_text(json.dumps(reports, indent=2) + '\n')
    for asset in ('brain.js', 'brain.css', 'style.css'):
        (web / asset).unlink(missing_ok=True)
    return web / 'index.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs-dir', type=Path, default=ROOT / 'outputs')
    print(build_site(parser.parse_args().outputs_dir))
