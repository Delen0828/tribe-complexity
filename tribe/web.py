"""Build a shared static viewer for reports under an outputs directory."""
import argparse
from html import escape
import json
from pathlib import Path
import shutil
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent


def page(title, content):
    return (f'<!doctype html><html lang="en"><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{escape(title)}</title><link rel="stylesheet" href="style.css">'
            f'<body><header><a href="index.html">TRIBE / Output reports</a></header>'
            f'<main><h1>{escape(title)}</h1>{content}</main></body></html>')


def redirect(path, target):
    path.write_text('<!doctype html><meta charset="utf-8">'
                    f'<meta http-equiv="refresh" content="0;url={escape(target, quote=True)}">'
                    f'<a href="{escape(target, quote=True)}">Open shared report viewer</a>')


def build_site(outputs):
    outputs = outputs.resolve()
    web = outputs / 'web'
    web.mkdir(parents=True, exist_ok=True)
    viewer = ROOT.parent / 'visualizer'
    for name in ('style.css', 'brain.css', 'brain.js'):
        shutil.copyfile(viewer / name, web / name)
    html = (viewer / 'brain.html').read_text().replace('<body>', '<body data-catalog="reports.json">')
    html = html.replace('href="./"', 'href="index.html"').replace('Explore annotations ↗', 'All output reports ↗')
    (web / 'brain.html').write_text(html)
    reports, cards = [], []
    for folder in sorted(outputs.iterdir()):
        if not folder.is_dir() or folder == web:
            continue
        name = folder.name
        prefix = '../' + quote(name, safe='') + '/'
        report_path = folder / 'explorer.json'
        target = None
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            partial = report.get('complete') is False
            label = f'{name} · {report["count"]:,} stimuli' + (' · partial' if partial else '')
            reports.append(dict(id=name, label=label, data=prefix + 'explorer.json'))
            target = 'brain.html?dataset=' + quote(name, safe='')
            detail = 'Grouped cortical means, contrasts, and member images.'
        elif (folder / 'spectrum.png').is_file():
            target = 'spectrum-' + quote(name, safe='') + '.html'
            content = ('<p>One stimulus per rating bin, with shared-scale cortical predictions at t=0.</p>'
                       f'<p><a href="{prefix}spectrum.pdf">Download PDF</a> · '
                       f'<a href="{prefix}selection.json">Selection</a></p>'
                       f'<img style="width:100%" src="{prefix}spectrum.png" alt="Complexity spectrum cortical maps">')
            (web / ('spectrum-' + name + '.html')).write_text(page(name, content))
            label, detail = name, 'Complexity spectrum.'
        else:
            detail = 'No visualization report generated.'
            if (folder / 'selection.json').is_file():
                selection = json.loads((folder / 'selection.json').read_text())
                detail += f' Selected stimuli: {len(selection.get("rows", [])):,}.'
            cards.append(f'<li><strong>{escape(name)}</strong><p>{escape(detail)}</p></li>')
            continue
        cards.append(f'<li><a href="{escape(target, quote=True)}">{escape(label)}</a><p>{detail}</p></li>')
        # Keep existing bookmarks usable, but store executable viewer assets only once.
        redirect(folder / 'index.html', '../web/' + target)
        for asset in ('brain.js', 'brain.css', 'style.css'):
            (folder / asset).unlink(missing_ok=True)
    if (outputs / 'comparison.png').is_file():
        content = ('<p>Original stimuli, t=0 cortical predictions, and signed contrasts.</p>'
                   '<p><a href="../comparison.pdf">Download PDF</a></p>'
                   '<img style="width:100%" src="../comparison.png" alt="Stimuli and cortical maps">'
                   '<img style="width:100%" src="../directional_contrasts.png" alt="Directional contrasts">')
        (web / 'comparison.html').write_text(page('MASSVIS 751 and 4849', content))
        cards.insert(0, '<li><a href="comparison.html">Two-stimulus comparison</a></li>')
    (web / 'reports.json').write_text(json.dumps(reports, indent=2) + '\n')
    (web / 'index.html').write_text(page('Output visualizations',
        '<p>Choose a report. Partial reports summarize only completed predictions.</p><ul>' + ''.join(cards) + '</ul>'))
    return web / 'index.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs-dir', type=Path, default=ROOT / 'outputs')
    print(build_site(parser.parse_args().outputs_dir))
