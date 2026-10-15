"""Lay out the built site for publishing as one claude.ai Artifact.

    python artifact_bundle.py --site <site dir from build_site.py + compare.html> --out <bundle dir>

The benchmark page becomes the artifact page (page.html, without its own document skeleton, which the Artifact
adds), the token viewer becomes viewer.html, and app.js, style.css and data/ go alongside. tables/*.csv stay local:
the Artifact frame blocks downloads and spans.csv alone is over the 16 MB per-file limit. Prints the publish
batches (each under the 64 MB per-publish limit) as JSON lists of paths.
"""
import argparse
import json
import os
import re
import shutil

FILE_LIMIT = 16 * 1024 * 1024
BATCH_LIMIT = 60 * 1024 * 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'data', 'traj'), exist_ok=True)

    page = open(os.path.join(a.site, 'compare.html')).read()
    page = re.sub(r'^<!doctype html><html[^>]*><head>(<meta[^>]*>)*', '', page)
    page = page.replace('</head><body>', '\n', 1)
    page = re.sub(r'</body></html>\s*$', '\n', page)
    page = page.replace('href="index.html', 'href="viewer.html')
    assert page.lstrip().startswith('<title>'), page[:200]
    open(os.path.join(a.out, 'page.html'), 'w').write(page)

    viewer = open(os.path.join(a.site, 'index.html')).read()
    assert viewer.count('href="compare.html"') == 1
    open(os.path.join(a.out, 'viewer.html'), 'w').write(viewer.replace('href="compare.html"', 'href="./"'))

    files = ['viewer.html', 'app.js', 'style.css', 'data/index.json']
    files += ['data/traj/' + f for f in sorted(os.listdir(os.path.join(a.site, 'data', 'traj')))]
    for f in files[1:]:
        if f.endswith('.json'):
            # token pieces that split a UTF-8 character decode to U+FFFD, which the Artifact publisher refuses in
            # text files; the JSON escape parses back to the same string
            text = open(os.path.join(a.site, f), encoding='utf-8').read()
            open(os.path.join(a.out, f), 'w', encoding='utf-8').write(text.replace('\ufffd', '\\ufffd'))
        else:
            shutil.copy(os.path.join(a.site, f), os.path.join(a.out, f))

    batches, cur, size = [], [], 0
    for f in files:
        n = os.path.getsize(os.path.join(a.out, f))
        assert n <= FILE_LIMIT, f'{f}: {n} bytes is over the per-file limit'
        if cur and size + n > BATCH_LIMIT:
            batches.append(cur)
            cur, size = [], 0
        cur.append(f)
        size += n
    batches.append(cur)
    print(json.dumps(batches))


if __name__ == '__main__':
    main()
