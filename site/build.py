"""The GitHub Pages site: site/index.html, the pages in pages/ (each wrapped in a full HTML document, since they are
written as page bodies) and the figures. Run by .github/workflows/pages.yml; locally:
  python3 site/build.py   -> _site/"""
import shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "_site"
HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
        'initial-scale=1,viewport-fit=cover"><style>body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}'
        '[hidden]{display:none!important}</style></head><body>\n')
if OUT.exists(): shutil.rmtree(OUT)
(OUT / "figures").mkdir(parents=True)
shutil.copy2(ROOT / "site" / "index.html", OUT / "index.html")
for p in sorted((ROOT / "pages").glob("*.html")):
    s = p.read_text()
    (OUT / p.name).write_text(s if s.lstrip().lower().startswith("<!doctype") else HEAD + s + "\n</body></html>\n")
for p in sorted((ROOT / "figures").glob("*.png")): shutil.copy2(p, OUT / "figures" / p.name)
(OUT / ".nojekyll").write_text("")
print("built", OUT, sorted(x.name for x in OUT.iterdir()))
