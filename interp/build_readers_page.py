"""The reader-heads walkthrough page: interp/readers_page_template.html with the figures of figures/fig_readers.py embedded.
  uv run --no-project interp/build_readers_page.py   -> results/readers/reader_heads.html (not on the site; move to pages/ to publish it there)"""
import base64, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
s = (ROOT / "interp" / "readers_page_template.html").read_text()
s = re.sub(r"/\*IMG:(\w+)\*/", lambda m: "data:image/png;base64," + base64.b64encode((ROOT / "figures" / "readers" / f"{m.group(1)}.png").read_bytes()).decode(), s)
(ROOT / "results" / "readers" / "reader_heads.html").write_text(s); print("wrote results/readers/reader_heads.html", f"{len(s) / 1e6:.1f} MB")
