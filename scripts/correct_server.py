"""Coordinate-correction web tool: click the actual mound summit on the relief.

Shows each site's relief (hillshade + wide-LRM) with the published point marked.
The analyst clicks the true mound summit; the click pixel is converted to a
precise UTM15N / WGS84 coordinate and saved to:

    data/correct/corrected_coords.csv

Pure standard library. Run on the machine that built data/correct/ (build_correct.py).

Usage:
    python scripts/correct_server.py            # http://localhost:8787
"""
from __future__ import annotations
import csv
import json
import sys
import math
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

CORR = Path("data/correct")
COLS = ["id", "label", "pub_utm_x", "pub_utm_y", "corr_utm_x", "corr_utm_y",
        "corr_lat", "corr_lon", "shift_m", "reviewer", "ts"]

try:
    from pyproj import Transformer
    _TO_WGS = Transformer.from_crs("EPSG:26915", "EPSG:4326", always_xy=True)
    def to_wgs(x, y): return _TO_WGS.transform(x, y)
except Exception:
    def to_wgs(x, y): return (None, None)


def sites():
    return json.loads((CORR / "sites.json").read_text())


def load_corr():
    p = CORR / "corrected_coords.csv"
    if not p.exists():
        return {}
    with p.open() as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


def save_corr(rec):
    site = {s["id"]: s for s in sites()}[rec["id"]]
    # click fractional -> UTM
    fx, fy = float(rec["fx"]), float(rec["fy"])
    x = site["utm_xmin"] + fx * site["w"] * site["res"]
    y = site["utm_ymax"] - fy * site["h"] * site["res"]
    lon, lat = to_wgs(x, y)
    shift = math.hypot(x - site["pub_utm_x"], y - site["pub_utm_y"])
    row = {"id": rec["id"], "label": site["label"],
           "pub_utm_x": site["pub_utm_x"], "pub_utm_y": site["pub_utm_y"],
           "corr_utm_x": round(x, 1), "corr_utm_y": round(y, 1),
           "corr_lat": round(lat, 6) if lat else "", "corr_lon": round(lon, 6) if lon else "",
           "shift_m": round(shift, 1), "reviewer": rec.get("reviewer", ""),
           "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    cur = load_corr()
    cur[rec["id"]] = row
    with (CORR / "corrected_coords.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in cur.values():
            w.writerow({k: r.get(k, "") for k in COLS})
    return row


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Correct mound coordinates</title>
<style>
 body{margin:0;font:14px system-ui,sans-serif;background:#10141b;color:#e7edf5}
 header{display:flex;gap:14px;align-items:center;padding:10px 16px;background:#1b2230;flex-wrap:wrap}
 select,input,button{font:inherit;background:#0d1117;color:#e7edf5;border:1px solid #2b3648;border-radius:6px;padding:6px 10px}
 button{cursor:pointer}.spacer{flex:1}
 main{display:flex;gap:16px;padding:16px;align-items:flex-start}
 .imgwrap{position:relative;display:inline-block;border:1px solid #2b3648;border-radius:8px;overflow:hidden}
 .imgwrap img{display:block;max-width:74vh;cursor:crosshair}
 .mk{position:absolute;transform:translate(-50%,-50%);pointer-events:none;font-size:22px;line-height:1}
 .pub{color:#4aa3ff} .corr{color:#2ecc71}
 aside{min-width:260px}.row{margin:8px 0}.k{color:#8aa0b8}
 b.green{color:#2ecc71}
 kbd{background:#0d1117;border:1px solid #2b3648;border-radius:4px;padding:1px 6px}
</style></head><body>
<header>
 <strong>Correct mound summit</strong>
 <select id=site></select>
 <label>reviewer <input id=rev size=6></label>
 <span class=spacer></span>
 <button onclick="save()">Save summit (S)</button>
 <button onclick="reset()">Reset</button>
 <span id=status class=k></span>
</header>
<main>
 <div class=imgwrap id=wrap><img id=img onclick="click_img(event)">
   <div class="mk pub" id=pub>✚</div>
   <div class="mk corr" id=corr style=display:none>●</div>
 </div>
 <aside>
   <h3 id=label></h3>
   <div class=row><span class=k>published UTM15N</span><br><span id=pubutm></span></div>
   <div class=row><span class=k>clicked summit</span><br><b class=green id=clk>— click the mound —</b></div>
   <div class=row><span class=k>shift from published</span> <b id=shift>—</b></div>
   <div class=row><span class=k>saved</span> <span id=saved></span></div>
   <p class=k>Blue ✚ = published point · Green ● = your click. Click the center of the mound
      (the compact warm/red rise), then Save. <kbd>S</kbd> save · <kbd>J</kbd>/<kbd>K</kbd> next/prev.</p>
 </aside>
</main>
<script>
const $=s=>document.querySelector(s); let SITES=[],S=null,click=null;
async function load(){ SITES=await (await fetch('/api/sites')).json();
  $('#site').innerHTML=SITES.map(s=>`<option value="${s.id}">${s.label}</option>`).join('');
  pick(SITES[0].id);}
function pick(id){ S=SITES.find(s=>s.id===id); click=null;
  $('#img').src='/'+S.img+'?t='+Date.now();
  $('#label').textContent=S.label;
  $('#pubutm').textContent=S.pub_utm_x.toFixed(0)+', '+S.pub_utm_y.toFixed(0)+'  ('+S.pub_lat+', '+S.pub_lon+')';
  $('#clk').textContent='— click the mound —'; $('#shift').textContent='—';
  $('#corr').style.display='none';
  $('#img').onload=()=>placePub();
  // existing correction?
  fetch('/api/corrected/'+S.id).then(r=>r.json()).then(c=>{
    $('#saved').textContent=c&&c.corr_utm_x?`${c.corr_utm_x}, ${c.corr_utm_y} (shift ${c.shift_m} m)`:'(none)'; });
}
function placePub(){ const im=$('#img'); const w=im.clientWidth,h=im.clientHeight;
  $('#pub').style.left=(S.pub_fx*w)+'px'; $('#pub').style.top=(S.pub_fy*h)+'px'; }
function click_img(e){ const im=$('#img'); const r=im.getBoundingClientRect();
  const fx=(e.clientX-r.left)/r.width, fy=(e.clientY-r.top)/r.height; click={fx,fy};
  $('#corr').style.left=(fx*im.clientWidth)+'px'; $('#corr').style.top=(fy*im.clientHeight)+'px';
  $('#corr').style.display='block';
  const x=S.utm_xmin+fx*S.w*S.res, y=S.utm_ymax-fy*S.h*S.res;
  const sh=Math.hypot(x-S.pub_utm_x,y-S.pub_utm_y);
  $('#clk').textContent=x.toFixed(0)+', '+y.toFixed(0); $('#shift').textContent=sh.toFixed(1)+' m'; }
async function save(){ if(!click){$('#status').textContent='click the mound first';return;}
  const r=await (await fetch('/api/correct',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({id:S.id,fx:click.fx,fy:click.fy,reviewer:$('#rev').value})})).json();
  $('#saved').textContent=`${r.corr_utm_x}, ${r.corr_utm_y} (shift ${r.shift_m} m)`;
  $('#status').textContent='saved ✓'; setTimeout(()=>$('#status').textContent='',1500); }
function reset(){ click=null; $('#corr').style.display='none'; $('#clk').textContent='— click the mound —'; $('#shift').textContent='—'; }
function step(d){ const i=SITES.findIndex(s=>s.id===S.id); const j=Math.max(0,Math.min(SITES.length-1,i+d));
  $('#site').value=SITES[j].id; pick(SITES[j].id); }
$('#site').onchange=e=>pick(e.target.value);
window.addEventListener('resize',()=>{placePub(); if(click){const im=$('#img');$('#corr').style.left=(click.fx*im.clientWidth)+'px';$('#corr').style.top=(click.fy*im.clientHeight)+'px';}});
document.addEventListener('keydown',e=>{ if(e.target.tagName==='INPUT')return;
  const k=e.key.toLowerCase(); if(k==='s')save(); else if(k==='j')step(1); else if(k==='k')step(-1); });
load();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _s(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass
    def do_GET(self):
        p = urlparse(self.path).path
        try:
            if p in ("/", "/index.html"):
                return self._s(200, PAGE, "text/html; charset=utf-8")
            if p == "/api/sites":
                return self._s(200, sites())
            if p.startswith("/api/corrected/"):
                return self._s(200, load_corr().get(p.rsplit("/", 1)[1], {}))
            if p.startswith("/img/"):
                fp = CORR / p.lstrip("/")
                if fp.exists():
                    return self._s(200, fp.read_bytes(), "image/png")
                return self._s(404, {"error": "no image"})
            return self._s(404, {"error": "not found"})
        except Exception as e:
            return self._s(500, {"error": str(e)})
    def do_POST(self):
        if urlparse(self.path).path != "/api/correct":
            return self._s(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        rec = json.loads(self.rfile.read(n) or b"{}")
        try:
            return self._s(200, save_corr(rec))
        except Exception as e:
            return self._s(500, {"error": str(e)})


def main():
    port = 8787
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    print(f"Correction UI -> http://localhost:{port}/   sites: {[s['id'] for s in sites()]}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    main()
