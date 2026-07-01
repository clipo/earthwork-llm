"""Detection-review web app: see what the pipeline identified, validate each one.

Serves the relief thumbnails built by scripts/build_review.py and records a
verdict per candidate (confirmed mound / rejected / uncertain) with an optional
note and reviewer initials. Verdicts persist to:

    data/review/{aoi}/verdicts.csv

Pure Python standard library — no Flask, no build step.

Usage:
    python scripts/review_server.py            # http://localhost:8765
    python scripts/review_server.py --port 9000
"""
from __future__ import annotations
import csv, json, sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

REVIEW = Path("data/review")
VERDICT_COLS = ["aoi", "id", "source", "score", "utm_x", "utm_y", "lat", "lon",
                "lrm_wide_m", "verdict", "note", "reviewer", "ts"]


def list_aois():
    out = []
    if not REVIEW.exists():
        return out
    for d in sorted(REVIEW.iterdir()):
        cj = d / "candidates.json"
        if cj.exists():
            doc = json.loads(cj.read_text())
            vd = load_verdicts(d.name)
            done = sum(1 for v in vd.values() if v.get("verdict"))
            out.append({"aoi": d.name, "count": doc["count"], "reviewed": done})
    return out


def load_candidates(aoi: str):
    doc = json.loads((REVIEW / aoi / "candidates.json").read_text())
    verdicts = load_verdicts(aoi)
    for c in doc["candidates"]:
        v = verdicts.get(c["id"], {})
        c["verdict"] = v.get("verdict", "")
        c["note"] = v.get("note", "")
        c["reviewer"] = v.get("reviewer", "")
    return doc


def verdict_path(aoi: str) -> Path:
    return REVIEW / aoi / "verdicts.csv"


def load_verdicts(aoi: str) -> dict:
    p = verdict_path(aoi)
    if not p.exists():
        return {}
    out = {}
    with p.open() as fh:
        for row in csv.DictReader(fh):
            out[row["id"]] = row
    return out


def save_verdict(rec: dict):
    aoi = rec["aoi"]
    verdicts = load_verdicts(aoi)
    cand = {c["id"]: c for c in json.loads((REVIEW / aoi / "candidates.json").read_text())["candidates"]}
    base = cand.get(rec["id"], {})
    row = {k: base.get(k, "") for k in VERDICT_COLS}
    row.update({"aoi": aoi, "id": rec["id"], "verdict": rec.get("verdict", ""),
                "note": rec.get("note", ""), "reviewer": rec.get("reviewer", ""),
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    verdicts[rec["id"]] = row
    p = verdict_path(aoi)
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=VERDICT_COLS)
        w.writeheader()
        for r in verdicts.values():
            w.writerow({k: r.get(k, "") for k in VERDICT_COLS})


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Yazoo detection review</title>
<style>
 :root{--bg:#11151c;--panel:#1b2230;--ink:#e7edf5;--mut:#8aa0b8;--ok:#2ecc71;--no:#e74c3c;--maybe:#f1c40f;--acc:#4aa3ff}
 *{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--ink)}
 header{display:flex;gap:16px;align-items:center;padding:10px 16px;background:var(--panel);position:sticky;top:0;z-index:5;flex-wrap:wrap}
 select,input,button{font:inherit;background:#0d1117;color:var(--ink);border:1px solid #2b3648;border-radius:6px;padding:6px 10px}
 button{cursor:pointer}.spacer{flex:1}
 .bar{height:8px;background:#0d1117;border-radius:6px;overflow:hidden;width:220px}
 .bar>div{height:100%;background:var(--acc)}
 main{display:grid;grid-template-columns:300px 1fr;gap:0;height:calc(100vh - 53px)}
 #list{overflow:auto;border-right:1px solid #2b3648}
 .row{display:flex;gap:8px;align-items:center;padding:8px 12px;border-bottom:1px solid #222c3a;cursor:pointer}
 .row:hover{background:#161d29}.row.sel{background:#1f2a3b}
 .dot{width:10px;height:10px;border-radius:50%;background:#3a4555;flex:none}
 .dot.confirmed{background:var(--ok)}.dot.rejected{background:var(--no)}.dot.uncertain{background:var(--maybe)}
 .rid{font-weight:600}.rs{color:var(--mut);font-size:12px}
 #detail{overflow:auto;padding:18px 22px}
 #thumb{width:100%;max-width:1000px;background:#0d1117;border:1px solid #2b3648;border-radius:8px}
 .meta{color:var(--mut);margin:10px 0}.meta b{color:var(--ink)}
 .verbtns{display:flex;gap:10px;margin:14px 0}
 .verbtns button{font-size:15px;padding:10px 18px}
 .vc{border-color:var(--ok)}.vc.on{background:var(--ok);color:#06240f}
 .vr{border-color:var(--no)}.vr.on{background:var(--no);color:#2a0a06}
 .vu{border-color:var(--maybe)}.vu.on{background:var(--maybe);color:#2a2406}
 #note{width:100%;max-width:1000px}
 kbd{background:#0d1117;border:1px solid #2b3648;border-radius:4px;padding:1px 6px;font-size:12px}
 a{color:var(--acc)}
</style></head><body>
<header>
 <strong>Yazoo detection review</strong>
 <select id=aoi></select>
 <label>reviewer <input id=rev size=6 placeholder="init"></label>
 <select id=filter>
   <option value=all>all</option><option value=todo>unreviewed</option>
   <option value=confirmed>confirmed</option><option value=rejected>rejected</option>
   <option value=uncertain>uncertain</option>
 </select>
 <div class=spacer></div>
 <span id=tally class=rs></span>
 <div class=bar><div id=barfill style=width:0></div></div>
</header>
<main>
 <div id=list></div>
 <div id=detail><p class=rs>Pick an AOI.</p></div>
</main>
<script>
const $=s=>document.querySelector(s);
let AOI=null, CANDS=[], SEL=null;
async function loadAois(){
  const a=await (await fetch('/api/aois')).json();
  $('#aoi').innerHTML=a.map(x=>`<option value="${x.aoi}">${x.aoi} (${x.reviewed}/${x.count})</option>`).join('');
  if(a.length){AOI=a[0].aoi; loadCands();}
}
async function loadCands(){
  AOI=$('#aoi').value;
  const d=await (await fetch('/api/candidates/'+AOI)).json();
  CANDS=d.candidates; renderList(); if(CANDS.length) select(0);
}
function visible(){
  const f=$('#filter').value;
  return CANDS.filter(c=> f=='all'?1 : f=='todo'?!c.verdict : c.verdict==f);
}
function renderList(){
  const vis=visible();
  $('#list').innerHTML=vis.map((c,i)=>`<div class=row data-id="${c.id}" onclick="selectId('${c.id}')">
    <span class="dot ${c.verdict}"></span>
    <span class=rid>${c.id}</span>
    <span class=rs>${c.source} · ${c.score} · ${c.lrm_wide_m??'?'}m</span></div>`).join('')
    || '<p class=rs style=padding:12px>none</p>';
  const done=CANDS.filter(c=>c.verdict).length;
  $('#tally').textContent=`${done}/${CANDS.length} reviewed`;
  $('#barfill').style.width=(CANDS.length?100*done/CANDS.length:0)+'%';
  markSel();
}
function markSel(){document.querySelectorAll('.row').forEach(r=>r.classList.toggle('sel',r.dataset.id===(SEL&&SEL.id)));}
function selectId(id){const i=CANDS.findIndex(c=>c.id===id); if(i>=0)select(i);}
function select(i){
  SEL=CANDS[i];
  const c=SEL;
  const llm = c.source=='terrallm' ? `<div class=meta style="background:#161d29;padding:10px 12px;border-radius:6px">
     <b>TerraLLM</b> · shield <b>${c.shield_decision||'?'}</b> · height <b>${c.height_m??'?'} m</b>
     · area <b>${c.area_m2??'?'} m²</b> · NLCD ${c.nlcd_class||'?'}
     ${c.justification?`<br><b>justification:</b> ${c.justification}`:''}
     ${c.llm_analysis && c.llm_analysis!='N/A'?`<br><b>LLM:</b> ${c.llm_analysis}`:''}</div>` : '';
  $('#detail').innerHTML=`<img id=thumb src="/thumbs/${AOI}/${c.id}.png">
   <div class=meta><b>${c.id}</b> · source <b>${c.source}</b> · score <b>${c.score}</b>
     · wide-LRM <b>${c.lrm_wide_m??'?'} m</b>${c.diameter_m?` · ⌀ <b>${c.diameter_m} m</b>`:''}
     <br>UTM15N <b>${c.utm_x.toFixed(0)}, ${c.utm_y.toFixed(0)}</b>
     · <a href="https://www.google.com/maps/search/?api=1&query=${c.lat},${c.lon}" target=_blank>map ${c.lat}, ${c.lon}</a></div>
   ${llm}
   <div class=verbtns>
     <button class="vc ${c.verdict=='confirmed'?'on':''}" onclick="vote('confirmed')">✓ Mound <kbd>C</kbd></button>
     <button class="vr ${c.verdict=='rejected'?'on':''}" onclick="vote('rejected')">✗ Not <kbd>X</kbd></button>
     <button class="vu ${c.verdict=='uncertain'?'on':''}" onclick="vote('uncertain')">? Uncertain <kbd>U</kbd></button>
   </div>
   <input id=note placeholder="note (optional)" value="${(c.note||'').replace(/"/g,'&quot;')}"
      onchange="saveNote(this.value)">
   <p class=rs>Keys: <kbd>C</kbd>/<kbd>X</kbd>/<kbd>U</kbd> verdict · <kbd>J</kbd>/<kbd>K</kbd> next/prev</p>`;
  markSel();
}
async function post(rec){
  await fetch('/api/verdict',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify(rec)});
}
async function vote(v){
  if(!SEL)return; SEL.verdict=v; SEL.reviewer=$('#rev').value;
  await post({aoi:AOI,id:SEL.id,verdict:v,note:SEL.note||'',reviewer:$('#rev').value});
  renderList(); next();
}
async function saveNote(t){ if(!SEL)return; SEL.note=t;
  await post({aoi:AOI,id:SEL.id,verdict:SEL.verdict||'',note:t,reviewer:$('#rev').value}); }
function step(d){const vis=visible(); if(!vis.length)return;
  let i=vis.findIndex(c=>c.id===(SEL&&SEL.id)); i=Math.max(0,Math.min(vis.length-1,(i<0?0:i)+d));
  selectId(vis[i].id); $('#detail').scrollTop=0;}
function next(){step(1)}
document.addEventListener('keydown',e=>{
  if(e.target.tagName=='INPUT')return;
  const k=e.key.toLowerCase();
  if(k=='c')vote('confirmed'); else if(k=='x')vote('rejected'); else if(k=='u')vote('uncertain');
  else if(k=='j')step(1); else if(k=='k')step(-1);
});
$('#aoi').onchange=loadCands; $('#filter').onchange=renderList;
loadAois();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        try:
            if p in ("/", "/index.html"):
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if p == "/api/aois":
                return self._send(200, list_aois())
            if p.startswith("/api/candidates/"):
                return self._send(200, load_candidates(p.rsplit("/", 1)[1]))
            if p.startswith("/thumbs/"):
                _, _, aoi, fn = p.split("/", 3)
                fp = REVIEW / aoi / "thumbs" / fn
                if fp.exists():
                    return self._send(200, fp.read_bytes(), "image/png")
                return self._send(404, {"error": "no thumb"})
            return self._send(404, {"error": "not found"})
        except Exception as e:
            return self._send(500, {"error": str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/verdict":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        rec = json.loads(self.rfile.read(n) or b"{}")
        try:
            save_verdict(rec)
            return self._send(200, {"ok": True})
        except Exception as e:
            return self._send(500, {"error": str(e)})


def main():
    port = 8765
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"Review UI → http://localhost:{port}/   (Ctrl-C to stop)")
    print(f"AOIs available: {[a['aoi'] for a in list_aois()] or 'none — run build_review.py first'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
