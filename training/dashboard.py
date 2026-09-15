#!/usr/bin/env python3
"""crforge training dashboard — stdlib only, serves on 0.0.0.0:8099.

Reads ~/runs/*.log (and a few marker files), parses progress lines and SB3
tables, exposes /api/status + /api/series + / and renders a dark HTML page
that auto-refreshes. Built for a Tailscale-only audience.
"""
import json
import os
import re
import subprocess
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

HOME = Path.home()
RUNS = HOME / "runs"
BC = HOME / "bc"
PORT = 8099

EP_RE = re.compile(
    r"\[(\d+)/(\d+) \(\s*(\d+)%\) \| ep (\d+)\] win=(\d+)% loss=(\d+)% draw=(\d+)% \| reward=(-?[\d.]+)"
)
KV_RE = re.compile(r"^\|\s+([A-Za-z_/]+)\s+\|\s+(-?[\d.eE+]+)\s*\|\s*$")
PERDECK_RE = re.compile(r"per-deck \(worker 0\.\.N-1\): (.+)$")
SAVE_RE = re.compile(r"--save\s+(\S+)")

_STAT = {"prev": None}


def read_tail(path: Path, max_bytes: int = 700_000) -> list[str]:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []


def parse_run(name: str) -> dict:
    path = RUNS / f"{name}.log"
    lines = read_tail(path)
    eps: list[list] = []
    tables: list[dict] = []
    perdeck: list[str] = []
    cur: dict = {}
    for ln in lines:
        m = EP_RE.search(ln)
        if m:
            eps.append([int(m.group(4)), int(m.group(5)), float(m.group(8)), int(m.group(1)), int(m.group(2))])
        m = PERDECK_RE.search(ln)
        if m:
            perdeck.append(m.group(1).strip())
            perdeck = perdeck[-7:]
        m = KV_RE.match(ln)
        if m:
            try:
                cur[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
            if m.group(1) == "value_loss":
                tables.append(cur)
                cur = {}
    mtime = path.stat().st_mtime if path.exists() else 0
    tail80 = "\n".join(lines[-80:])
    status = "pending"
    if path.exists():
        if proc_for(name):
            status = "running"
        elif "Training done in" in tail80 or "Done!" in tail80:
            status = "done"
        else:
            status = "stopped"
    return {
        "name": name,
        "exists": path.exists(),
        "mtime": mtime,
        "status": status,
        "eps": eps,
        "tables": tables,
        "perdeck": perdeck,
        "last": eps[-1] if eps else None,
        "tail": lines[-12:],
    }


PROC_CACHE: dict = {"ts": 0, "procs": []}


def procs() -> list[dict]:
    now = time.time()
    if now - PROC_CACHE["ts"] < 4:
        return PROC_CACHE["procs"]
    out = []
    try:
        ps = subprocess.run(
            ["ps", "-eo", "pid,etimes,args"], capture_output=True, text=True, timeout=10
        ).stdout
        for ln in ps.splitlines():
            if "multi_selfplay_train.py" not in ln and "pretrain_bc" not in ln and "train_ppo" not in ln and "collect_" not in ln:
                continue
            parts = ln.split(None, 2)
            if len(parts) < 3:
                continue
            pid, etimes, args = parts
            if "grep" in args or "dashboard" in args:
                continue
            exe = args.split()[0] if args.split() else ""
            if "python" not in exe:  # skip bash wrappers etc.
                continue
            m = SAVE_RE.search(args)
            run_name = Path(m.group(1)).parent.parent.name if m else ""
            out.append(
                {
                    "pid": pid,
                    "etimes": int(etimes),
                    "run": run_name,
                    "cmd": args[:220],
                    "kind": (
                        "selfplay" if "multi_selfplay" in args
                        else "bc" if "pretrain_bc" in args or "collect_" in args
                        else "ppo"
                    ),
                }
            )
    except Exception:
        pass
    PROC_CACHE.update(ts=now, procs=out)
    return out


def proc_for(run_name: str) -> bool:
    return any(p["run"] == run_name for p in procs())


def host_stats() -> dict:
    try:
        vals = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        cur = [int(v) for v in vals[:8]]
        idle = cur[3] + cur[4]
        total = sum(cur)
        stat = {"idle": idle, "total": total}
        prev = _STAT["prev"]
        cpu_pct = None
        if prev and total > prev["total"]:
            dt = total - prev["total"]
            di = idle - prev["idle"]
            cpu_pct = round(100.0 * (dt - di) / dt, 1)
        _STAT["prev"] = stat
    except Exception:
        cpu_pct = None
    mem = {}
    try:
        mi = {}
        for ln in Path("/proc/meminfo").read_text().splitlines():
            k, v = ln.split(":", 1)
            mi[k] = int(v.strip().split()[0])
        mem = {
            "total_gb": round(mi["MemTotal"] / 1048576, 1),
            "used_gb": round((mi["MemTotal"] - mi["MemAvailable"]) / 1048576, 1),
            "pct": round(100.0 * (mi["MemTotal"] - mi["MemAvailable"]) / mi["MemTotal"], 0),
        }
    except Exception:
        pass
    try:
        up = float(Path("/proc/uptime").read_text().split()[0])
        la = Path("/proc/loadavg").read_text().split()[:3]
    except Exception:
        up, la = 0, []
    return {"cpu_pct": cpu_pct, "mem": mem, "uptime_h": round(up / 3600, 1), "loadavg": la, "cpus": os.cpu_count()}


def build_status() -> dict:
    names = ["fixed_bar", "mirror10m", "warm_pool", "pool12", "ft_bc"]
    runs = {n: parse_run(n) for n in names}
    active = None
    for p in procs():
        if p["run"]:
            active = p
            break
    active_run = active["run"] if active else None
    if not active_run:
        cands = [(r["mtime"], n) for n, r in runs.items() if r["exists"]]
        active_run = max(cands)[1] if cands else None

    phase = None
    if active_run and active_run in runs:
        r = runs[active_run]
        last = r["last"]
        tb = r["tables"][-1] if r["tables"] else {}
        fps = tb.get("fps")
        steps = last[3] if last else None
        total = last[4] if last and len(last) > 4 else None
        if total is None:
            for ln in r["tail"]:
                m = re.search(r"\[(\d+)/(\d+)", ln)
                if m:
                    steps, total = int(m.group(1)), int(m.group(2))
        eta = None
        if steps and total and fps and fps > 0:
            eta = int((total - steps) / fps)
        phase = {
            "run": active_run,
            "kind": active["kind"] if active else "unknown",
            "pid": active["pid"] if active else None,
            "elapsed_min": round(active["etimes"] / 60, 1) if active else None,
            "steps": steps,
            "total": total,
            "win": last[1] if last else None,
            "loss": 100 - last[1] if last else None,
            "ep": last[0] if last else None,
            "reward": last[2] if last else None,
            "fps": fps,
            "eta_s": eta,
            "entropy": tb.get("entropy_loss"),
            "ev": tb.get("explained_variance"),
            "kl": tb.get("approx_kl"),
            "perdeck": r["perdeck"],
        }

    def d(name):
        r = runs.get(name)
        return r["status"] if r else "pending"

    bc_data = False
    if BC.exists():
        bc_data = any(BC.glob("data/*"))
    scale_log = HOME / "scaling_test.log"
    scale_txt = scale_log.read_text() if scale_log.exists() else ""
    scale_status = "pending"
    if "previous run finished" in scale_txt:
        scale_status = "done" if "SCALE-DONE" in scale_txt else "running"
    elif scale_txt:
        scale_status = "pending"  # watcher is queued, still waiting
    ctrl_log = HOME / "control_evals.log"
    ctrl_txt = ctrl_log.read_text() if ctrl_log.exists() else ""
    ctrl_status = "pending"
    if "bridge up" in ctrl_txt:
        ctrl_status = "done" if "CONTROL-EVALS-DONE" in ctrl_txt else "running"
    queue = [
        {"name": "L2 · hog vs FAST rule_based (giant) — 10M", "status": d("fixed_bar")},
        {"name": "L1 · mirror hog vs hog — 10M", "status": d("mirror10m")},
        {"name": "Skalertest · envs 7/12/16 (fps)", "status": scale_status},
        {"name": "Kontrolleval · fersk vs trent (læringsbevis)", "status": ctrl_status},
        {"name": "Curriculum varm-start · pool hog,yard,lava — 20M", "status": d("warm_pool")},
        {"name": "Pool-12 · rotasjon + eval/2M (varmstart)", "status": d("pool12")},
        {"name": "BC-1 · heuristikk-datainnsamling", "status": "done" if bc_data else ("running" if any(p["kind"] == "bc" for p in procs()) else "pending")},
        {"name": "BC-2 · behavior cloning-trening", "status": "done" if (BC / "bc_model.zip").exists() else "pending"},
        {"name": "BC-3 · PPO finjustering fra BC", "status": "done" if d("ft_bc") == "done" else ("running" if d("ft_bc") == "running" else "pending")},
    ]
    return {
        "ts": time.time(),
        "host": host_stats(),
        "active_run": active_run,
        "phase": phase,
        "queue": queue,
        "runs": {n: {"status": r["status"], "exists": r["exists"], "eps": len(r["eps"]), "last": r["last"]} for n, r in runs.items()},
    }


def series_for(name: str, limit: int = 320) -> dict:
    r = parse_run(name)
    eps = r["eps"]

    def thin(seq, k):
        if len(seq) <= limit:
            return seq
        step = len(seq) / limit
        return [seq[int(i * step)] for i in range(limit)]

    win = thin([[e[0], e[1]] for e in eps], limit)
    rew = thin([[e[0], e[2]] for e in eps], limit)
    tabs = [
        [t.get("total_timesteps"), t.get("entropy_loss"), t.get("explained_variance"), t.get("approx_kl"), t.get("fps")]
        for t in r["tables"]
    ]
    tabs = thin(tabs, limit)
    return {"win": win, "reward": rew, "tables": tabs, "tail": r["tail"]}


HTML = r"""<!DOCTYPE html>
<html lang="no"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>crforge · clash-training</title>
<style>
:root{--bg:#0b0e14;--card:#131722;--card2:#0f1320;--line:#1f2637;--tx:#e8ecf4;--dim:#8b93a7;--acc:#5eead4;--orange:#fb923c;--blue:#60a5fa;--purple:#c084fc;--red:#f87171;--green:#4ade80}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;padding:18px;max-width:1180px;margin:0 auto}
h1{font-size:17px;font-weight:600;letter-spacing:.5px} h1 span{color:var(--acc)}
.sub{color:var(--dim);font-size:12px;margin-top:2px}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:12px;margin-top:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h2{font-size:11px;text-transform:uppercase;letter-spacing:1.2px;color:var(--dim);margin-bottom:10px;font-weight:600}
.c12{grid-column:span 12}.c8{grid-column:span 8}.c6{grid-column:span 6}.c4{grid-column:span 4}
.big{font-size:26px;font-weight:700}.acc{color:var(--acc)}.or{color:var(--orange)}.bl{color:var(--blue)}.pu{color:var(--purple)}.rd{color:var(--red)}.dim{color:var(--dim)}.gr{color:var(--green)}
.bar{height:10px;background:var(--card2);border:1px solid var(--line);border-radius:6px;overflow:hidden;margin:10px 0 6px}
.bar>div{height:100%;background:linear-gradient(90deg,#2dd4bf,#5eead4);width:0%}
.kv{display:flex;justify-content:space-between;padding:3px 0;font-size:13px}
.kv b{font-weight:600}
.qrow{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px dashed var(--line);font-size:13px}
.qrow:last-child{border-bottom:0}
.pill{font-size:10px;padding:2px 8px;border-radius:99px;border:1px solid var(--line)}
.pill.done{color:var(--green);border-color:#14532d;background:#052e1620}
.pill.running{color:var(--orange);border-color:#7c2d12;background:#43140720}
.pill.pending{color:var(--dim)}
.pill.stopped{color:#fbbf24;border-color:#78350f;background:#451a0320}
pre{font-size:11.5px;color:#c7d0e0;background:var(--card2);border:1px solid var(--line);border-radius:8px;padding:10px;overflow-x:auto;white-space:pre-wrap}
canvas{width:100%;height:150px;display:block}
.legend{font-size:11px;color:var(--dim);margin-top:4px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
.hb{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--dim)}
.hb b{color:var(--tx)}
.stat{margin-right:14px}
@media(max-width:800px){.c8,.c6,.c4{grid-column:span 12}}
</style></head><body>
<h1>crforge <span>·</span> clash-training</h1>
<div class="sub" id="sub">laster …</div>
<div class="grid">
  <div class="card c8">
    <h2>Nå</h2>
    <div class="big acc" id="phase-name">–</div>
    <div class="bar"><div id="phase-bar"></div></div>
    <div class="kv"><span class="dim">fremdrift</span><b id="phase-prog">–</b></div>
    <div class="kv"><span class="dim">win / loss / draw</span><b id="phase-win">–</b></div>
    <div class="kv"><span class="dim">reward</span><b id="phase-rew">–</b></div>
    <div class="kv"><span class="dim">fps · ETA</span><b id="phase-eta">–</b></div>
    <div class="kv"><span class="dim">entropi · EV · KL</span><b id="phase-met">–</b></div>
    <div class="kv"><span class="dim">kjører</span><b id="phase-proc">–</b></div>
  </div>
  <div class="card c4">
    <h2>Kø og milepæler</h2>
    <div id="queue"></div>
  </div>
  <div class="card c4">
    <h2>Maskin</h2>
    <div class="kv"><span class="dim">CPU</span><b id="h-cpu">–</b></div>
    <div class="kv"><span class="dim">RAM</span><b id="h-mem">–</b></div>
    <div class="kv"><span class="dim">Load</span><b id="h-la">–</b></div>
    <div class="kv"><span class="dim">Uptime</span><b id="h-up">–</b></div>
    <div class="legend">VM har ledig kapasitet — flere envs vurderes for BC-fasen.</div>
  </div>
  <div class="card c8">
    <h2>Per-deck (denne kjøringen)</h2>
    <pre id="perdeck">–</pre>
    <h2 style="margin-top:12px" id="tail-h">Siste logglinjer</h2>
    <pre id="tail">–</pre>
  </div>
  <div class="card c6">
    <h2>Win% over episoder</h2>
    <canvas id="ch-win"></canvas>
    <div class="legend"><span class="dot" style="background:var(--orange)"></span>L2 fixed-bar <span class="dot" style="background:var(--blue);margin-left:10px"></span>L1 mirror <span class="dot" style="background:#34d399;margin-left:10px"></span>varm-start pool <span class="dot" style="background:#facc15;margin-left:10px"></span>pool-12</div>
  </div>
  <div class="card c6">
    <h2>Reward over episoder</h2>
    <canvas id="ch-rew"></canvas>
    <div class="legend"><span class="dot" style="background:var(--orange)"></span>L2 fixed-bar <span class="dot" style="background:var(--blue);margin-left:10px"></span>L1 mirror <span class="dot" style="background:#34d399;margin-left:10px"></span>varm-start pool <span class="dot" style="background:#facc15;margin-left:10px"></span>pool-12</div>
  </div>
  <div class="card c12">
    <h2>Trening (entropi / explained variance)</h2>
    <canvas id="ch-train" style="height:160px"></canvas>
    <div class="legend"><span class="dot" style="background:var(--green)"></span>entropy_loss <span class="dot" style="background:var(--purple);margin-left:10px"></span>explained_variance</div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
const fmt=(n,d=1)=>n==null?"–":Number(n).toFixed(d);
const hms=s=>s==null?"–":(s>=3600?Math.floor(s/3600)+"t "+Math.floor(s%3600/60)+"m":Math.floor(s/60)+"m "+Math.round(s%60)+"s");
function setPill(el,st){el.className="pill "+st;el.textContent=st=="done"?"ferdig":st=="running"?"kjører":st=="stopped"?"stoppet":"venter"}
let ACTIVE=null;
async function status(){
  try{
    const r=await fetch("api/status");const s=await r.json();
    ACTIVE=s.active_run||ACTIVE;
    $("sub").textContent=new Date(s.ts*1000).toLocaleTimeString("no-NO")+" · oppdateres hvert 15. sek";
    const p=s.phase;
    if(p){
      $("phase-name").textContent=(p.run==="fixed_bar"?"L2 · hog vs FAST rule_based (giant)":p.run==="mirror10m"?"L1 · mirror hog vs hog":p.run);
      const pct=p.total?Math.round(100*p.steps/p.total):0;
      $("phase-bar").style.width=pct+"%";
      $("phase-prog").textContent=(p.steps??"–")+" / "+(p.total??"–")+" · "+pct+"% · ep "+(p.ep??"–");
      $("phase-win").textContent=p.win+"% / "+p.loss+"% / "+(100-p.win-p.loss)+"%";
      $("phase-rew").textContent=fmt(p.reward);
      $("phase-eta").textContent=fmt(p.fps,0)+" fps · ETA "+hms(p.eta_s);
      $("phase-met").textContent=fmt(p.entropy,2)+" · "+fmt(p.ev,2)+" · "+fmt(p.kl,4);
      $("phase-proc").textContent="pid "+p.pid+" · "+fmt(p.elapsed_min,0)+" min";
      $("perdeck").textContent=p.perdeck.length?p.perdeck.join("\n"):"–";
    }
    const q=$("queue");q.innerHTML="";
    for(const it of s.queue){const d=document.createElement("div");d.className="qrow";
      const n=document.createElement("span");n.textContent=it.name;
      const pl=document.createElement("span");setPill(pl,it.status);
      d.append(n,pl);q.append(d);}
    $("h-cpu").textContent=s.host.cpu_pct==null?"–":s.host.cpu_pct+" % av "+s.host.cpus;
    $("h-mem").textContent=s.host.mem.used_gb+" / "+s.host.mem.total_gb+" GB";
    $("h-la").textContent=(s.host.loadavg||[]).join(" · ");
    $("h-up").textContent=s.host.uptime_h+" t";
  }catch(e){$("sub").textContent="feil: "+e}
}
async function series(){
  if(ACTIVE==null) await status();
  const want=["fixed_bar","mirror10m","warm_pool","pool12"];if(ACTIVE&&!want.includes(ACTIVE))want.push(ACTIVE);
  const data={};
  for(const n of want){try{data[n]=await (await fetch("api/series?run="+n)).json();}catch(e){data[n]={win:[],reward:[],tables:[],tail:[]}}}
  const fb=data["fixed_bar"]||{win:[],reward:[],tables:[]},mr=data["mirror10m"]||{win:[],reward:[],tables:[]},wp=data["warm_pool"]||{win:[],reward:[],tables:[]},p12=data["pool12"]||{win:[],reward:[],tables:[]};
  draw("ch-win",[[fb.win,"#fb923c"],[mr.win,"#60a5fa"],[wp.win,"#34d399"],[p12.win,"#facc15"]],{ymin:0,ymax:100,unit:"%"});
  draw("ch-rew",[[fb.reward,"#fb923c"],[mr.reward,"#60a5fa"],[wp.reward,"#34d399"],[p12.reward,"#facc15"]],{unit:""});
  const te=n=>n.tables.map(t=>[t[0],t[1]]);
  const ev=n=>n.tables.map(t=>[t[0],t[2]]);
  draw("ch-train",[[te(fb),"#4ade80"],[te(mr),"#4ade80"],[ev(fb),"#c084fc"],[ev(mr),"#c084fc"]],{unit:""});
  const act=data[ACTIVE]||fb;
  const th=$("tail-h");if(th)th.textContent="Siste logglinjer · "+(ACTIVE||"–");
  $("tail").textContent=(act.tail||[]).slice(-12).join("\n");
}
function bounds(seriesList,dash){
  let xs=[],ys=[];
  for(const [d] of seriesList)for(const p of d){xs.push(p[0]);ys.push(p[1]);}
  if(!xs.length)return null;
  let x0=Math.min(...xs),x1=Math.max(...xs);
  let y0=dash.ymin!==undefined?dash.ymin:Math.min(...ys);
  let y1=dash.ymax!==undefined?dash.ymax:Math.max(...ys);
  if(y0===y1){y0-=1;y1+=1}
  const margin=(y1-y0)*0.08;if(dash.ymin===undefined)y0-=margin;if(dash.ymax===undefined)y1+=margin;
  return {x0,x1,y0,y1};
}
function draw(id,seriesList,dash){
  const c=$(id);const dpr=window.devicePixelRatio||1;
  const w=c.clientWidth,h=150;c.width=w*dpr;c.height=h*dpr;
  const ctx=c.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
  const b=bounds(seriesList,dash);if(!b)return;
  const X=x=>18+(w-30)*(x-b.x0)/Math.max(1e-9,b.x1-b.x0);
  const Y=y=>8+(h-26)*(1-(y-b.y0)/Math.max(1e-9,b.y1-b.y0));
  ctx.strokeStyle="#1f2637";ctx.fillStyle="#8b93a7";ctx.font="10px monospace";
  for(let i=0;i<=4;i++){const y=8+(h-26)*i/4;ctx.beginPath();ctx.moveTo(18,y);ctx.lineTo(w-12,y);ctx.stroke();
    const v=b.y1-(b.y1-b.y0)*i/4;ctx.fillText((Math.abs(v)>=100?v.toFixed(0):v.toFixed(1)),0,y+3);}
  ctx.fillText("ep "+b.x0,18,h-6);ctx.fillText("ep "+b.x1,w-70,h-6);
  for(const [d,col] of seriesList){
    if(!d.length)continue;ctx.strokeStyle=col;ctx.lineWidth=1.6;ctx.beginPath();
    d.forEach((p,i)=>{const px=X(p[0]),py=Y(p[1]);i?ctx.lineTo(px,py):ctx.moveTo(px,py)});ctx.stroke();
  }
}
status();series();setInterval(status,15000);setInterval(series,15000);
</script></body></html>
"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                self._send(200, HTML.encode(), "text/html; charset=utf-8")
            elif self.path.startswith("/api/status"):
                self._send(200, json.dumps(build_status()).encode(), "application/json")
            elif self.path.startswith("/api/series"):
                q = self.path.split("run=", 1)[1].split("&")[0] if "run=" in self.path else "fixed_bar"
                self._send(200, json.dumps(series_for(q)).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")
        except BrokenPipeError:
            pass
        except Exception as exc:  # keep serving
            self._send(500, json.dumps({"error": str(exc)}).encode(), "application/json")


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    print(f"dashboard on :{PORT}", flush=True)
    srv.serve_forever()
