import json
import os
import sys

html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartLift — RL Elevator Visualization</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#07080f;--panel:#0d0f1a;--border:rgba(255,255,255,0.07);
  --blue:#3d7fff;--green:#00e5a0;--amber:#f0a030;--red:#ff4466;--purple:#b464ff;
  --text:#e8eeff;--muted:#5a6480;
}
body{background:var(--bg);font-family:'JetBrains Mono',monospace;color:var(--text);
     min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:16px 12px;gap:12px}

.header{text-align:center}
.title{font-family:'Syne',sans-serif;font-size:26px;font-weight:800;letter-spacing:-1px;
       background:linear-gradient(135deg,#6ab0ff,#00e5a0);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{font-size:10px;color:var(--muted);letter-spacing:3px;margin-top:2px}

/* load bar */
.load-bar{width:100%;max-width:880px;background:var(--panel);border:1px solid var(--border);
          border-radius:10px;padding:10px 16px;display:flex;align-items:center;gap:12px;font-size:11px}
.load-bar input[type=file]{display:none}
.file-btn{padding:6px 14px;border-radius:6px;border:1px solid rgba(61,127,255,0.35);
          background:rgba(61,127,255,0.1);color:#80b4ff;cursor:pointer;font-family:inherit;
          font-size:10px;font-weight:700;letter-spacing:.5px;transition:all .2s;white-space:nowrap}
.file-btn:hover{background:rgba(61,127,255,0.22)}
.file-status{font-size:10px;color:var(--muted);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.file-status.ok{color:var(--green)}
.file-status.err{color:var(--red)}

/* tabs */
.tabs{display:flex;gap:6px;flex-wrap:wrap;justify-content:center}
.tab{padding:7px 16px;border-radius:8px;border:1px solid var(--border);background:transparent;
     color:var(--muted);font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600;
     letter-spacing:.5px;cursor:pointer;transition:all .2s}
.tab:hover{border-color:var(--blue);color:#80b4ff}
.tab.ql   {background:rgba(61,127,255,0.15);border-color:var(--blue);color:#80b4ff}
.tab.sarsa{background:rgba(0,229,160,0.12);border-color:var(--green);color:var(--green)}
.tab.mpc  {background:rgba(240,160,48,0.12);border-color:var(--amber);color:var(--amber)}
.tab.fcfs {background:rgba(255,68,102,0.10);border-color:var(--red);color:var(--red)}
.tab.near {background:rgba(180,100,255,0.12);border-color:var(--purple);color:var(--purple)}

/* main */
.main{display:flex;gap:12px;width:100%;max-width:880px;height:540px}

/* building */
.building-wrap{display:flex;background:var(--panel);border:1px solid var(--border);border-radius:12px;overflow:hidden;flex-shrink:0}
.floor-labels{display:flex;flex-direction:column;padding:6px 6px 6px 8px;justify-content:space-around}
.fl{font-size:9px;font-family:'Syne',sans-serif;font-weight:700;color:var(--muted);
    letter-spacing:1px;text-align:right;flex:1;display:flex;align-items:center;justify-content:flex-end}
.building{display:flex;flex-direction:column;width:180px}
.frow{flex:1;display:flex;align-items:center;border-bottom:1px solid rgba(255,255,255,0.04);
      padding:0 6px;gap:5px;transition:background .2s}
.frow:last-child{border-bottom:none}
.frow.cur{background:rgba(61,127,255,0.07)}
.frow.hasw{background:rgba(240,160,48,0.04)}
.fplat{flex:1;height:2px;background:rgba(255,255,255,0.05);border-radius:1px}
.fplat.cur{background:rgba(61,127,255,0.5);box-shadow:0 0 5px rgba(61,127,255,.4)}
.warea{display:flex;gap:3px;align-items:center;width:52px;justify-content:flex-end}
.pax{width:7px;height:7px;border-radius:50%;background:var(--amber);flex-shrink:0;
     animation:pp 1.6s ease-in-out infinite}
.pax:nth-child(2){animation-delay:.2s}
.pax:nth-child(3){animation-delay:.4s}
.pax:nth-child(4){animation-delay:.6s}
.pax:nth-child(5){animation-delay:.8s}
.pmore{font-size:8px;color:var(--amber);font-weight:700}
@keyframes pp{0%,100%{opacity:.5;transform:scale(.85)}50%{opacity:1;transform:scale(1.1)}}

/* shaft */
.shaft{width:60px;position:relative;background:linear-gradient(to bottom,rgba(15,20,45,.9),rgba(7,8,20,.95));
       border-left:1px solid rgba(61,127,255,0.08);overflow:hidden}
.srail{position:absolute;top:0;left:50%;transform:translateX(-50%);width:2px;height:100%;
       background:linear-gradient(to bottom,transparent,rgba(61,127,255,0.2) 10%,rgba(61,127,255,0.2) 90%,transparent)}
.sglow{position:absolute;inset:0;background:radial-gradient(ellipse at 50% 50%,rgba(61,127,255,0.04),transparent 70%)}
.ecar{position:absolute;left:50%;transform:translateX(-50%);width:50px;z-index:10;
      transition:top .52s cubic-bezier(0.4,0,0.2,1)}
.cable{position:absolute;top:-700px;left:50%;transform:translateX(-50%);width:2px;height:700px;
       background:linear-gradient(to bottom,rgba(80,120,255,0.05),rgba(61,127,255,0.55))}
.darrow{position:absolute;top:-17px;left:50%;transform:translateX(-50%);font-size:12px;
        transition:all .3s;filter:drop-shadow(0 0 4px currentColor)}
.cab{width:50px;height:50px;border-radius:7px;border:2px solid var(--blue);
     background:linear-gradient(135deg,#0d1835,#172a56);
     display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px;position:relative;
     box-shadow:0 0 18px rgba(61,127,255,.3),inset 0 1px 0 rgba(255,255,255,.07);transition:border-color .3s,box-shadow .3s}
.cab.c-sarsa{border-color:var(--green);box-shadow:0 0 18px rgba(0,229,160,.3),inset 0 1px 0 rgba(255,255,255,.07)}
.cab.c-mpc  {border-color:var(--amber);box-shadow:0 0 18px rgba(240,160,48,.3),inset 0 1px 0 rgba(255,255,255,.07)}
.cab.c-fcfs {border-color:var(--red);box-shadow:0 0 18px rgba(255,68,102,.3),inset 0 1px 0 rgba(255,255,255,.07)}
.cab.c-near {border-color:var(--purple);box-shadow:0 0 18px rgba(180,100,255,.3),inset 0 1px 0 rgba(255,255,255,.07)}
.cab.full   {border-color:var(--red)!important;box-shadow:0 0 20px rgba(255,68,102,.5)!important}
.cab.nearfull{border-color:var(--amber)!important}
.doors{display:flex;gap:2px;width:34px;height:28px}
.door{flex:1;background:rgba(61,127,255,0.1);border:1px solid rgba(61,127,255,0.2);border-radius:2px;transition:transform .3s}
.door.ol{transform:scaleX(.05);transform-origin:left}
.door.or{transform:scaleX(.05);transform-origin:right}
.cinfo{font-size:9px;font-weight:700;color:rgba(180,210,255,0.6);letter-spacing:.5px}

/* right */
.right{flex:1;display:flex;flex-direction:column;gap:8px;min-width:0;overflow:hidden}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px}
.ct{font-family:'Syne',sans-serif;font-size:10px;font-weight:700;letter-spacing:2.5px;
    color:var(--muted);text-transform:uppercase;margin-bottom:9px}
.sgrid{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.stat{background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.05);border-radius:6px;padding:7px 9px}
.slabel{font-size:9px;color:var(--muted);letter-spacing:1px;text-transform:uppercase}
.sval{font-size:17px;font-weight:700;margin-top:1px;line-height:1}
.sval.bl{color:var(--blue)} .sval.gr{color:var(--green)} .sval.am{color:var(--amber)} .sval.re{color:var(--red)}
.ibadges{display:flex;flex-wrap:wrap;gap:3px;margin-top:5px;min-height:18px}
.badge{font-size:9px;padding:2px 6px;border-radius:9px;background:rgba(61,127,255,0.12);
       border:1px solid rgba(61,127,255,0.25);color:#80b4ff;font-weight:700}
.pwrap{margin-top:7px}
.plab{display:flex;justify-content:space-between;font-size:9px;color:var(--muted);margin-bottom:3px}
.pbar{height:3px;background:rgba(255,255,255,0.04);border-radius:2px;overflow:hidden}
.pfill{height:100%;border-radius:2px;transition:width .1s;background:linear-gradient(90deg,var(--blue),var(--green))}

.qcard{padding:12px}
.qhint{font-size:9px;color:var(--muted);margin-bottom:6px;letter-spacing:.3px}
.qvals{display:flex;justify-content:center;gap:8px}
.qv{font-size:10px;font-weight:700;padding:4px 12px;border-radius:5px;border:1px solid;text-align:center;min-width:70px}
.qv.best{background:rgba(0,229,160,0.12);border-color:var(--green);color:var(--green)}
.qv.mid {background:rgba(61,127,255,0.07);border-color:rgba(61,127,255,0.25);color:#80b4ff}
.qv.low {background:rgba(255,255,255,0.02);border-color:rgba(255,255,255,0.05);color:var(--muted)}

.logcard{flex:1;display:flex;flex-direction:column;min-height:0;overflow:hidden}
.loginner{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:2px}
.loginner::-webkit-scrollbar{width:2px}
.loginner::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.06)}
.le{font-size:9px;padding:3px 7px;border-radius:4px;border-left:2px solid;animation:lin .2s ease;line-height:1.5}
.le.serve {border-color:var(--green);background:rgba(0,229,160,.05);color:rgba(0,229,160,.9)}
.le.arrive{border-color:var(--amber);background:rgba(240,160,48,.05);color:rgba(240,160,48,.9)}
.le.move  {border-color:var(--blue);background:rgba(61,127,255,.05);color:rgba(130,180,255,.9)}
@keyframes lin{from{opacity:0;transform:translateY(-3px)}to{opacity:1}}

/* controls */
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:center;width:100%;max-width:880px}
.btn{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;padding:8px 18px;
     border-radius:7px;border:1px solid;cursor:pointer;transition:all .2s;letter-spacing:.5px}
.bplay{background:rgba(61,127,255,0.1);border-color:rgba(61,127,255,0.35);color:#80b4ff}
.bplay:hover{background:rgba(61,127,255,0.22)}
.breset{background:rgba(255,68,102,0.08);border-color:rgba(255,68,102,0.28);color:#ff8899}
.breset:hover{background:rgba(255,68,102,0.18)}
.div{width:1px;height:26px;background:var(--border)}
.spgrp{display:flex;gap:3px}
.bsp{font-family:'JetBrains Mono',monospace;font-size:10px;padding:5px 10px;border-radius:5px;
     border:1px solid rgba(255,255,255,0.07);background:transparent;color:var(--muted);cursor:pointer;transition:all .2s}
.bsp.on{background:rgba(61,127,255,0.18);border-color:rgba(61,127,255,0.4);color:#80b4ff}
.splabel{font-size:10px;color:var(--muted);letter-spacing:1px}
</style>
</head>
<body>

<div class="header">
  <div class="title">SmartLift</div>
  <div class="subtitle">RL Elevator Visualization · Real Q-Table Insight</div>
</div>

<!-- File loader -->
<div class="load-bar">
  <label class="file-btn" for="ql-input">📂 Load Q-Learning JSON</label>
  <input type="file" id="ql-input" accept=".json" onchange="loadTable(event,'ql')">
  <span class="file-status" id="ql-status">✓ Using default Q-table</span>
  <label class="file-btn" for="sarsa-input">📂 Load SARSA JSON</label>
  <input type="file" id="sarsa-input" accept=".json" onchange="loadTable(event,'sarsa')">
  <span class="file-status" id="sarsa-status">✓ Using default Q-table</span>
</div>

<div class="tabs">
  <button class="tab ql"   id="tab-ql"    onclick="setCtrl('ql')">Q-Learning</button>
  <button class="tab"      id="tab-sarsa" onclick="setCtrl('sarsa')">SARSA</button>
  <button class="tab"      id="tab-mpc"   onclick="setCtrl('mpc')">MPC (6‑step)</button>
  <button class="tab"      id="tab-fcfs"  onclick="setCtrl('fcfs')">FCFS</button>
  <button class="tab"      id="tab-near"  onclick="setCtrl('near')">Nearest‑Request</button>
</div>

<div class="main">
  <div style="display:flex">
    <div class="floor-labels" id="flabels"></div>
    <div class="building-wrap">
      <div class="building" id="building"></div>
      <div class="shaft" id="shaft">
        <div class="srail"></div><div class="sglow"></div>
        <div class="ecar" id="ecar">
          <div class="cable"></div>
          <div class="darrow" id="darrow">●</div>
          <div class="cab" id="cab">
            <div class="doors"><div class="door" id="dl"></div><div class="door" id="dr"></div></div>
            <div class="cinfo" id="cinfo">0/8</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="right">
    <div class="card">
      <div class="ct">Live Status</div>
      <div class="sgrid">
        <div class="stat"><div class="slabel">Floor</div><div class="sval bl" id="sf">F5</div></div>
        <div class="stat"><div class="slabel">Direction</div><div class="sval" id="sd">IDLE</div></div>
        <div class="stat"><div class="slabel">Served</div><div class="sval gr" id="ss">0</div></div>
        <div class="stat"><div class="slabel">Waiting</div><div class="sval am" id="sw">0</div></div>
      </div>
      <div class="ibadges" id="ibadges"></div>
      <div class="pwrap">
        <div class="plab"><span>Episode</span><span id="sstep">0 / 500</span></div>
        <div class="pbar"><div class="pfill" id="pfill" style="width:0%"></div></div>
      </div>
    </div>

    <div class="card qcard">
      <div class="ct">Decision (Current State)</div>
      <div class="qhint" id="qhint">—</div>
      <div class="qvals" id="qvals">
        <span class="qv low">↓ —</span><span class="qv low">● —</span><span class="qv low">↑ —</span>
      </div>
    </div>

    <div class="card logcard">
      <div class="ct">Event Log</div>
      <div class="loginner" id="log"></div>
    </div>
  </div>
</div>

<div class="controls">
  <button class="btn bplay" id="bplay" onclick="togglePlay()">⏸ PAUSE</button>
  <button class="btn breset" onclick="resetSim()">↺ RESET</button>
  <div class="div"></div>
  <span class="splabel">SPEED</span>
  <div class="spgrp">
    <button class="bsp"    id="sp1" onclick="setSpd(1)">1×</button>
    <button class="bsp on" id="sp2" onclick="setSpd(2)">2×</button>
    <button class="bsp"    id="sp4" onclick="setSpd(4)">4×</button>
    <button class="bsp"    id="sp8" onclick="setSpd(8)">8×</button>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════════════════
//  Q-TABLES  (default empty — will be filled from embedded data)
// ═══════════════════════════════════════════════════════════
let QL_TABLE = {};
let SARSA_TABLE = {};
"""

# Try to load the JSON files, but don't fail if they don't exist
try:
    with open('results/qtable_qlearning.json', 'r') as f:
        ql = json.load(f)
    ql_js = json.dumps(ql, separators=(',', ':'))
    html += f"\nQL_TABLE = {ql_js};\n"
    print("✓ Loaded Q-Learning table")
except FileNotFoundError:
    html += "\n// QL_TABLE remains empty — will use fallback\n"
    print("⚠ Q-Learning table not found, using empty fallback")
except Exception as e:
    html += "\n// QL_TABLE remains empty — will use fallback\n"
    print(f"⚠ Error loading Q-Learning table: {e}")

try:
    with open('results/qtable_sarsa.json', 'r') as f:
        sarsa = json.load(f)
    sarsa_js = json.dumps(sarsa, separators=(',', ':'))
    html += f"\nSARSA_TABLE = {sarsa_js};\n"
    print("✓ Loaded SARSA table")
except FileNotFoundError:
    html += "\n// SARSA_TABLE remains empty — will use fallback\n"
    print("⚠ SARSA table not found, using empty fallback")
except Exception as e:
    html += "\n// SARSA_TABLE remains empty — will use fallback\n"
    print(f"⚠ Error loading SARSA table: {e}")

html += r"""
// ═══════════════════════════════════════════════════════════
//  FILE LOADER  — overwrites embedded tables dynamically
// ═══════════════════════════════════════════════════════════
function loadTable(evt, which){
  const file = evt.target.files[0];
  if(!file) return;
  const statusEl = document.getElementById(which==='ql' ? 'ql-status' : 'sarsa-status');
  const reader = new FileReader();
  reader.onload = e => {
    try {
      const parsed = JSON.parse(e.target.result);
      if(which === 'ql')    QL_TABLE    = parsed;
      else                  SARSA_TABLE = parsed;
      const n = Object.keys(parsed).length;
      statusEl.textContent = `✓ ${file.name}  (${n} states)`;
      statusEl.className = 'file-status ok';
      if((ctrl==='ql'&&which==='ql')||(ctrl==='sarsa'&&which==='sarsa')) resetSim();
    } catch(err) {
      statusEl.textContent = '✗ Invalid JSON file';
      statusEl.className = 'file-status err';
    }
  };
  reader.readAsText(file);
}

// ═══════════════════════════════════════════════════════════
//  CONFIG
// ═══════════════════════════════════════════════════════════
const N=10, CAP=8, STEPS=500, LAM=0.3;
let ctrl='ql', spd=2, playing=true;
let sim=null, aid=null, lts=0, acc=0;
const STEP_MS=540;

// ═══════════════════════════════════════════════════════════
//  RNG
// ═══════════════════════════════════════════════════════════
function mkRng(seed){
  let s=seed|0;
  return ()=>{s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296};
}
let rng=mkRng(42);

// ═══════════════════════════════════════════════════════════
//  Q-TABLE LOOKUP
// ═══════════════════════════════════════════════════════════
function qLookup(table, st){
  const key=`(${st.f}, ${st.d}, ${st.cA}, ${st.cB}, ${st.cH}, ${st.dA}, ${st.dB}, ${st.L})`;
  return table[key] || null;
}

function buildState(sim){
  const f=sim.floor;
  const wkeys = Object.keys(sim.waiting);
  const cA = +wkeys.some(k => +k > f && sim.waiting[k] && sim.waiting[k].length > 0);
  const cB = +wkeys.some(k => +k < f && sim.waiting[k] && sim.waiting[k].length > 0);
  const cH = +(sim.waiting[f] != null && sim.waiting[f].length > 0);
  const dA = +sim.inside.some(p => p.dest > f);
  const dB = +sim.inside.some(p => p.dest < f);
  const n  = sim.inside.length;
  const L  = n===0 ? 0 : n<CAP ? 1 : 2;
  const d  = sim.dir + 1;
  return {f, d, cA, cB, cH, dA, dB, L};
}

// ═══════════════════════════════════════════════════════════
//  CONTROLLERS
// ═══════════════════════════════════════════════════════════
function nearestRaw(f, inside, waiting){
  if(inside.some(p=>p.dest===f)) return 1;
  const wf = waiting[f];
  if(wf && wf.length>0) return 1;
  const targets=[
    ...inside.map(p=>p.dest),
    ...Object.keys(waiting).filter(k=>waiting[k] && waiting[k].length>0).map(Number)
  ];
  if(!targets.length) return 1;
  const nearest=targets.reduce((a,b)=>Math.abs(b-f)<Math.abs(a-f)?b:a);
  return nearest>f?2:nearest<f?0:1;
}

function rlSelect(table, sim){
  const st = buildState(sim);
  const q  = qLookup(table, st);
  if(q !== null){
    const best = q.indexOf(Math.max(...q));
    return {action:best, q, st, known:true};
  }
  const fb = nearestRaw(sim.floor, sim.inside, sim.waiting);
  return {action:fb, q:null, st, known:false};
}

function nearestSelect(sim){ 
  return {action:nearestRaw(sim.floor,sim.inside,sim.waiting), q:null, st:null, known:true}; 
}

function fcfsSelect(sim){
  const f=sim.floor, inside=sim.inside, waiting=sim.waiting;
  if(inside.some(p=>p.dest===f)) return {action:1,q:null,st:null,known:true};
  const wf=waiting[f];
  if(wf&&wf.length>0) return {action:1,q:null,st:null,known:true};
  let target=null, oldest=Infinity;
  for(const k of Object.keys(waiting)){
    const ps=waiting[k];
    if(ps && ps.length>0 && ps[0].arrT<oldest){oldest=ps[0].arrT;target=+k;}
  }
  if(inside.length>0){
    const nd=inside.reduce((a,b)=>Math.abs(b.dest-f)<Math.abs(a.dest-f)?b:a).dest;
    if(target===null||Math.abs(nd-f)<=Math.abs(target-f)) target=nd;
  }
  if(target===null) return {action:1,q:null,st:null,known:true};
  return {action:target>f?2:target<f?0:1, q:null,st:null,known:true};
}

function mpcSelect(sim, H=6){
  let best=-Infinity, bestA=1;
  for(let a=0;a<3;a++){
    const sc=rollout(sim,a,H);
    if(sc>best){best=sc;bestA=a;}
  }
  return {action:bestA, q:null, st:null, known:true};
}

function rollout(sim0, firstA, H){
  let f=sim0.floor, ins=sim0.inside.map(p=>({...p}));
  let wait={};
  for(const k of Object.keys(sim0.waiting)) wait[k]=sim0.waiting[k] ? [...sim0.waiting[k]] : [];
  let tot=0, g=1, disc=0.95;
  for(let i=0;i<H;i++){
    const a=i===0?firstA:nearestRaw(f,ins,wait);
    let r=0;
    if(a===2&&f<N-1){f++;r-=0.5;}
    else if(a===0&&f>0){f--;r-=0.5;}
    const drop=ins.filter(p=>p.dest===f);
    ins=ins.filter(p=>p.dest!==f);
    r+=drop.length*10;
    const wq=wait[f]||[];
    const sp=CAP-ins.length;
    if(wq.length>0&&sp>0){const b=wq.splice(0,sp);ins.push(...b);if(wq.length===0)delete wait[f];}
    const tw=Object.values(wait).reduce((a,b)=>a+(b?b.length:0),0);
    r-=Math.min(tw,10);
    tot+=g*r; g*=disc;
  }
  return tot;
}

function select(sim){
  if(ctrl==='ql')    return rlSelect(QL_TABLE, sim);
  if(ctrl==='sarsa') return rlSelect(SARSA_TABLE, sim);
  if(ctrl==='near')  return nearestSelect(sim);
  if(ctrl==='fcfs')  return fcfsSelect(sim);
  if(ctrl==='mpc')   return mpcSelect(sim, 6);
  return {action:1,q:null,st:null,known:true};
}

// ═══════════════════════════════════════════════════════════
//  SIMULATION
// ═══════════════════════════════════════════════════════════
function initSim(){
  rng=mkRng(42+Math.floor(Math.random()*999));
  sim={floor:5,dir:0,inside:[],waiting:{},step:0,served:0,moves:0,nextId:0,lastRes:null};
}

function spawnPassengers(){
  const L=LAM, eL=Math.exp(-L);
  let k=0,p=1; do{k++;p*=rng();}while(p>eL);
  const n=k-1;
  for(let i=0;i<n;i++){
    const o=Math.floor(rng()*N);
    let d=Math.floor(rng()*N); while(d===o)d=Math.floor(rng()*N);
    if(!sim.waiting[o]) sim.waiting[o]=[];
    sim.waiting[o].push({id:sim.nextId++,origin:o,dest:d,arrT:sim.step});
    log('arrive',`Passenger F${o}→F${d}`);
  }
}

function stepSim(){
  if(sim.step>=STEPS) return false;
  spawnPassengers();
  const res=select(sim);
  sim.lastRes=res;
  const a=res.action;
  if(a===2&&sim.floor<N-1){sim.floor++;sim.dir=1;sim.moves++;log('move',`▲ Up → F${sim.floor}`);}
  else if(a===0&&sim.floor>0){sim.floor--;sim.dir=-1;sim.moves++;log('move',`▼ Down → F${sim.floor}`);}
  else{sim.dir=0;}
  serveFloor();
  sim.step++;
  return true;
}

function serveFloor(){
  const f=sim.floor;
  const drop=sim.inside.filter(p=>p.dest===f);
  sim.inside=sim.inside.filter(p=>p.dest!==f);
  drop.forEach(p=>{sim.served++;log('serve',`✓ Delivered F${p.origin}→F${f} wait:${sim.step-p.arrT}s`);});
  const wq=sim.waiting[f]||[];
  const sp=CAP-sim.inside.length;
  if(wq.length>0&&sp>0){
    const b=wq.splice(0,sp);
    sim.inside.push(...b);
    if(wq.length===0) delete sim.waiting[f];
    log('move',`⬆ Boarded ${b.length} at F${f}`);
  }
}

function log(type,msg){
  const el=document.createElement('div');
  el.className='le '+type;
  el.textContent=`[${String(sim.step).padStart(3,'0')}] ${msg}`;
  const l=document.getElementById('log');
  l.prepend(el);
  if(l.children.length>100) l.lastChild.remove();
}

// ═══════════════════════════════════════════════════════════
//  DOM BUILDING
// ═══════════════════════════════════════════════════════════
function buildDOM(){
  const fl=document.getElementById('flabels');
  fl.innerHTML='';
  for(let f=N-1;f>=0;f--){const d=document.createElement('div');d.className='fl';d.textContent=`F${f}`;fl.appendChild(d);}
  const b=document.getElementById('building');
  b.innerHTML='';
  for(let f=N-1;f>=0;f--){
    const r=document.createElement('div');
    r.className='frow';r.id=`r${f}`;
    r.innerHTML=`<div class="fplat" id="p${f}"></div><div class="warea" id="w${f}"></div>`;
    b.appendChild(r);
  }
}

// ═══════════════════════════════════════════════════════════
//  RENDER
// ═══════════════════════════════════════════════════════════
const CTAB_CLASS={ql:'ql',sarsa:'sarsa',mpc:'mpc',fcfs:'fcfs',near:'near'};
const CAB_CLASS ={ql:'',sarsa:'c-sarsa',mpc:'c-mpc',fcfs:'c-fcfs',near:'c-near'};

function render(){
  if(!sim) return;
  const f=sim.floor;
  const tw=Object.values(sim.waiting).reduce((a,b)=>a+(b?b.length:0),0);

  for(let i=0;i<N;i++){
    const row=document.getElementById(`r${i}`);
    const plat=document.getElementById(`p${i}`);
    if(!row||!plat) continue;
    const wc=(sim.waiting[i]||[]).length;
    row.className='frow'+(i===f?' cur':wc>0?' hasw':'');
    plat.className='fplat'+(i===f?' cur':'');
    const wa=document.getElementById(`w${i}`);
    wa.innerHTML='';
    for(let d=0;d<Math.min(wc,5);d++){const p=document.createElement('div');p.className='pax';wa.appendChild(p);}
    if(wc>5){const m=document.createElement('span');m.className='pmore';m.textContent=`+${wc-5}`;wa.appendChild(m);}
  }

  const shaft=document.getElementById('shaft');
  const sH=shaft.clientHeight||520;
  const rowH=sH/N;
  document.getElementById('ecar').style.top=((N-1-f)*rowH+rowH/2-25)+'px';

  const cab=document.getElementById('cab');
  const load=sim.inside.length;
  const cc=CAB_CLASS[ctrl];
  cab.className='cab'+(cc?' '+cc:'')+(load>=CAP?' full':load>=CAP*.6?' nearfull':'');

  const open=sim.dir===0 && load>0;
  document.getElementById('dl').className='door'+(open?' ol':'');
  document.getElementById('dr').className='door'+(open?' or':'');
  document.getElementById('cinfo').textContent=`${load}/${CAP}`;

  const ar=document.getElementById('darrow');
  const dm={'-1':['▼','var(--red)'],'0':['●','var(--muted)'],'1':['▲','var(--green)']}[sim.dir]||['●','var(--muted)'];
  ar.textContent=dm[0]; ar.style.color=dm[1];

  document.getElementById('sf').textContent=`F${f}`;
  const sd=document.getElementById('sd');
  sd.textContent=sim.dir===1?'UP':sim.dir===-1?'DOWN':'IDLE';
  sd.style.color=sim.dir===1?'var(--green)':sim.dir===-1?'var(--red)':'var(--muted)';
  document.getElementById('ss').textContent=sim.served;
  document.getElementById('sw').textContent=tw;
  document.getElementById('sstep').textContent=`${sim.step} / ${STEPS}`;
  document.getElementById('pfill').style.width=(sim.step/STEPS*100)+'%';

  const ib=document.getElementById('ibadges');
  ib.innerHTML='';
  sim.inside.forEach(p=>{const b=document.createElement('span');b.className='badge';b.textContent=`→F${p.dest}`;ib.appendChild(b);});

  const qv=document.getElementById('qvals');
  const qh=document.getElementById('qhint');
  const res=sim.lastRes;
  if(res && res.q && (ctrl==='ql'||ctrl==='sarsa')){
    const q=res.q, mx=Math.max(...q), labels=['↓','●','↑'];
    qv.innerHTML=q.map((v,i)=>{
      const cls=v===mx?'best':mx-v<8?'mid':'low';
      const mark=i===res.action?' ◀':'';
      return `<span class="qv ${cls}">${labels[i]} ${v.toFixed(1)}${mark}</span>`;
    }).join('');
    const st=res.st;
    qh.textContent=`F${st.f} d${st.d} [↑${st.cA} ↓${st.cB} H${st.cH}] [↑${st.dA} ↓${st.dB}] L${st.L}${res.known?'':' (fallback)'}`;
  } else if(res && !res.q){
    const labels=['↓ DOWN','● STAY','↑ UP'];
    qv.innerHTML=labels.map((l,i)=>`<span class="qv ${i===res.action?'best':'low'}">${l}</span>`).join('');
    qh.textContent=ctrl==='mpc'?'MPC 6-step lookahead':ctrl==='fcfs'?'FCFS: oldest call':'Nearest-Request: greedy heuristic';
  } else {
    qv.innerHTML='<span class="qv low">↓ —</span><span class="qv low">● —</span><span class="qv low">↑ —</span>';
    qh.textContent='—';
  }
}

// ═══════════════════════════════════════════════════════════
//  ANIMATION LOOP
// ═══════════════════════════════════════════════════════════
function loop(ts){
  if(!playing){aid=null;return;}
  if(!lts) lts=ts;
  acc+=(ts-lts)*spd; lts=ts;
  while(acc>=STEP_MS){
    acc-=STEP_MS;
    if(!stepSim()){
      playing=false;
      const b=document.getElementById('bplay');
      b.textContent='✓ DONE'; b.style.opacity='.5';
      break;
    }
    render();
  }
  aid=requestAnimationFrame(loop);
}

// ═══════════════════════════════════════════════════════════
//  CONTROLS
// ═══════════════════════════════════════════════════════════
function togglePlay(){
  playing=!playing;
  const b=document.getElementById('bplay');
  if(playing){b.textContent='⏸ PAUSE';b.style.opacity='1';lts=0;aid=requestAnimationFrame(loop);}
  else{b.textContent='▶ PLAY';if(aid)cancelAnimationFrame(aid);}
}
function resetSim(){
  if(aid)cancelAnimationFrame(aid);
  playing=true; lts=0; acc=0;
  const b=document.getElementById('bplay');
  b.textContent='⏸ PAUSE'; b.style.opacity='1';
  document.getElementById('log').innerHTML='';
  initSim(); render();
  aid=requestAnimationFrame(loop);
}
function setSpd(s){
  spd=s;
  ['1','2','4','8'].forEach(v=>document.getElementById('sp'+v).classList.toggle('on',+v===s));
}
function setCtrl(c){
  ctrl=c;
  ['ql','sarsa','mpc','fcfs','near'].forEach(n=>{
    const t=document.getElementById('tab-'+n);
    if(t) t.className='tab'+(n===c?' '+CTAB_CLASS[c]:'');
  });
  resetSim();
}

// ═══════════════════════════════════════════════════════════
//  INITIALIZE
// ═══════════════════════════════════════════════════════════
buildDOM(); initSim(); render();
aid=requestAnimationFrame(loop);
</script>
</body>
</html>"""

# Save the HTML file
output_path = 'smartlift_viz.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✓ Visualization saved to: {output_path}")