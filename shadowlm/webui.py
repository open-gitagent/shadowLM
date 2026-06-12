"""The built-in studio dashboard — a single-file SPA served at `/`.

Left sidebar navigation, the workflow in order: Datasets → Models → Trainings
→ Runs → Playground. Vanilla HTML/CSS/JS over the same JSON protocol the SDK
speaks — no build step, no frameworks, view-source friendly, like the rest of
the box.
"""

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ShadowLM · slm♥</title>
<link rel="icon" type="image/png" href="/logo.png">
<style>
  :root {
    --ink:#16120E; --umbra:#221C16; --panel:#2A231C; --bone:#F2EAE0;
    --muted:#9A8F82; --heart:#E5484D; --ok:#3FB950; --warn:#D29922;
    --line:#3A3129;
  }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--ink); color:var(--bone); height:100vh; display:flex;
         font:14px/1.5 "JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace; }
  /* ---- left sidebar ---- */
  #side { width:212px; min-width:212px; display:flex; flex-direction:column;
          border-right:1px solid var(--line); padding:16px 10px; gap:2px; }
  #side .brand { padding:4px 12px 16px; font-weight:700; display:flex;
                 gap:10px; align-items:center; }
  #side .brand img { width:36px; height:36px; border-radius:9px;
                     border:1px solid var(--line); }
  #side .brand .mark { color:var(--heart); }
  #side .brand .sub { font-weight:400; }
  #side a { color:var(--muted); text-decoration:none; padding:9px 12px;
            border-radius:9px; font-size:13.5px; display:flex; gap:9px;
            align-items:center; }
  #side a.on { color:var(--bone); background:var(--umbra);
               box-shadow:inset 2px 0 0 var(--heart); }
  #side a:hover { color:var(--bone); }
  #side .n { width:16px; text-align:center; color:var(--heart); opacity:.85; }
  #side .spacer { flex:1; }
  #side .foot { padding:10px 12px; display:grid; gap:8px; }
  #side .sub { color:var(--muted); font-size:11px; }
  input,select,textarea,button { background:var(--umbra); color:var(--bone);
    border:1px solid var(--line); border-radius:8px; padding:8px 11px; font:inherit; }
  input:focus,select:focus,textarea:focus { outline:1px solid var(--heart); }
  button { cursor:pointer; }
  button.primary { background:var(--heart); border-color:var(--heart);
                   color:#fff; font-weight:700; }
  button.ghost { background:transparent; }
  button.danger { color:var(--heart); background:transparent; }
  main { flex:1; overflow-y:auto; padding:24px 30px; min-width:0; }
  h2 { font-size:15px; margin-bottom:4px; }
  .lead { color:var(--muted); font-size:12.5px; margin-bottom:18px; }
  .grid { display:grid; gap:12px; }
  .cards { grid-template-columns:repeat(auto-fill,minmax(270px,1fr)); }
  .card { background:var(--umbra); border:1px solid var(--line);
          border-radius:12px; padding:14px; }
  .card.sel { border-color:var(--heart); }
  .card h4 { font-size:13.5px; margin-bottom:2px; }
  .card .meta { color:var(--muted); font-size:11.5px; }
  .pill { display:inline-block; padding:1px 9px; border-radius:999px;
          font-size:11px; border:1px solid var(--line); color:var(--muted); }
  .pill.red { color:var(--heart); border-color:var(--heart); }
  .pill.green { color:var(--ok); border-color:var(--ok); }
  .pill.gold { color:var(--warn); border-color:var(--warn); }
  .rowflex { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th { text-align:left; color:var(--muted); font-size:11px; text-transform:uppercase;
       letter-spacing:.1em; padding:8px 10px; border-bottom:1px solid var(--line); }
  td { padding:9px 10px; border-bottom:1px solid var(--line); }
  tr.click { cursor:pointer; } tr.click:hover { background:var(--umbra); }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%;
         margin-right:7px; }
  .dot.succeeded{background:var(--ok);} .dot.failed{background:var(--heart);}
  .dot.running{background:var(--warn); animation:pulse 1.2s infinite;}
  .dot.pending{background:var(--muted);} .dot.stopped{background:var(--warn);}
  @keyframes pulse { 50%{opacity:.35;} }
  svg.chart { width:100%; height:280px; background:var(--umbra);
              border:1px solid var(--line); border-radius:12px; margin:14px 0; }
  .err { color:var(--heart); white-space:pre-wrap; margin-top:10px; }
  form.stack { display:grid; gap:10px; max-width:760px; }
  form.stack .pair { display:grid; grid-template-columns:160px 1fr; gap:10px;
                     align-items:center; }
  form.stack label { color:var(--muted); font-size:12px; }
  textarea { resize:vertical; }
  pre.preview { background:var(--ink); border:1px solid var(--line);
    border-radius:8px; padding:10px; font-size:11.5px; overflow-x:auto;
    color:var(--muted); max-height:180px; }
  .msg { padding:9px 11px; border-radius:10px; font-size:13px;
         white-space:pre-wrap; max-width:75%; }
  .msg.you { background:var(--panel); align-self:flex-end; }
  .msg.slm { background:var(--umbra); border:1px solid var(--line); }
  .msg.slm::before { content:"slm♥ "; color:var(--heart); font-weight:700; }
  .back { color:var(--heart); text-decoration:none; font-size:12.5px; }
  /* ---- training wizard ---- */
  .steps { display:flex; gap:0; margin:18px 0 26px; max-width:860px; }
  .step { flex:1; text-align:center; position:relative; color:var(--muted);
          font-size:12px; padding-top:30px; cursor:default; }
  .step::before { content:attr(data-n); position:absolute; top:0; left:50%;
    transform:translateX(-50%); width:24px; height:24px; line-height:22px;
    border-radius:50%; border:1px solid var(--line); background:var(--umbra);
    font-size:11px; }
  .step::after { content:""; position:absolute; top:12px; left:calc(50% + 14px);
    width:calc(100% - 28px); height:1px; background:var(--line); }
  .step:last-child::after { display:none; }
  .step.done { color:var(--bone); cursor:pointer; }
  .step.done::before { content:"✓"; color:var(--ok); border-color:var(--ok); }
  .step.cur { color:var(--bone); }
  .step.cur::before { color:#fff; background:var(--heart);
                      border-color:var(--heart); font-weight:700; }
  .wizbody { max-width:860px; }
  .wizfoot { display:flex; gap:10px; margin-top:22px; max-width:860px; }
  .methodcard { cursor:pointer; }
  .methodcard:hover { border-color:var(--muted); }
  .methodcard.sel { border-color:var(--heart);
                    box-shadow:0 0 0 1px var(--heart) inset; }
  .review td:first-child { color:var(--muted); width:180px; }
  /* ---- playground ---- */
  #pg { display:flex; flex-direction:column; height:100%; min-height:0; }
  #pg-top { display:flex; gap:10px; align-items:center; position:relative;
            padding-bottom:12px; }
  #pg-top .selbtn { display:flex; gap:8px; align-items:center; font-weight:700; }
  #pg-top .selbtn .pill { font-weight:400; }
  .pop { position:absolute; top:46px; left:0; z-index:10; width:520px;
         background:var(--panel); border:1px solid var(--line);
         border-radius:14px; padding:12px; box-shadow:0 18px 48px #0009; }
  .tabs { display:flex; background:var(--ink); border-radius:999px; padding:3px;
          margin-bottom:10px; }
  .tabs button { flex:1; border:none; background:transparent; border-radius:999px;
                 padding:7px; color:var(--muted); }
  .tabs button.on { background:var(--umbra); color:var(--bone);
                    box-shadow:inset 0 0 0 1px var(--line); }
  .pop input.search { width:100%; margin-bottom:8px; }
  .pop .list { max-height:300px; overflow-y:auto; }
  .pop .item { display:flex; justify-content:space-between; align-items:center;
               gap:8px; padding:9px 10px; border-radius:9px; cursor:pointer;
               font-size:13px; }
  .pop .item:hover { background:var(--umbra); }
  .pop .item .meta { color:var(--muted); font-size:11px; }
  #pg-log { flex:1; overflow-y:auto; display:flex; flex-direction:column;
            gap:10px; padding:14px 2px; }
  .hero { flex:1; display:flex; flex-direction:column; align-items:center;
          justify-content:center; gap:22px; }
  .hero h1 { font-size:26px; font-weight:700; }
  .hero h1 .mark { color:var(--heart); }
  .hero .sub2 { color:var(--muted); font-size:13px; }
  .heroform { width:min(680px, 90%); display:flex; gap:8px;
              background:var(--umbra); border:1px solid var(--line);
              border-radius:999px; padding:8px 8px 8px 20px; }
  .heroform input { flex:1; background:transparent; border:none; padding:8px 0; }
  .heroform input:focus { outline:none; }
  .heroform button { border-radius:999px; width:42px; height:42px; }
  .cmp { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .cmp .col { background:var(--umbra); border:1px solid var(--line);
              border-radius:10px; padding:10px 12px; font-size:13px;
              white-space:pre-wrap; }
  .cmp .col b { display:block; font-size:10.5px; letter-spacing:.1em;
                text-transform:uppercase; margin-bottom:6px; }
  .cmp .col.base b { color:var(--muted); }
  .cmp .col.tuned { border-color:var(--heart); }
  .cmp .col.tuned b { color:var(--heart); }
  #pg-form { display:flex; gap:8px; padding-top:12px;
             border-top:1px solid var(--line); }
  #pg-form input { flex:1; }
  .switch { display:flex; gap:6px; align-items:center; color:var(--muted);
            font-size:12px; cursor:pointer; }
</style>
</head>
<body>
<aside id="side">
  <div class="brand"><img src="/logo.png" alt="ShadowLM">
    <div>ShadowLM<div class="sub"><span class="mark">slm♥</span> trainer</div></div>
  </div>
  <a href="#datasets"><span class="n">1</span>Datasets</a>
  <a href="#models"><span class="n">2</span>Models</a>
  <a href="#train"><span class="n">3</span>Trainings</a>
  <a href="#runs"><span class="n">4</span>Runs</a>
  <a href="#playground"><span class="n">♥</span>Playground</a>
  <div class="spacer"></div>
  <div class="foot">
    <span class="sub" id="health">connecting…</span>
    <input id="apikey" type="password" placeholder="API key"
           title="Sent as Bearer auth; stored in this browser only">
  </div>
</aside>
<main id="page"></main>
<script>
const $ = s => document.querySelector(s);
const S = { datasets:[], models:{catalog:[],recent:[]}, methods:[], jobs:[],
            pick:{dataset:null, model:null}, run:null, msgs:[], timer:null,
            pg:{msgs:[], busy:false} };

const key = () => localStorage.getItem("slm_api_key") || "";
$("#apikey").value = key();
$("#apikey").addEventListener("change", e => localStorage.setItem("slm_api_key", e.target.value));

async function api(path, opts={}) {
  const headers = {"Content-Type":"application/json"};
  if (key()) headers["Authorization"] = "Bearer " + key();
  const r = await fetch(path, {...opts, headers});
  if (!r.ok) throw new Error((await r.json().catch(()=>({}))).error || r.statusText);
  return r.json();
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

// ---- router ------------------------------------------------------------------
const routes = { datasets: pageDatasets, models: pageModels, train: pageTrain,
                 runs: pageRuns, playground: pagePlayground };
function route() {
  clearInterval(S.timer); S.timer = null;
  const [name, arg] = location.hash.replace("#","").split("/");
  document.querySelectorAll("#side a").forEach(a =>
    a.classList.toggle("on", a.hash === "#" + (name || "datasets")));
  if (name === "runs" && arg) return pageRunDetail(arg);
  (routes[name] || pageDatasets)();
}
window.addEventListener("hashchange", route);

// ---- boot ----------------------------------------------------------------------
(async () => {
  try {
    const h = await api("/v1/health");
    $("#health").textContent = `backend=${h.backend} · v${h.version}`;
    S.methods = (await api("/v1/methods")).methods;
  } catch(e) { $("#health").textContent = "⚠ " + e.message; }
  route();
})();

// ====== DATASETS ==================================================================
async function pageDatasets() {
  S.datasets = (await api("/v1/datasets")).datasets;
  $("#page").innerHTML = `
    <h2>Datasets</h2>
    <p class="lead">Upload once, train many times. JSONL — chat, instruction,
       preference, or raw text rows; the format is auto-detected.</p>
    <div class="grid cards" id="dscards">
      ${S.datasets.map(d => `
        <div class="card">
          <div class="rowflex" style="justify-content:space-between">
            <h4>${esc(d.name)}</h4><span class="pill">${d.format}</span>
          </div>
          <div class="meta">${d.rows} rows · ${d.dataset_id}</div>
          <div class="rowflex" style="margin-top:10px">
            <button onclick="useDataset('${d.dataset_id}')" class="primary">train on this ›</button>
            <button class="ghost" onclick="previewDataset('${d.dataset_id}')">preview</button>
            <button class="danger" onclick="dropDataset('${d.dataset_id}')">delete</button>
          </div>
          <div id="pv-${d.dataset_id}"></div>
        </div>`).join("")}
      <div class="card">
        <h4>New dataset</h4>
        <form class="stack" id="dsform" style="margin-top:8px">
          <input id="ds-name" placeholder="name (e.g. support-tickets-v1)">
          <textarea id="ds-rows" rows="6" placeholder='one JSON row per line:
{"messages":[{"role":"user","content":"…"},{"role":"assistant","content":"…"}]}'></textarea>
          <div class="rowflex">
            <input type="file" id="ds-file" accept=".jsonl,.json">
            <button class="primary">upload ›</button>
          </div>
        </form>
      </div>
    </div>`;
  $("#dsform").addEventListener("submit", async e => {
    e.preventDefault();
    let text = $("#ds-rows").value.trim();
    const f = $("#ds-file").files[0];
    if (f) text = await f.text();
    const rows = text.split("\n").filter(Boolean).map(l => JSON.parse(l));
    if (!rows.length) return alert("no rows");
    await api("/v1/datasets", {method:"POST", body:JSON.stringify(
      {name: $("#ds-name").value, rows})});
    pageDatasets();
  });
}
window.useDataset = id => { S.pick.dataset = id; location.hash = "#train"; };
window.dropDataset = async id => { await api("/v1/datasets/"+id, {method:"DELETE"}); pageDatasets(); };
window.previewDataset = async id => {
  const d = await api("/v1/datasets/"+id);
  $("#pv-"+id).innerHTML = `<pre class="preview">${esc(
    d.preview.map(r => JSON.stringify(r)).join("\n"))}</pre>`;
};

// ====== MODELS ====================================================================
async function pageModels() {
  S.models = await api("/v1/models");
  const card = (m, extra="") => `
    <div class="card ${S.pick.model===m.id?"sel":""}">
      <div class="rowflex" style="justify-content:space-between">
        <h4>${esc(m.id.split("/").pop())}</h4>
        <span>${m.dev?'<span class="pill green">dev pick</span>':""}
        ${m.gated?'<span class="pill gold">HF token</span>':""}</span>
      </div>
      <div class="meta">${esc(m.id)}</div>
      <div class="meta">${m.params || ""} ${m.note ? "· " + esc(m.note) : extra}</div>
      <div class="rowflex" style="margin-top:10px">
        <button class="primary" onclick="useModel('${m.id}')">train this ›</button>
        <button class="ghost" onclick="tryModel('${m.id}')">playground</button>
      </div>
    </div>`;
  $("#page").innerHTML = `
    <h2>Models</h2>
    <p class="lead">Any open model on the HuggingFace hub works — these are
       good starting points. Server backend: <b>${S.models.server_backend}</b>.</p>
    <div class="grid cards">
      ${S.models.recent.filter(r => !S.models.catalog.some(c=>c.id===r))
        .map(id => card({id, note:"recently trained here"})).join("")}
      ${S.models.catalog.map(m => card(m)).join("")}
      <div class="card">
        <h4>Custom</h4>
        <div class="meta">any HF hub id</div>
        <form class="rowflex" style="margin-top:10px" id="modelform">
          <input id="model-free" placeholder="org/model-name" style="flex:1">
          <button class="primary">use ›</button>
        </form>
      </div>
    </div>`;
  $("#modelform").addEventListener("submit", e => {
    e.preventDefault();
    if ($("#model-free").value.trim()) useModel($("#model-free").value.trim());
  });
}
window.useModel = id => { S.pick.model = id; location.hash = "#train"; };
window.tryModel = id => { S.pick.model = id; location.hash = "#playground"; };

// ====== TRAININGS — the guided wizard =============================================
const WIZ_STEPS = ["Data", "Model", "Method", "Tune", "Launch"];
const REC = f => f === "preference" ? ["dpo"]
           : f === "text" ? ["cpt"]
           : f === "prompt" ? ["grpo"]
           : ["lora", "qlora", "dora"];  // chat / sharegpt / instruction
const LORA_FAMILY = ["lora", "qlora", "dora", "adapter", "more"];

function wiz() {
  if (!S.wiz) S.wiz = { step: 0, ds: null, model: null, method: null,
    p: { steps: 60, lorar: 16, lr: "", batch: 2, eval: false } };
  return S.wiz;
}

async function pageTrain() {
  if (!S.datasets.length) S.datasets = (await api("/v1/datasets")).datasets;
  if (!S.methods.length) S.methods = (await api("/v1/methods")).methods;
  if (!S.models.catalog.length) S.models = await api("/v1/models");
  const w = wiz();
  // arriving with a pick from another page pre-completes that step
  if (S.pick.dataset) { w.ds = S.pick.dataset; S.pick.dataset = null;
                        w.step = Math.max(w.step, 1); }
  if (S.pick.model)   { w.model = S.pick.model; S.pick.model = null;
                        w.step = Math.max(w.step, 2); }
  renderWiz();
}

function renderWiz() {
  const w = wiz();
  const can = [!!w.ds, !!w.model, !!w.method, true, true];
  $("#page").innerHTML = `
    <h2>New training</h2>
    <p class="lead">five decisions, in the order they depend on each other —
       everything else has defaults.</p>
    <div class="steps">
      ${WIZ_STEPS.map((s, i) => `<div class="step
        ${i < w.step ? "done" : i === w.step ? "cur" : ""}" data-n="${i + 1}"
        ${i < w.step ? `onclick="wizGo(${i})"` : ""}>${s}</div>`).join("")}
    </div>
    <div class="wizbody" id="wizbody"></div>
    <div class="wizfoot" id="wizfoot"></div>`;
  [wizData, wizModel, wizMethod, wizTune, wizLaunch][w.step]();
  const foot = $("#wizfoot");
  if (w.step > 0) foot.innerHTML += `<button class="ghost" onclick="wizGo(${w.step - 1})">‹ back</button>`;
  if (w.step < 4) foot.innerHTML += `<button class="primary" id="wiznext"
      ${can[w.step] ? "" : "disabled"} onclick="wizGo(${w.step + 1})">continue ›</button>`;
}
window.wizGo = n => { wiz().step = n; renderWiz(); };

// step 1 — Data: the dataset decides what training even means
function wizData() {
  const w = wiz();
  $("#wizbody").innerHTML = `
    <div class="grid cards">
      ${S.datasets.map(d => `
        <div class="card methodcard ${w.ds === d.dataset_id ? "sel" : ""}"
             onclick="wizPickDs('${d.dataset_id}')">
          <div class="rowflex" style="justify-content:space-between">
            <h4>${esc(d.name)}</h4><span class="pill">${d.format}</span></div>
          <div class="meta">${d.rows} rows</div>
        </div>`).join("")}
      <div class="card"><h4>New dataset</h4>
        <div class="meta">paste or upload on the Datasets page</div>
        <button class="ghost" style="margin-top:10px"
                onclick="location.hash='#datasets'">go to Datasets ›</button>
      </div>
    </div>`;
}
window.wizPickDs = id => { const w = wiz(); w.ds = id; renderWiz(); };

// step 2 — Model
function wizModel() {
  const w = wiz();
  const card = m => `
    <div class="card methodcard ${w.model === m.id ? "sel" : ""}"
         onclick="wizPickModel('${m.id}')">
      <div class="rowflex" style="justify-content:space-between">
        <h4>${esc(m.id.split("/").pop())}</h4>
        <span>${m.dev ? '<span class="pill green">dev pick</span>' : ""}
        ${m.gated ? '<span class="pill gold">HF token</span>' : ""}</span></div>
      <div class="meta">${esc(m.id)}</div>
      <div class="meta">${m.params || ""}${m.note ? " · " + esc(m.note) : ""}</div>
    </div>`;
  $("#wizbody").innerHTML = `
    <div class="grid cards">
      ${S.models.recent.filter(r => !S.models.catalog.some(c => c.id === r))
        .map(id => card({ id, note: "recently trained here" })).join("")}
      ${S.models.catalog.map(card).join("")}
      <div class="card"><h4>Custom</h4>
        <form class="rowflex" style="margin-top:8px"
              onsubmit="event.preventDefault(); wizPickModel(this.q.value.trim())">
          <input name="q" placeholder="org/model-name" style="flex:1"
                 value="${esc(w.model && !S.models.catalog.some(c => c.id === w.model) ? w.model : "")}">
          <button class="primary">use</button>
        </form></div>
    </div>`;
}
window.wizPickModel = id => { if (!id) return; const w = wiz(); w.model = id; renderWiz(); };

// step 3 — Method: ranked by what the chosen data supports
function wizMethod() {
  const w = wiz();
  const meta = S.datasets.find(d => d.dataset_id === w.ds) || {};
  const rec = REC(meta.format);
  const ordered = [...S.methods].sort((a, b) =>
    (rec.includes(b.name) - rec.includes(a.name)));
  $("#wizbody").innerHTML = `
    <p class="sub" style="margin-bottom:12px">your dataset is
      <b>${meta.format || "?"}</b> — recommended methods first.</p>
    <div class="grid cards">
      ${ordered.map(m => `
        <div class="card methodcard ${w.method === m.name ? "sel" : ""}"
             onclick="wizPickMethod('${m.name}')">
          <div class="rowflex" style="justify-content:space-between">
            <h4>${m.name}</h4>
            <span>${rec.includes(m.name) ? '<span class="pill red">recommended</span>' : ""}
            <span class="pill">${m.trainer}</span></span></div>
          <div class="meta">${esc(m.description)}</div>
          <div class="meta">default lr ${m.default_lr}</div>
        </div>`).join("")}
    </div>`;
}
window.wizPickMethod = name => {
  const w = wiz(); w.method = name;
  const m = S.methods.find(x => x.name === name);
  if (m && !w.p.lrTouched) w.p.lr = String(m.default_lr);  // method teaches lr
  renderWiz();
};

// step 4 — Tune: only the knobs this method actually has
function wizTune() {
  const w = wiz();
  const showRank = LORA_FAMILY.includes(w.method);
  $("#wizbody").innerHTML = `
    <form class="stack" onsubmit="event.preventDefault()">
      <div class="pair"><label>max steps</label>
        <input id="z-steps" type="number" value="${w.p.steps}" min="1"
               onchange="wiz().p.steps = +this.value"></div>
      ${showRank ? `<div class="pair"><label>lora r ${w.method === "adapter" ? "(adapter width)" : ""}</label>
        <input id="z-rank" type="number" value="${w.p.lorar}" min="1"
               onchange="wiz().p.lorar = +this.value"></div>` : ""}
      <div class="pair"><label>learning rate</label>
        <input id="z-lr" value="${esc(w.p.lr)}"
               onchange="wiz().p.lr = this.value; wiz().p.lrTouched = true"></div>
      <div class="pair"><label>batch size</label>
        <input id="z-batch" type="number" value="${w.p.batch}" min="1"
               onchange="wiz().p.batch = +this.value"></div>
      <div class="pair"><label>held-out eval</label>
        <label class="rowflex"><input type="checkbox" ${w.p.eval ? "checked" : ""}
          style="width:auto" onchange="wiz().p.eval = this.checked">
          hold out 10% — see overfitting, not just training loss</label></div>
    </form>`;
}

// step 5 — Launch: the dry-run, then the button
function wizLaunch() {
  const w = wiz();
  const meta = S.datasets.find(d => d.dataset_id === w.ds) || {};
  const m = S.methods.find(x => x.name === w.method) || {};
  $("#wizbody").innerHTML = `
    <table class="review">
      <tr><td>dataset</td><td>${esc(meta.name || w.ds)} · ${meta.rows} rows · ${meta.format}</td></tr>
      <tr><td>model</td><td>${esc(w.model)}</td></tr>
      <tr><td>method</td><td>${w.method} <span class="pill">${m.trainer || ""}</span></td></tr>
      <tr><td>max steps</td><td>${w.p.steps}</td></tr>
      ${LORA_FAMILY.includes(w.method) ? `<tr><td>lora r</td><td>${w.p.lorar}</td></tr>` : ""}
      <tr><td>learning rate</td><td>${esc(w.p.lr || "(method default)")}</td></tr>
      <tr><td>batch size</td><td>${w.p.batch}</td></tr>
      <tr><td>held-out eval</td><td>${w.p.eval ? "10% held out" : "off"}</td></tr>
    </table>
    <div class="rowflex" style="margin-top:18px">
      <button class="primary" onclick="wizStart()">start training ›</button>
      <span class="sub">this exact config runs — nothing hidden</span>
    </div>
    <div class="err" id="wizerr"></div>`;
}
window.wizStart = async () => {
  const w = wiz();
  const config = { method: w.method, max_steps: w.p.steps,
                   per_device_train_batch_size: w.p.batch };
  if (LORA_FAMILY.includes(w.method)) config.lora_r = w.p.lorar;
  if (w.p.lr) config.learning_rate = parseFloat(w.p.lr);
  try {
    const out = await api("/v1/finetunes", { method: "POST", body: JSON.stringify({
      base_model: w.model, config, dataset_id: w.ds,
      eval_dataset: w.p.eval ? "auto" : null,
      load_in_4bit: false, max_seq_length: 2048 }) });
    S.wiz = null;  // next training starts fresh
    location.hash = "#runs/" + out.job_id;
  } catch (e) { $("#wizerr").textContent = e.message; }
};

// ====== RUNS ======================================================================
async function pageRuns() {
  const render = async () => {
    S.jobs = (await api("/v1/finetunes")).jobs;
    if (!document.querySelector("#runstable")) return;  // navigated away
    $("#runstable tbody").innerHTML = S.jobs.map(j => `
      <tr class="click" onclick="location.hash='#runs/${j.job_id}'">
        <td><span class="dot ${j.status}"></span>${j.job_id.slice(0,10)}</td>
        <td>${esc((j.base_model||"").split("/").pop())}</td>
        <td>${j.method || "?"}</td><td>${j.status}</td><td>${j.steps}</td>
        <td>${j.final_loss != null ? j.final_loss.toFixed(4) : "—"}</td>
      </tr>`).join("") || '<tr><td colspan="6" class="sub">no runs yet — start one in Trainings</td></tr>';
  };
  $("#page").innerHTML = `
    <h2>Runs</h2><p class="lead">every training this server has executed.</p>
    <table id="runstable"><thead><tr>
      <th>run</th><th>model</th><th>method</th><th>status</th><th>steps</th><th>final loss</th>
    </tr></thead><tbody></tbody></table>`;
  await render();
  S.timer = setInterval(render, 2000);
}

function chartSVG(steps, evals) {
  const pts = steps.map(s => s.loss), W = 860, H = 270, P = 36;
  if (!pts.length) return '<p class="sub">waiting for the first metric…</p>';
  const all = pts.concat(evals.map(e => e.loss));
  const lo = Math.min(...all), hi = Math.max(...all), span = (hi - lo) || 1;
  const x = i => P + i * (W - 2*P) / Math.max(1, pts.length - 1);
  const y = v => H - P - (v - lo) * (H - 2*P) / span;
  const line = pts.map((v,i) => `${i?"L":"M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const ev = evals.map(e => `<circle cx="${x(Math.min(pts.length-1, Math.max(0,e.step-1)))}"
      cy="${y(e.loss)}" r="4" fill="none" stroke="#D29922" stroke-width="2"/>`).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <text x="${P}" y="${P-12}" fill="#9A8F82" font-size="11">loss ${hi.toFixed(3)}</text>
    <text x="${P}" y="${H-P+18}" fill="#9A8F82" font-size="11">${lo.toFixed(3)}</text>
    <path d="${line}" fill="none" stroke="#E5484D" stroke-width="2.5"
          stroke-linejoin="round" stroke-linecap="round"/>${ev}</svg>`;
}

async function pageRunDetail(id) {
  S.run = id;
  if (S.msgs.runFor !== id) { S.msgs = []; S.msgs.runFor = id; }
  const render = async () => {
    if (location.hash !== "#runs/" + id) return;
    const [job, m, list] = await Promise.all([
      api("/v1/finetunes/"+id), api(`/v1/finetunes/${id}/metrics`), api("/v1/finetunes")]);
    const j = list.jobs.find(x => x.job_id === id) || {};
    const last = m.steps[m.steps.length-1];
    $("#page").innerHTML = `
      <a class="back" href="#runs">‹ all runs</a>
      <div class="rowflex" style="margin-top:8px">
        <h2>${id}</h2>
        <span class="pill ${job.status==="succeeded"?"green":job.status==="failed"?"red":"gold"}">${job.status}</span>
        ${(job.status==="running"||job.status==="pending")
          ? `<button class="ghost" onclick="cancelRun('${id}')">cancel</button>` : ""}
        ${job.status==="succeeded"
          ? `<button class="primary" onclick="playRun('${id}','${esc(j.base_model||"")}')">open in playground ›</button>` : ""}
      </div>
      <p class="sub" style="margin-top:4px">
        ${esc(j.base_model||"")} · ${j.method||"?"}
        ${last ? `· step ${last.step} · loss ${last.loss.toFixed(4)}` : ""}
        ${job.final_loss != null ? `· final ${job.final_loss.toFixed(4)}` : ""}
        ${last && last.tokens_per_s ? `· ${Math.round(last.tokens_per_s)} tok/s` : ""}
      </p>
      ${chartSVG(m.steps, m.evals)}
      ${job.error ? `<div class="err">${esc(job.error)}</div>` : ""}`;
  };
  await render();
  S.timer = setInterval(render, 1800);
}
window.cancelRun = async id => api(`/v1/finetunes/${id}/cancel`, {method:"POST"});
window.playRun = (id, base) => {
  S.pick.model = base; S.pick.adapter = id; location.hash = "#playground";
};

// ====== PLAYGROUND ================================================================
function pgState() {
  if (!S.pgv2) S.pgv2 = { msgs:[], busy:false, model:null, adapter:null,
    compare:false, sys:"", temp:0.7, max:256, pop:false, tab:"hub", q:"",
    gear:false };
  return S.pgv2;
}

async function pagePlayground() {
  S.jobs = (await api("/v1/finetunes")).jobs;
  if (!S.models.catalog.length) S.models = await api("/v1/models");
  const p = pgState();
  if (S.pick.model) { p.model = S.pick.model; S.pick.model = null; }
  if (S.pick.adapter) {
    p.adapter = S.pick.adapter; S.pick.adapter = null; p.compare = true;
    const j = S.jobs.find(x => x.job_id === p.adapter);
    if (j) p.model = j.base_model;
  }
  if (!p.model) p.model = "mlx-community/Qwen2.5-0.5B-Instruct-4bit";
  pgShell();
}

function pgLabel() {
  const p = pgState();
  if (p.adapter) {
    const j = S.jobs.find(x => x.job_id === p.adapter);
    return `${p.adapter.slice(0,8)} <span class="pill red">fine-tuned</span>
            <span class="meta">${esc((j?.base_model||p.model).split("/").pop())}</span>`;
  }
  return `${esc(p.model.split("/").pop())} <span class="pill">base</span>`;
}

function pgShell() {
  const p = pgState();
  $("#page").innerHTML = `
    <div id="pg">
      <div id="pg-top">
        <button class="ghost selbtn" onclick="pgTogglePop()">${pgLabel()} ⌄</button>
        ${p.adapter ? `<label class="switch">
          <input type="checkbox" ${p.compare?"checked":""} style="width:auto"
                 onchange="pgState().compare=this.checked"> compare base ↔ finetuned
          </label>` : ""}
        <span class="spacer" style="flex:1"></span>
        <button class="ghost" onclick="pgState().gear=!pgState().gear; pgShell()">⚙</button>
        <button class="ghost" onclick="S.pgv2.msgs=[]; pgShell()">clear</button>
        <div class="pop" id="pg-pop" style="display:${p.pop?"block":"none"}"></div>
      </div>
      ${p.gear ? `
      <div class="rowflex" style="padding-bottom:12px">
        <input placeholder="system prompt" value="${esc(p.sys)}" size="40"
               onchange="pgState().sys=this.value">
        <input type="number" value="${p.temp}" step="0.1" min="0" max="2"
               style="width:76px" title="temperature"
               onchange="pgState().temp=parseFloat(this.value)||0.7">
        <input type="number" value="${p.max}" min="1" style="width:86px"
               title="max new tokens" onchange="pgState().max=+this.value||256">
      </div>` : ""}
      ${p.msgs.length ? `
        <div id="pg-log"></div>
        <form id="pg-form">
          <input id="pg-in" placeholder="say something…" autocomplete="off">
          <button class="primary">›</button>
        </form>` : `
        <div class="hero">
          <h1>Talk to what you trained<span class="mark">.</span></h1>
          <div class="sub2">${p.adapter
            ? "compare is on — the base model and your finetune answer side by side"
            : "or any open model — pick a fine-tuned run to unlock compare"}</div>
          <form class="heroform" id="pg-form">
            <input id="pg-in" placeholder="say something…" autocomplete="off">
            <button class="primary">›</button>
          </form>
        </div>`}
    </div>`;
  if (p.pop) pgRenderPop();
  $("#pg-form").addEventListener("submit", pgSend);
  pgRender();
  const inp = $("#pg-in"); if (inp) inp.focus();
}

window.pgTogglePop = () => { const p = pgState(); p.pop = !p.pop; pgShell(); };

function pgRenderPop() {
  const p = pgState();
  const q = p.q.toLowerCase();
  let items = "";
  if (p.tab === "hub") {
    const seen = new Set();
    const rows = [];
    for (const id of S.models.recent) if (!seen.has(id)) { seen.add(id);
      rows.push({ id, meta: "recently trained here" }); }
    for (const m of S.models.catalog) if (!seen.has(m.id)) { seen.add(m.id);
      rows.push({ id: m.id, meta: (m.params||"") + (m.note ? " · " + m.note : ""),
                  gated: m.gated, dev: m.dev }); }
    const vis = rows.filter(r => r.id.toLowerCase().includes(q));
    items = vis.map(r => `
      <div class="item" onclick="pgPickHub('${r.id}')">
        <span>${esc(r.id)}</span>
        <span class="meta">${r.dev?'<span class="pill green">dev pick</span> ':""}
          ${r.gated?'<span class="pill gold">HF token</span> ':""}${esc(r.meta||"")}</span>
      </div>`).join("");
    if (q && !rows.some(r => r.id.toLowerCase() === q))
      items += `<div class="item" onclick="pgPickHub('${esc(p.q)}')">
        <span>use “${esc(p.q)}”</span><span class="meta">any HF hub id</span></div>`;
  } else {
    const done = S.jobs.filter(j => j.status === "succeeded"
      && (j.job_id.includes(q) || (j.base_model||"").toLowerCase().includes(q)));
    items = done.map(j => `
      <div class="item" onclick="pgPickTuned('${j.job_id}')">
        <span>${j.job_id.slice(0,10)} <span class="pill red">${j.method||""}</span></span>
        <span class="meta">${esc((j.base_model||"").split("/").pop())}
          ${j.final_loss!=null ? "· loss " + j.final_loss.toFixed(3) : ""}</span>
      </div>`).join("")
      || '<div class="item"><span class="meta">no fine-tuned runs yet — train one first</span></div>';
  }
  $("#pg-pop").innerHTML = `
    <div class="tabs">
      <button class="${p.tab==="hub"?"on":""}" onclick="pgTab('hub')">Hub models</button>
      <button class="${p.tab==="tuned"?"on":""}" onclick="pgTab('tuned')">Fine-tuned</button>
    </div>
    <input class="search" placeholder="search…" value="${esc(p.q)}"
           oninput="pgState().q=this.value; pgRenderPop()">
    <div class="list">${items}</div>`;
}
window.pgTab = t => { const p = pgState(); p.tab = t; pgRenderPop(); };
window.pgPickHub = id => { const p = pgState();
  p.model = id; p.adapter = null; p.compare = false; p.pop = false; pgShell(); };
window.pgPickTuned = id => { const p = pgState();
  const j = S.jobs.find(x => x.job_id === id);
  p.adapter = id; if (j) p.model = j.base_model; p.compare = true;
  p.pop = false; pgShell(); };

function pgRender() {
  const p = pgState();
  const log = $("#pg-log"); if (!log) return;
  log.innerHTML = p.msgs.map(m => {
    if (m.role === "user") return `<div class="msg you">${esc(m.content)}</div>`;
    if (m.compare) return `<div class="cmp">
        <div class="col base"><b>base</b>${esc(m.base ?? "…")}</div>
        <div class="col tuned"><b>finetuned ♥</b>${esc(m.tuned ?? "…")}</div>
      </div>`;
    return `<div class="msg slm">${esc(m.content ?? "…")}</div>`;
  }).join("");
  log.scrollTop = 1e9;
}

async function pgSend(e) {
  e.preventDefault();
  const p = pgState();
  if (p.busy) return;
  const text = $("#pg-in").value.trim(); if (!text) return;
  const firstTurn = p.msgs.length === 0;
  const compare = p.compare && p.adapter;
  p.msgs.push({ role:"user", content:text });
  const history = [];
  if (p.sys) history.push({ role:"system", content:p.sys });
  for (const m of p.msgs) {
    if (m.role === "user") history.push({ role:"user", content:m.content });
    else history.push({ role:"assistant",
                        content: m.compare ? (m.tuned ?? "") : (m.content ?? "") });
  }
  const turn = compare ? { role:"assistant", compare:true }
                       : { role:"assistant", content:null };
  p.msgs.push(turn); p.busy = true;
  if (firstTurn) pgShell(); else pgRender();

  const ask = ad => api("/v1/chat", { method:"POST", body:JSON.stringify({
      model:p.model, adapter:ad, messages:history, max_new_tokens:p.max,
      temperature:p.temp, top_p:0.95 })}).then(o => o.text)
      .catch(err => "⚠ " + err.message);
  try {
    if (compare) {
      turn.tuned = await ask(p.adapter); pgRender();  // serial: one GPU slot
      turn.base = await ask(null);
    } else {
      turn.content = await ask(p.adapter);
    }
  } finally { p.busy = false; pgRender(); }
}
</script>
</body>
</html>
"""
