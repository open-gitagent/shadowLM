"""The built-in studio dashboard — a single-file SPA served at `/`.

Four sections, the workflow in order: Datasets → Models → Trainings → Runs.
Vanilla HTML/CSS/JS over the same JSON protocol the SDK speaks — no build
step, no frameworks, view-source friendly, like the rest of the box.
"""

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ShadowLM · slm♥</title>
<style>
  :root {
    --ink:#16120E; --umbra:#221C16; --panel:#2A231C; --bone:#F2EAE0;
    --muted:#9A8F82; --heart:#E5484D; --ok:#3FB950; --warn:#D29922;
    --line:#3A3129;
  }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--ink); color:var(--bone); height:100vh; display:flex;
         flex-direction:column;
         font:14px/1.5 "JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace; }
  header { display:flex; align-items:center; gap:18px; padding:12px 20px;
           border-bottom:1px solid var(--line); }
  .mark { color:var(--heart); font-weight:700; }
  nav { display:flex; gap:4px; }
  nav a { color:var(--muted); text-decoration:none; padding:6px 14px;
          border-radius:8px; font-size:13px; }
  nav a.on { color:var(--bone); background:var(--umbra);
             box-shadow:inset 0 -2px 0 var(--heart); }
  nav a:hover { color:var(--bone); }
  .spacer { flex:1; }
  .sub { color:var(--muted); font-size:12px; }
  input,select,textarea,button { background:var(--umbra); color:var(--bone);
    border:1px solid var(--line); border-radius:8px; padding:8px 11px; font:inherit; }
  input:focus,select:focus,textarea:focus { outline:1px solid var(--heart); }
  button { cursor:pointer; }
  button.primary { background:var(--heart); border-color:var(--heart);
                   color:#fff; font-weight:700; }
  button.ghost { background:transparent; }
  button.danger { color:var(--heart); background:transparent; }
  main { flex:1; overflow-y:auto; padding:22px 26px; }
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
  /* chat drawer on run detail */
  #chatlog { display:flex; flex-direction:column; gap:8px; margin-top:10px; }
  .msg { padding:9px 11px; border-radius:10px; font-size:13px;
         white-space:pre-wrap; max-width:75%; }
  .msg.you { background:var(--panel); align-self:flex-end; }
  .msg.slm { background:var(--umbra); border:1px solid var(--line); }
  .msg.slm::before { content:"slm♥ "; color:var(--heart); font-weight:700; }
  .back { color:var(--heart); text-decoration:none; font-size:12.5px; }
</style>
</head>
<body>
<header>
  <span class="mark">slm♥</span><b>ShadowLM</b>
  <nav id="nav">
    <a href="#datasets">Datasets</a><a href="#models">Models</a>
    <a href="#train">Trainings</a><a href="#runs">Runs</a>
  </nav>
  <span class="spacer"></span>
  <span class="sub" id="health">connecting…</span>
  <input id="apikey" type="password" placeholder="API key" size="14"
         title="Sent as Bearer auth; stored in this browser only">
</header>
<main id="page"></main>
<script>
const $ = s => document.querySelector(s);
const S = { datasets:[], models:{catalog:[],recent:[]}, methods:[], jobs:[],
            pick:{dataset:null, model:null}, run:null, msgs:[], timer:null };

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
                 runs: pageRuns };
function route() {
  clearInterval(S.timer); S.timer = null;
  const [name, arg] = location.hash.replace("#","").split("/");
  document.querySelectorAll("#nav a").forEach(a =>
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

// ====== TRAININGS (configure & launch) ============================================
async function pageTrain() {
  if (!S.datasets.length) S.datasets = (await api("/v1/datasets")).datasets;
  if (!S.methods.length) S.methods = (await api("/v1/methods")).methods;
  const mOpts = S.methods.map(m =>
    `<option value="${m.name}" ${m.name==="lora"?"selected":""}>${m.name} — ${esc(m.description.slice(0,58))}</option>`).join("");
  $("#page").innerHTML = `
    <h2>New training</h2>
    <p class="lead">dataset → model → method. Everything else has defaults.</p>
    <form class="stack" id="trainform">
      <div class="pair"><label>dataset</label>
        <select id="t-ds" required>
          <option value="">— pick a dataset —</option>
          ${S.datasets.map(d => `<option value="${d.dataset_id}"
            ${S.pick.dataset===d.dataset_id?"selected":""}>${esc(d.name)} (${d.rows} rows, ${d.format})</option>`).join("")}
        </select></div>
      <div class="pair"><label>model</label>
        <input id="t-model" required placeholder="HF hub id — see Models tab"
               value="${esc(S.pick.model || "mlx-community/Qwen2.5-0.5B-Instruct-4bit")}"></div>
      <div class="pair"><label>method</label><select id="t-method">${mOpts}</select></div>
      <div class="pair"><label>max steps</label><input id="t-steps" type="number" value="60" min="1"></div>
      <div class="pair"><label>lora r</label><input id="t-lorar" type="number" value="16" min="1"></div>
      <div class="pair"><label>learning rate</label><input id="t-lr" placeholder="per-method default"></div>
      <div class="pair"><label>batch size</label><input id="t-batch" type="number" value="2" min="1"></div>
      <div class="pair"><label>held-out eval</label>
        <label class="rowflex"><input type="checkbox" id="t-eval" style="width:auto"> hold out 10% — watch for overfitting</label></div>
      <div class="pair"><label></label>
        <button class="primary" style="justify-self:start">start training ›</button></div>
    </form>`;
  $("#trainform").addEventListener("submit", async e => {
    e.preventDefault();
    const config = { method: $("#t-method").value,
      max_steps: +$("#t-steps").value || 60, lora_r: +$("#t-lorar").value || 16,
      per_device_train_batch_size: +$("#t-batch").value || 2 };
    if ($("#t-lr").value) config.learning_rate = parseFloat($("#t-lr").value);
    const out = await api("/v1/finetunes", {method:"POST", body:JSON.stringify({
      base_model: $("#t-model").value, config,
      dataset_id: $("#t-ds").value,
      eval_dataset: $("#t-eval").checked ? "auto" : null,
      load_in_4bit: false, max_seq_length: 2048 })});
    location.hash = "#runs/" + out.job_id;
  });
}

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
    const chatable = job.status === "succeeded";
    $("#page").innerHTML = `
      <a class="back" href="#runs">‹ all runs</a>
      <div class="rowflex" style="margin-top:8px">
        <h2>${id}</h2>
        <span class="pill ${job.status==="succeeded"?"green":job.status==="failed"?"red":"gold"}">${job.status}</span>
        ${(job.status==="running"||job.status==="pending")
          ? `<button class="ghost" onclick="cancelRun('${id}')">cancel</button>` : ""}
      </div>
      <p class="sub" style="margin-top:4px">
        ${esc(j.base_model||"")} · ${j.method||"?"}
        ${last ? `· step ${last.step} · loss ${last.loss.toFixed(4)}` : ""}
        ${job.final_loss != null ? `· final ${job.final_loss.toFixed(4)}` : ""}
        ${last && last.tokens_per_s ? `· ${Math.round(last.tokens_per_s)} tok/s` : ""}
      </p>
      ${chartSVG(m.steps, m.evals)}
      ${job.error ? `<div class="err">${esc(job.error)}</div>` : ""}
      ${chatable ? `
        <h2 style="margin-top:8px">Chat with this run</h2>
        <div id="chatlog">${S.msgs.map(x => `<div class="msg ${x.role==="user"?"you":"slm"}">${esc(x.content)}</div>`).join("")}</div>
        <form class="rowflex" style="margin-top:10px" id="chatform">
          <input id="chat-in" placeholder="message the finetuned model…" style="flex:1">
          <button class="primary">›</button>
        </form>` : ""}`;
    if (chatable) $("#chatform").addEventListener("submit", ev => sendChat(ev, j));
  };
  await render();
  S.timer = setInterval(render, 1800);
}
window.cancelRun = async id => api(`/v1/finetunes/${id}/cancel`, {method:"POST"});

async function sendChat(e, j) {
  e.preventDefault();
  const text = $("#chat-in").value.trim(); if (!text) return;
  S.msgs.push({role:"user", content:text});
  S.msgs.push({role:"assistant", content:"…"});
  try {
    const out = await api("/v1/chat", {method:"POST", body:JSON.stringify({
      model: j.base_model, adapter: j.job_id,
      messages: S.msgs.slice(0,-1), max_new_tokens: 256,
      temperature: 0.7, top_p: 0.95 })});
    S.msgs[S.msgs.length-1].content = out.text;
  } catch(err) { S.msgs[S.msgs.length-1].content = "⚠ " + err.message; }
}
</script>
</body>
</html>
"""
