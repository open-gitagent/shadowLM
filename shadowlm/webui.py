"""The built-in dashboard — a single-file SPA served at `/` by shadowlm.serve.

Vanilla HTML/CSS/JS over the same JSON protocol the SDK speaks: list jobs,
watch live loss curves, submit finetunes, chat with results. No build step,
no frameworks — view-source friendly, like the rest of the box.
"""

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ShadowLM · slm♥</title>
<style>
  :root {
    --ink: #16120E; --umbra: #221C16; --panel: #2A231C;
    --bone: #F2EAE0; --muted: #9A8F82; --heart: #E5484D; --ok: #3FB950;
    --warn: #D29922; --line: #3A3129;
  }
  * { box-sizing: border-box; margin: 0; }
  body {
    background: var(--ink); color: var(--bone);
    font: 14px/1.45 "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
    height: 100vh; display: flex; flex-direction: column;
  }
  header {
    display: flex; align-items: center; gap: 12px; padding: 12px 18px;
    border-bottom: 1px solid var(--line);
  }
  header .mark { color: var(--heart); font-weight: 700; }
  header .sub { color: var(--muted); font-size: 12px; }
  header .spacer { flex: 1; }
  input, select, textarea, button {
    background: var(--umbra); color: var(--bone); border: 1px solid var(--line);
    border-radius: 6px; padding: 7px 10px; font: inherit;
  }
  input:focus, select:focus, textarea:focus { outline: 1px solid var(--heart); }
  button { cursor: pointer; }
  button.primary { background: var(--heart); border-color: var(--heart);
                   color: #fff; font-weight: 700; }
  button.ghost { background: transparent; }
  main { flex: 1; display: flex; min-height: 0; }
  /* jobs rail */
  #rail { width: 270px; border-right: 1px solid var(--line); overflow-y: auto;
          padding: 10px; }
  #rail h3 { font-size: 11px; color: var(--muted); text-transform: uppercase;
             letter-spacing: .12em; margin: 8px 6px; }
  .job { padding: 8px 10px; border-radius: 8px; cursor: pointer;
         border: 1px solid transparent; margin-bottom: 4px; }
  .job:hover { background: var(--umbra); }
  .job.sel { border-color: var(--heart); background: var(--umbra); }
  .job .id { font-weight: 700; font-size: 13px; }
  .job .meta { color: var(--muted); font-size: 11px; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         margin-right: 6px; }
  .dot.succeeded { background: var(--ok); } .dot.failed { background: var(--heart); }
  .dot.running { background: var(--warn); animation: pulse 1.2s infinite; }
  .dot.pending { background: var(--muted); } .dot.stopped { background: var(--warn); }
  @keyframes pulse { 50% { opacity: .35; } }
  /* center */
  #center { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  #detail { flex: 1; padding: 18px; overflow-y: auto; }
  .row { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
  .badge { padding: 2px 10px; border-radius: 999px; font-size: 12px;
           border: 1px solid var(--line); }
  .badge.succeeded { color: var(--ok); border-color: var(--ok); }
  .badge.failed { color: var(--heart); border-color: var(--heart); }
  .badge.running, .badge.stopped { color: var(--warn); border-color: var(--warn); }
  svg { width: 100%; height: 260px; margin-top: 14px; background: var(--umbra);
        border: 1px solid var(--line); border-radius: 10px; }
  .stat { color: var(--muted); font-size: 12px; }
  .stat b { color: var(--bone); }
  .err { color: var(--heart); white-space: pre-wrap; margin-top: 10px; }
  /* new run form */
  #newrun { border-top: 1px solid var(--line); padding: 14px 18px;
            display: grid; gap: 8px;
            grid-template-columns: 2fr 1fr 1fr 1fr 1fr auto; }
  #newrun textarea { grid-column: 1 / -1; height: 74px; resize: vertical; }
  #newrun .hintline { grid-column: 1 / -1; color: var(--muted); font-size: 11px; }
  /* chat rail */
  #chat { width: 330px; border-left: 1px solid var(--line); display: flex;
          flex-direction: column; }
  #chat h3 { font-size: 11px; color: var(--muted); text-transform: uppercase;
             letter-spacing: .12em; padding: 12px 14px 4px; }
  #chatlog { flex: 1; overflow-y: auto; padding: 10px 14px; display: flex;
             flex-direction: column; gap: 8px; }
  .msg { padding: 8px 10px; border-radius: 10px; font-size: 13px;
         white-space: pre-wrap; max-width: 95%; }
  .msg.you { background: var(--panel); align-self: flex-end; }
  .msg.slm { background: var(--umbra); border: 1px solid var(--line); }
  .msg.slm::before { content: "slm♥ "; color: var(--heart); font-weight: 700; }
  #chatform { display: flex; gap: 8px; padding: 10px 14px;
              border-top: 1px solid var(--line); }
  #chatform input { flex: 1; }
  a { color: var(--heart); }
</style>
</head>
<body>
<header>
  <span class="mark">slm♥</span>
  <span><b>ShadowLM</b> server</span>
  <span class="sub" id="health">connecting…</span>
  <span class="spacer"></span>
  <input id="apikey" type="password" placeholder="API key (if required)"
         size="22" title="Sent as Bearer auth; stored in this browser only">
</header>
<main>
  <nav id="rail">
    <h3>Runs</h3>
    <div id="jobs"></div>
  </nav>
  <section id="center">
    <div id="detail"><p class="stat">No runs yet — submit one below.</p></div>
    <form id="newrun">
      <textarea id="f-data" placeholder='dataset rows, one JSON per line — e.g. {"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello!"}]}'></textarea>
      <input id="f-model" placeholder="base model (HF id)" required
             value="mlx-community/Qwen2.5-0.5B-Instruct-4bit">
      <select id="f-method"></select>
      <input id="f-steps" type="number" placeholder="max steps" value="60" min="1">
      <input id="f-lorar" type="number" placeholder="lora r" value="16" min="1">
      <input id="f-lr" placeholder="lr (per-method default)">
      <button class="primary" type="submit">finetune ›</button>
      <div class="hintline">any open model · any harness · any method — jobs run on this server's backend</div>
    </form>
  </section>
  <aside id="chat">
    <h3>Chat <span id="chat-with" style="text-transform:none"></span></h3>
    <div id="chatlog"></div>
    <form id="chatform">
      <input id="chat-in" placeholder="message the selected run's model…">
      <button class="primary">›</button>
    </form>
  </aside>
</main>
<script>
const $ = (s) => document.querySelector(s);
const state = { jobs: [], sel: null, msgs: [], health: null };

const key = () => localStorage.getItem("slm_api_key") || "";
$("#apikey").value = key();
$("#apikey").addEventListener("change", e => localStorage.setItem("slm_api_key", e.target.value));

async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json" };
  if (key()) headers["Authorization"] = "Bearer " + key();
  const r = await fetch(path, { ...opts, headers });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}

// ---- health + methods (once) -------------------------------------------------
(async () => {
  try {
    const h = await api("/v1/health");
    state.health = h;
    $("#health").textContent = `backend=${h.backend} · v${h.version}`;
    const m = await api("/v1/methods");
    $("#f-method").innerHTML = m.methods.map(x =>
      `<option value="${x.name}" title="${x.description}">${x.name}</option>`).join("");
  } catch (e) { $("#health").textContent = "⚠ " + e.message; }
})();

// ---- jobs poll -----------------------------------------------------------------
async function refreshJobs() {
  try {
    const { jobs } = await api("/v1/finetunes");
    state.jobs = jobs;
    $("#jobs").innerHTML = jobs.length ? jobs.map(j => `
      <div class="job ${j.job_id === state.sel ? "sel" : ""}" data-id="${j.job_id}">
        <div class="id"><span class="dot ${j.status}"></span>${j.job_id.slice(0, 8)}</div>
        <div class="meta">${j.method || "?"} · ${j.status} · ${j.steps} steps</div>
        <div class="meta">${j.base_model.split("/").pop()}</div>
      </div>`).join("") : '<p class="stat" style="padding:6px">none yet</p>';
    document.querySelectorAll(".job").forEach(el =>
      el.addEventListener("click", () => { state.sel = el.dataset.id; state.msgs = []; renderChatLog(); }));
    if (!state.sel && jobs.length) state.sel = jobs[0].job_id;
    if (state.sel) refreshDetail();
  } catch (e) { /* server briefly busy training — keep polling */ }
}

// ---- detail + chart -------------------------------------------------------------
function chartSVG(steps, evals) {
  const pts = steps.map(s => s.loss), W = 800, H = 250, P = 34;
  if (!pts.length) return '<p class="stat">waiting for first metric…</p>';
  const all = pts.concat(evals.map(e => e.loss));
  const lo = Math.min(...all), hi = Math.max(...all), span = (hi - lo) || 1;
  const x = i => P + i * (W - 2 * P) / Math.max(1, pts.length - 1);
  const y = v => H - P - (v - lo) * (H - 2 * P) / span;
  const line = pts.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const evDots = evals.map(e => {
    const i = Math.min(pts.length - 1, Math.max(0, e.step - 1));
    return `<circle cx="${x(i)}" cy="${y(e.loss)}" r="4" fill="none" stroke="#D29922" stroke-width="2"/>`;
  }).join("");
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <text x="${P}" y="${P - 12}" fill="#9A8F82" font-size="11">loss ${hi.toFixed(3)}</text>
    <text x="${P}" y="${H - P + 16}" fill="#9A8F82" font-size="11">${lo.toFixed(3)}</text>
    <path d="${line}" fill="none" stroke="#E5484D" stroke-width="2.5"
          stroke-linejoin="round" stroke-linecap="round"/>
    ${evDots}</svg>`;
}

async function refreshDetail() {
  const id = state.sel; if (!id) return;
  try {
    const [job, m] = await Promise.all([
      api(`/v1/finetunes/${id}`), api(`/v1/finetunes/${id}/metrics`)]);
    const last = m.steps[m.steps.length - 1];
    const j = state.jobs.find(x => x.job_id === id) || {};
    $("#chat-with").textContent = `· ${id.slice(0, 8)}`;
    $("#detail").innerHTML = `
      <div class="row">
        <h2 style="font-size:16px">${id}</h2>
        <span class="badge ${job.status}">${job.status}</span>
        ${job.status === "running" || job.status === "pending"
          ? `<button class="ghost" onclick="cancelJob('${id}')">cancel</button>` : ""}
      </div>
      <p class="stat" style="margin-top:6px">
        <b>${j.base_model || ""}</b> · method <b>${j.method || "?"}</b>
        ${last ? ` · step <b>${last.step}</b> · loss <b>${last.loss.toFixed(4)}</b>` : ""}
        ${job.final_loss != null ? ` · final <b>${job.final_loss.toFixed(4)}</b>` : ""}
        ${last && last.tokens_per_s ? ` · ${Math.round(last.tokens_per_s)} tok/s` : ""}
      </p>
      ${chartSVG(m.steps, m.evals)}
      ${job.error ? `<div class="err">${job.error}</div>` : ""}`;
  } catch (e) { /* transient */ }
}

window.cancelJob = async (id) => { await api(`/v1/finetunes/${id}/cancel`, { method: "POST" }); };

// ---- submit a run ----------------------------------------------------------------
$("#newrun").addEventListener("submit", async (e) => {
  e.preventDefault();
  const rows = $("#f-data").value.trim().split("\n").filter(Boolean).map(l => JSON.parse(l));
  if (!rows.length) return alert("paste at least one JSONL row");
  const config = { method: $("#f-method").value,
                   max_steps: parseInt($("#f-steps").value || "60"),
                   lora_r: parseInt($("#f-lorar").value || "16") };
  if ($("#f-lr").value) config.learning_rate = parseFloat($("#f-lr").value);
  const out = await api("/v1/finetunes", { method: "POST", body: JSON.stringify({
    base_model: $("#f-model").value, config,
    dataset: { format: "chat", rows },
    eval_dataset: null, load_in_4bit: false, max_seq_length: 2048 }) });
  state.sel = out.job_id;
});

// ---- chat -----------------------------------------------------------------------
function renderChatLog() {
  $("#chatlog").innerHTML = state.msgs.map(m =>
    `<div class="msg ${m.role === "user" ? "you" : "slm"}">${m.content}</div>`).join("");
  $("#chatlog").scrollTop = 1e9;
}
$("#chatform").addEventListener("submit", async (e) => {
  e.preventDefault();
  const j = state.jobs.find(x => x.job_id === state.sel);
  if (!j) return alert("select a run first");
  const text = $("#chat-in").value.trim(); if (!text) return;
  $("#chat-in").value = "";
  state.msgs.push({ role: "user", content: text }); renderChatLog();
  state.msgs.push({ role: "assistant", content: "…" }); renderChatLog();
  try {
    const out = await api("/v1/chat", { method: "POST", body: JSON.stringify({
      model: j.base_model, adapter: j.status === "succeeded" ? j.job_id : null,
      messages: state.msgs.slice(0, -1), max_new_tokens: 256,
      temperature: 0.7, top_p: 0.95 }) });
    state.msgs[state.msgs.length - 1].content = out.text;
  } catch (err) {
    state.msgs[state.msgs.length - 1].content = "⚠ " + err.message;
  }
  renderChatLog();
});

setInterval(refreshJobs, 1500);
refreshJobs();
</script>
</body>
</html>
"""
