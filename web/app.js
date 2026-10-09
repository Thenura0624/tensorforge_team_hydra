"use strict";
const $ = (id) => document.getElementById(id);
const state = {key: "", timer: null, job: null, resultJob: null, rows: [], offset: 0, total: 0, generation: 0};
const pretty = value => value ? value.replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase()) : "None";
function notify(message = "") { $("notice").textContent = message; $("notice").hidden = !message; }
function showPanel(name) {
  document.querySelectorAll(".panel").forEach(el => el.hidden = el.id !== name);
  document.querySelectorAll(".tab").forEach(el => { el.classList.toggle("active", el.dataset.panel === name); if (el.dataset.panel === name) el.setAttribute("aria-current", "page"); else el.removeAttribute("aria-current"); });
}
document.querySelectorAll(".tab").forEach(button => button.addEventListener("click", () => showPanel(button.dataset.panel)));
async function request(path, {method = "GET", body, publicRoute = false} = {}) {
  if (!publicRoute && !state.key) throw Error("Enter your team API key in Workspace access first.");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch(path, {method, credentials: "omit", cache: "no-store", signal: controller.signal,
      headers: {...(!publicRoute ? {"X-API-Key": state.key} : {}), ...(body ? {"Content-Type": "application/json"} : {})},
      body: body ? JSON.stringify(body) : undefined});
    const data = response.status === 204 ? null : await response.json();
    if (!response.ok) {
      const detail = data?.error?.details?.[0];
      const prefix = detail ? ` ${detail.index !== undefined ? `Ticket ${detail.index + 1}: ` : ""}${detail.field}: ${detail.issue}.` : "";
      throw Error((response.status === 401 ? "API key not accepted. Check the exact assigned key and try again." : data?.error?.message || `Request failed (${response.status}).`) + prefix);
    }
    return data;
  } catch (err) { if (err.name === "AbortError") throw Error("The request timed out. For a submitted job, check its job ID before retrying."); if (err instanceof TypeError) throw Error("Cannot reach the API. Check the server connection and try again."); throw err; }
  finally { clearTimeout(timeout); }
}
async function busy(button, fn) { const text = button.textContent; button.disabled = true; button.textContent = "Working…"; notify(); try { await fn(); } catch (err) { notify(err.message); } finally { button.disabled = false; button.textContent = text; } }
async function health() { try { const data = await request("/health", {publicRoute: true}); $("health-label").textContent = data.model_loaded === false ? "Model loading" : "API online"; $("health-dot").className = "dot online"; $("model-version").textContent = data.model_version || "Model loading"; } catch { $("health-label").textContent = "API unavailable"; $("health-dot").className = "dot offline"; } }
$("refresh-health").addEventListener("click", health); health();
$("key-form").addEventListener("submit", event => { event.preventDefault(); if (!$("api-key").value) return; state.key = $("api-key").value; $("api-key").value = ""; $("key-status").textContent = "Key set for this tab. It will be checked on your next request."; $("clear-key").hidden = false; notify(); });
$("clear-key").addEventListener("click", () => { state.key = ""; clearTimeout(state.timer); state.generation++; $("key-status").textContent = "Key cleared. Enter your team API key to begin."; $("api-key").required = true; $("clear-key").hidden = true; });
$("ticket-text").addEventListener("input", () => $("text-count").textContent = `${$("ticket-text").value.length.toLocaleString()} / 10,000`);
$("example").addEventListener("click", () => { $("channel").value = "chat"; $("subject").value = "Duplicate payment"; $("ticket-text").value = "I was charged twice for the same ride. Could you check the duplicate payment and refund it?"; $("ticket-text").dispatchEvent(new Event("input")); });
$("single-form").addEventListener("submit", event => { event.preventDefault(); busy($("classify"), async () => {
  if (!$("ticket-text").value.trim()) throw Error("Enter a customer message before classifying.");
  const ticket = {channel: $("channel").value, subject: $("subject").value, text: $("ticket-text").value};
  if ($("ticket-id").value) ticket.ticket_id = $("ticket-id").value;
  const data = await request("/predict", {method: "POST", body: ticket});
  $("result-empty").hidden = true; $("result").hidden = false;
  $("category").textContent = pretty(data.category); $("team").textContent = data.team;
  $("urgent").textContent = data.is_urgent ? "Urgent" : "Standard"; $("urgent").className = `pill${data.is_urgent ? " urgent" : ""}`;
  $("confidence").textContent = `${(data.confidence * 100).toFixed(1)}%`; $("confidence-bar").value = data.confidence;
  $("secondary").textContent = pretty(data.secondary_category); $("result-id").textContent = data.ticket_id || "Not provided";
  $("raw-result").textContent = JSON.stringify(data, null, 2);
}); });
function download(name, value) { const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type: "application/json"})); const a = document.createElement("a"); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
$("sample").addEventListener("click", () => download("hydra-example.json", {tickets: [{ticket_id: "001", channel: "chat", text: "My order is late."}, {ticket_id: "002", channel: "email", subject: "Lost bag", text: "I left my bag in the car. Can you help me contact the driver?"}]}));
$("batch-file").addEventListener("change", async () => { const file = $("batch-file").files[0]; if (!file) return; if (file.size > 25 * 1024 * 1024) { notify("Choose a file smaller than 25 MiB."); $("batch-file").value = ""; return; } try { $("batch-text").value = await file.text(); $("batch-file-name").textContent = file.name; notify(); } catch { notify("Could not read that file. Please choose it again."); } });
function parseTickets(text) {
  if (!text.trim()) throw Error("Upload a JSON/JSONL file or paste your tickets.");
  let data; try { data = JSON.parse(text); } catch { try { data = text.split(/\r?\n/).filter(line => line.trim()).map(line => JSON.parse(line)); } catch { throw Error("This is not valid JSON or JSONL. Download the example to see the expected format."); } }
  const tickets = Array.isArray(data) ? data : data?.tickets;
  if (!Array.isArray(tickets) || tickets.length < 1 || tickets.length > 5000) throw Error("Provide an array of 1–5,000 tickets, or an object with a tickets array.");
  const ids = new Set();
  tickets.forEach((ticket, index) => { if (!ticket || typeof ticket.ticket_id !== "string") throw Error(`Ticket ${index + 1} needs a string ticket_id.`); if (ids.has(ticket.ticket_id)) throw Error(`Ticket ${index + 1} has a duplicate ticket_id.`); ids.add(ticket.ticket_id); });
  const limit = (tickets.length <= 100 ? 5 : 25) * 1024 * 1024;
  if (new Blob([JSON.stringify({tickets})]).size > limit) throw Error(`Request exceeds the ${tickets.length <= 100 ? 5 : 25} MiB limit.`);
  return tickets;
}
$("batch-form").addEventListener("submit", event => { event.preventDefault(); busy($("run-batch"), async () => {
  const tickets = parseTickets($("batch-text").value);
  if (tickets.length <= 100) { const data = await request("/predict/batch", {method: "POST", body: {tickets}}); state.resultJob = null; state.rows = data.predictions; state.total = state.rows.length; state.offset = 0; renderRows(); }
  else { const data = await request("/batch/jobs", {method: "POST", body: {tickets}}); $("job-id").value = data.job_id; showPanel("jobs"); startTracking(data.job_id); }
}); });
function renderRows() {
  $("results-section").hidden = false; $("results-title").textContent = `${state.total.toLocaleString()} tickets classified`;
  $("results-body").replaceChildren();
  state.rows.forEach(row => { const tr = document.createElement("tr"); [row.ticket_id, pretty(row.category), row.team, row.is_urgent ? "Urgent" : "Standard", `${(row.confidence * 100).toFixed(1)}%`].forEach((value, index) => { const td = document.createElement("td"); if (index === 3) { const badge = document.createElement("span"); badge.className = `pill${row.is_urgent ? " urgent" : ""}`; badge.textContent = value; td.append(badge); } else td.textContent = value; tr.append(td); }); $("results-body").append(tr); });
  $("page-label").textContent = state.rows.length ? `Showing ${state.offset + 1}–${state.offset + state.rows.length} of ${state.total}` : "No results on this page";
  $("previous-page").disabled = !state.resultJob || state.offset === 0; $("next-page").disabled = !state.resultJob || state.offset + state.rows.length >= state.total;
}
async function loadPage(offset) { const data = await request(`/batch/jobs/${encodeURIComponent(state.resultJob)}/results?offset=${offset}&limit=100`); state.rows = data.predictions; state.offset = data.offset; state.total = data.total; renderRows(); }
$("previous-page").addEventListener("click", () => busy($("previous-page"), () => loadPage(Math.max(0, state.offset - 100))).then(renderRows));
$("next-page").addEventListener("click", () => busy($("next-page"), () => loadPage(state.offset + 100)).then(renderRows));
$("download-results").addEventListener("click", () => busy($("download-results"), async () => { const rows = state.resultJob ? (await request(`/batch/jobs/${encodeURIComponent(state.resultJob)}/results?offset=0&limit=5000`)).predictions : state.rows; download("hydra-predictions.json", {predictions: rows}); }));
$("job-form").addEventListener("submit", event => { event.preventDefault(); const id = $("job-id").value.trim(); if (id) startTracking(id); });
function startTracking(id) { clearTimeout(state.timer); $("results-section").hidden = true; $("job-info").hidden = true; state.resultJob = null; state.rows = []; state.job = id; const generation = ++state.generation; notify(); pollJob(generation); }
async function pollJob(generation) {
  try { const data = await request(`/batch/jobs/${encodeURIComponent(state.job)}`); if (generation !== state.generation) return;
    $("job-info").hidden = false; $("job-status").textContent = data.status; $("job-progress-text").textContent = `${data.processed.toLocaleString()} / ${data.total.toLocaleString()} tickets`; $("job-progress").max = data.total; $("job-progress").value = data.processed;
    $("job-detail").textContent = data.error?.message || (data.status === "succeeded" ? "Your results are ready. They are retained for at least six hours after completion." : ["queued", "running"].includes(data.status) ? "This job continues on the server if you close this tab. Keep its job ID to return later." : "This job is no longer processing.");
    $("view-job-results").hidden = data.status !== "succeeded"; $("cancel-job").hidden = !["queued", "running"].includes(data.status);
    if (["queued", "running"].includes(data.status)) state.timer = setTimeout(() => pollJob(generation), 3000);
  } catch (err) { if (generation === state.generation) { notify(`${err.message} Use Check job to retry.`); $("job-info").hidden = true; } }
}
$("view-job-results").addEventListener("click", () => busy($("view-job-results"), async () => { state.resultJob = state.job; await loadPage(0); }));
$("cancel-job").addEventListener("click", () => busy($("cancel-job"), async () => { await request(`/batch/jobs/${encodeURIComponent(state.job)}`, {method: "DELETE"}); clearTimeout(state.timer); await pollJob(++state.generation); }));
