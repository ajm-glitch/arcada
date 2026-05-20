/**
 * aRCADA chat frontend
 * Hybrid BM25 + semantic retrieval over the instrument catalog,
 * then Cloudflare Worker /plan → /dispatch → /status polling.
 */

let CONFIG = {};
let CHUNKS = [];         // [{id, title, type, source, location, keywords, text}, ...]
let EMBEDDINGS = null;   // Float32Array, shape [N × D]
let EMBED_DIM  = 768;
let miniSearch = null;
let WORKER_URL = "";
let PASSWORD   = "";
let INSTRUMENT_CATALOG = []; // deduplicated instrument-only chunks for /plan validation

// ── Boot ──────────────────────────────────────────────────────────────────────

async function boot() {
  CONFIG     = await fetch("config.json").then(r => r.json());
  WORKER_URL = CONFIG.workerUrl;
  EMBED_DIM  = CONFIG.embedDim || 768;

  [CHUNKS, EMBEDDINGS] = await Promise.all([
    fetch("../../public/chunks.json").then(r => r.json()),
    fetch("../../public/embeddings.bin")
      .then(r => r.arrayBuffer())
      .then(buf => new Float32Array(buf)),
  ]);

  // Build deduplicated instrument catalog for /plan validation (exclude paper/site-context chunks)
  const REFERENCE_TYPES = new Set(["paper", "site-context"]);
  const seenIds = new Set();
  INSTRUMENT_CATALOG = CHUNKS.filter(c => {
    if (REFERENCE_TYPES.has(c.type)) return false;
    const baseId = c.id.replace(/::.*$/, ""); // strip ::desc / ::params suffixes
    if (seenIds.has(baseId)) return false;
    seenIds.add(baseId);
    return true;
  });

  miniSearch = new MiniSearch({
    idField: "miniSearchId",
    fields: ["title", "text", "keywords", "type", "location"],
    storeFields: ["title", "type", "source", "location"],
    searchOptions: { boost: { title: 2, keywords: 1.5 }, fuzzy: 0.2 },
  });
  miniSearch.addAll(
    CHUNKS.map((c, i) => ({ ...c, miniSearchId: i }))
  );

  document.getElementById("inputForm").addEventListener("submit", onSubmit);
  document.getElementById("modalClose").addEventListener("click", () => {
    document.getElementById("modalOverlay").hidden = true;
  });
  document.querySelectorAll(".example-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.getElementById("queryInput").value = btn.dataset.query;
      document.getElementById("queryInput").focus();
    });
  });
}

// ── Retrieval ─────────────────────────────────────────────────────────────────

function bm25Search(query, k = 10) {
  return miniSearch.search(query).slice(0, k).map(r => {
    return { idx: r.id, score: r.score };
  });
}

async function semanticSearch(query, k = 10) {
  const res = await workerPost("/embed", { text: query });
  const qvec = new Float32Array(res.embedding);
  l2Normalize(qvec);

  const N = EMBEDDINGS.length / EMBED_DIM;
  const scores = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let dot = 0;
    const off = i * EMBED_DIM;
    for (let d = 0; d < EMBED_DIM; d++) dot += qvec[d] * EMBEDDINGS[off + d];
    scores[i] = dot;
  }

  return Array.from(scores)
    .map((s, i) => ({ idx: i, score: s }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

function hybridFuse(bm25, semantic, k = CONFIG.topK || 6) {
  const map = new Map();
  bm25.forEach(({ idx, score }, rank) => {
    map.set(idx, (map.get(idx) || 0) + 1 / (rank + 1) * 0.5);
  });
  semantic.forEach(({ idx, score }, rank) => {
    map.set(idx, (map.get(idx) || 0) + 1 / (rank + 1) * 0.5);
  });
  return [...map.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, k)
    .map(([idx]) => CHUNKS[idx])
    .filter(Boolean);
}

function l2Normalize(vec) {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < vec.length; i++) vec[i] /= norm;
}

// ── Worker calls ──────────────────────────────────────────────────────────────

async function workerPost(path, body) {
  const headers = { "Content-Type": "application/json" };
  if (PASSWORD) headers["Authorization"] = `Bearer ${PASSWORD}`;
  const r = await fetch(WORKER_URL + path, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`Worker ${path} returned ${r.status}`);
  return r.json();
}

async function workerGet(path) {
  const headers = {};
  if (PASSWORD) headers["Authorization"] = `Bearer ${PASSWORD}`;
  const r = await fetch(WORKER_URL + path, { headers });
  if (!r.ok) throw new Error(`Worker ${path} returned ${r.status}`);
  return r.json();
}

async function streamChat(query, context) {
  const headers = { "Content-Type": "application/json" };
  if (PASSWORD) headers["Authorization"] = `Bearer ${PASSWORD}`;
  return fetch(WORKER_URL + "/chat", {
    method: "POST",
    headers,
    body: JSON.stringify({ query, context }),
  });
}

// ── UI helpers ─────────────────────────────────────────────────────────────────

function addMessage(role, content) {
  const list = document.getElementById("messageList");
  const div = document.createElement("div");
  div.className = `message ${role}`;
  const label = role === "user" ? "Your request" : role === "assistant" ? "aRCADA" : "Error";
  div.innerHTML = `<div class="message-label">${label}</div><div class="message-content"></div>`;
  div.querySelector(".message-content").textContent = content;
  list.appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return div.querySelector(".message-content");
}

function setStatus(text) {
  const bar = document.getElementById("statusBar");
  document.getElementById("statusText").textContent = text;
  bar.hidden = !text;
}

function renderInstruments(chunks) {
  const list = document.getElementById("instrumentList");
  const hint = document.querySelector(".panel-hint");
  list.innerHTML = "";
  if (hint) hint.style.display = "none";

  const seen = new Set();
  chunks.forEach((c, i) => {
    if (seen.has(c.id)) return;
    seen.add(c.id);
    const card = document.createElement("div");
    card.className = `instrument-card ${i < 2 ? "primary" : "secondary"}`;
    card.innerHTML = `
      <div class="instrument-card-name">${c.title}</div>
      <div class="instrument-card-meta">${c.location || ""}</div>
      <span class="instrument-card-badge badge-${c.source}">${c.source}</span>
    `;
    list.appendChild(card);
  });
}

function showDownloadModal(metadata) {
  const body = document.getElementById("modalBody");
  const dl   = document.getElementById("downloadBtn");

  const instruments = (metadata.instruments || []).map(i =>
    `<li><strong>${i.instrument_name}</strong> — ${i.n_records} records, ${i.coverage_start?.slice(0, 10)} to ${i.coverage_end?.slice(0, 10)}${i.gaps?.length ? ` (${i.gaps.length} gap${i.gaps.length > 1 ? "s" : ""})` : ""}</li>`
  ).join("");

  body.innerHTML = `
    <p>Your data package is ready. ${metadata.instruments?.length || 0} instrument${metadata.instruments?.length !== 1 ? "s" : ""}, ${metadata.total_records?.toLocaleString() || "?"} total records.</p>
    <ul style="margin: 0.75rem 0 0.75rem 1.2rem; line-height: 1.8;">${instruments}</ul>
    <p>Download the <code>.zarr</code> archive and accompanying <code>metadata.json</code> from the GitHub Actions artifact (requires GitHub account).</p>
    <p style="margin-top:0.5rem;">Format: <code>Zarr v2</code> — open with <code>xarray.open_zarr()</code> or <code>zarr.open()</code>.</p>
  `;

  if (metadata.downloadUrl) {
    dl.href = metadata.downloadUrl;
    dl.hidden = false;
  } else {
    dl.hidden = true;
  }

  document.getElementById("modalOverlay").hidden = false;
}

// ── Main submit flow ──────────────────────────────────────────────────────────

async function onSubmit(e) {
  e.preventDefault();
  const query = document.getElementById("queryInput").value.trim();
  if (!query) return;

  const submitBtn = document.getElementById("submitBtn");
  submitBtn.disabled = true;
  document.getElementById("queryInput").value = "";

  addMessage("user", query);
  setStatus("Searching instrument catalog…");

  try {
    // 1. Hybrid retrieval
    const [bm25, semantic] = await Promise.all([
      bm25Search(query),
      semanticSearch(query),
    ]);
    const context = hybridFuse(bm25, semantic);
    renderInstruments(context);

    // 2. Stream conversational response
    setStatus("Generating response…");
    const contentEl = addMessage("assistant", "");
    const resp = await streamChat(query, context);
    if (!resp.ok) throw new Error(`Chat stream failed: ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6).trim();
        if (raw === "[DONE]") break;
        try {
          const chunk = JSON.parse(raw);
          const text = chunk?.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
          fullText += text;
          contentEl.textContent = fullText;
        } catch { /* partial JSON — skip */ }
      }
    }

    // 3. Build data plan
    setStatus("Building data plan…");
    const { plan } = await workerPost("/plan", { query, context, catalog: INSTRUMENT_CATALOG });

    if (plan?.instruments?.length) {
      const planEl = addMessage("assistant", "");
      planEl.innerHTML = `<strong>Data plan:</strong><pre>${JSON.stringify(plan, null, 2)}</pre>`;

      // 4. Dispatch GitHub Actions job
      setStatus("Dispatching data pull job…");
      const { runId } = await workerPost("/dispatch", { plan });

      if (runId) {
        // 5. Poll for completion
        setStatus(`Job running (ID ${runId}) — this may take several minutes…`);
        await pollJob(runId, plan);
      } else {
        setStatus("");
        addMessage("error", "Could not start data pull job. Check Worker configuration.");
      }
    } else {
      setStatus("");
      addMessage("assistant", "No instruments matched your request. Try rephrasing or check the example queries.");
    }

  } catch (err) {
    setStatus("");
    addMessage("error", `Error: ${err.message}`);
    console.error(err);
  } finally {
    submitBtn.disabled = false;
  }
}

async function pollJob(runId, plan) {
  const interval = CONFIG.pollIntervalMs || 8000;
  const timeout  = CONFIG.pollTimeoutMs  || 3_600_000;
  const deadline = Date.now() + timeout;

  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, interval));
    try {
      const status = await workerGet(`/status/${runId}`);
      setStatus(`Job ${runId}: ${status.status}…`);

      if (status.status === "completed") {
        setStatus("");
        if (status.conclusion === "success") {
          showDownloadModal({ ...status, instruments: plan?.instruments || [] });
        } else {
          addMessage("error", `Data pull job failed: ${status.conclusion}. Check GitHub Actions logs.`);
        }
        return;
      }
    } catch (e) {
      console.warn("Poll error:", e);
    }
  }

  setStatus("");
  addMessage("error", "Timed out waiting for data pull job. Check GitHub Actions for job status.");
}

boot().catch(console.error);
