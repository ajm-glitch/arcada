/**
 * aRCADA chat frontend
 * Hybrid BM25 + semantic retrieval, multi-turn conversation,
 * intent classification, and GitHub Actions data pull.
 */

let CONFIG = {};
let CHUNKS = [];
let EMBEDDINGS = null;
let EMBED_DIM  = 768;
let miniSearch = null;
let WORKER_URL = "";
let PASSWORD   = "";
let INSTRUMENT_CATALOG = [];

// ── Conversation state ────────────────────────────────────────────────────────

let CONV_STATE      = "idle";   // "idle" | "awaiting_clarification"
let PENDING_QUERY   = null;
let PENDING_CONTEXT = null;
// History in Gemini format: [{role:"user"|"model", parts:[{text}]}]
let HISTORY = [];

// ── Boot ──────────────────────────────────────────────────────────────────────

async function boot() {
  initTheme();
  CONFIG     = await fetch("config.json").then(r => r.json());
  WORKER_URL = CONFIG.workerUrl;
  EMBED_DIM  = CONFIG.embedDim || 768;

  // Probe worker; show password gate if 401
  const accessible = await probeWorker();
  if (!accessible) await promptPassword();

  setStatus("Loading catalog…");
  [CHUNKS, EMBEDDINGS] = await Promise.all([
    fetch("../public/chunks.json").then(r => r.json()),
    fetch("../public/embeddings.bin")
      .then(r => r.arrayBuffer())
      .then(buf => new Float32Array(buf)),
  ]);
  setStatus("");

  // Build deduplicated instrument-only catalog for /plan validation
  const REFERENCE_TYPES = new Set(["paper", "site-context"]);
  const seenIds = new Set();
  INSTRUMENT_CATALOG = CHUNKS.filter(c => {
    if (REFERENCE_TYPES.has(c.type)) return false;
    const baseId = c.id.replace(/::.*$/, "");
    if (seenIds.has(baseId)) return false;
    seenIds.add(baseId);
    return true;
  }).map(c => ({ ...c, id: c.id.replace(/::.*$/, "") }));

  miniSearch = new MiniSearch({
    idField: "miniSearchId",
    fields: ["title", "text", "keywords", "type", "location"],
    storeFields: ["title", "type", "source", "location"],
    searchOptions: { boost: { title: 2, keywords: 1.5 }, fuzzy: 0.2 },
  });
  miniSearch.addAll(CHUNKS.map((c, i) => ({ ...c, miniSearchId: i })));

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

  // Show welcome (async, appears while user reads the interface)
  showWelcome();
}

// ── Password gate ─────────────────────────────────────────────────────────────

async function probeWorker() {
  try {
    const headers = { "Content-Type": "application/json" };
    if (PASSWORD) headers["Authorization"] = `Bearer ${PASSWORD}`;
    const r = await fetch(WORKER_URL + "/embed", {
      method: "POST", headers,
      body: JSON.stringify({ text: "probe" }),
    });
    return r.status !== 401;
  } catch { return true; }
}

function promptPassword() {
  return new Promise(resolve => {
    const gate  = document.getElementById("passwordGate");
    const form  = document.getElementById("passwordForm");
    const errEl = document.getElementById("passwordError");
    const input = document.getElementById("passwordInput");
    gate.hidden = false;

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const pw = input.value.trim();
      if (!pw) return;
      PASSWORD = pw;
      const ok = await probeWorker();
      if (ok) {
        gate.hidden  = true;
        errEl.hidden = true;
        resolve();
      } else {
        PASSWORD    = "";
        input.value = "";
        errEl.hidden = false;
      }
    });
  });
}

// ── Welcome message ───────────────────────────────────────────────────────────

async function showWelcome() {
  const contentEl = addMessage("assistant", "");
  contentEl.innerHTML = '<span class="typing-dots"><span>.</span><span>.</span><span>.</span></span>';

  try {
    const papers  = CHUNKS.filter(c => c.type === "paper" && c.first_author && c.year);
    const samples = sampleDiversePapers(papers, 3);
    const res = await workerPost("/welcome", {
      paperSamples: samples.map(p => ({ title: p.title, first_author: p.first_author, year: p.year })),
    });
    const text = res.text || "Hello! I'm aRCADA. What would you like to explore today?";
    contentEl.innerHTML = renderMarkdown(text);
    // Seed history with a synthetic exchange so subsequent turns have context
    HISTORY.push({ role: "user",  parts: [{ text: "Hello!" }] });
    HISTORY.push({ role: "model", parts: [{ text: text }] });
  } catch {
    const fallback = "Hello! I'm aRCADA, your Regional Cabled Array data assistant. I can help you access seismic, geochemical, acoustic, and oceanographic data from the Cascadia margin. What would you like to explore today?";
    contentEl.textContent = fallback;
    HISTORY.push({ role: "user",  parts: [{ text: "Hello!" }] });
    HISTORY.push({ role: "model", parts: [{ text: fallback }] });
  }
}

function sampleDiversePapers(papers, n) {
  const topics = ["seismic", "eruption", "methane", "hydrophone", "fin whale", "hydrothermal", "carbon", "tremor", "sonar", "ctd", "earthquake", "acoustic"];
  const picked = [];
  for (const topic of topics) {
    if (picked.length >= n) break;
    const match = papers.find(p =>
      !picked.includes(p) &&
      (p.title.toLowerCase().includes(topic) || (p.keywords || []).some(k => k.toLowerCase().includes(topic)))
    );
    if (match) picked.push(match);
  }
  // Fill remaining slots from unpicked papers
  const rest = papers.filter(p => !picked.includes(p));
  while (picked.length < n && rest.length) {
    picked.push(rest.splice(Math.floor(Math.random() * rest.length), 1)[0]);
  }
  return picked;
}

// ── Retrieval ─────────────────────────────────────────────────────────────────

function bm25Search(query, k = 10) {
  return miniSearch.search(query).slice(0, k).map(r => ({ idx: r.id, score: r.score }));
}

async function semanticSearch(query, k = 10) {
  const res  = await workerPost("/embed", { text: query });
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
  bm25.forEach(({ idx }, rank)     => map.set(idx, (map.get(idx) || 0) + 1 / (rank + 1) * 0.5));
  semantic.forEach(({ idx }, rank) => map.set(idx, (map.get(idx) || 0) + 1 / (rank + 1) * 0.5));
  return [...map.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, k)
    .map(([idx]) => CHUNKS[idx])
    .filter(Boolean);
}

// For LITERATURE queries: prefer paper chunks, supplement with instruments if thin
function paperBoostedContext(bm25Results, semanticResults, k = 10) {
  const paperIdxSet = new Set(CHUNKS.map((c, i) => c.type === "paper" ? i : -1).filter(i => i >= 0));
  const paperBm25     = bm25Results.filter(r => paperIdxSet.has(r.idx));
  const paperSemantic = semanticResults.filter(r => paperIdxSet.has(r.idx));
  const papers = hybridFuse(paperBm25, paperSemantic, k);
  if (papers.length >= 4) return papers;
  // Fewer than 4 paper hits — fall back to mixed results with a larger window
  return hybridFuse(bm25Results, semanticResults, k);
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
    method: "POST", headers, body: JSON.stringify(body),
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

async function streamChat(query, context, history = []) {
  const headers = { "Content-Type": "application/json" };
  if (PASSWORD) headers["Authorization"] = `Bearer ${PASSWORD}`;
  return fetch(WORKER_URL + "/chat", {
    method: "POST", headers,
    body: JSON.stringify({ query, context, history }),
  });
}

// ── Markdown renderer ─────────────────────────────────────────────────────────

function renderMarkdown(text) {
  if (!text) return "";
  // Escape HTML
  let h = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  // Fenced code blocks (must come before inline code)
  h = h.replace(/```[\w]*\n([\s\S]*?)```/g, (_, c) => `<pre><code>${c.trimEnd()}</code></pre>`);
  // Inline code
  h = h.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  // Bold / italic
  h = h.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  h = h.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
  // Headers
  h = h.replace(/^### (.+)$/gm, "<h4>$1</h4>");
  h = h.replace(/^## (.+)$/gm,  "<h3>$1</h3>");
  h = h.replace(/^# (.+)$/gm,   "<h2>$1</h2>");
  // Bullet lists
  h = h.replace(/^[ \t]*[-•*] (.+)$/gm, "<li>$1</li>");
  h = h.replace(/((?:<li>[^\n]*\n?)+)/g, "<ul>$1</ul>");
  // Numbered lists
  h = h.replace(/^\d+\. (.+)$/gm, "<nli>$1</nli>");
  h = h.replace(/((?:<nli>[^\n]*\n?)+)/g, m =>
    "<ol>" + m.replace(/<nli>/g, "<li>").replace(/<\/nli>/g, "</li>") + "</ol>");
  // Paragraphs
  h = h.replace(/\n\n+/g, "</p><p>");
  h = h.replace(/\n/g, "<br>");
  h = `<p>${h}</p>`;
  h = h.replace(/<p>\s*(<(?:pre|ul|ol|h[2-4]))/g, "$1"); // strip wrapping <p> around block elements
  h = h.replace(/(<\/(?:pre|ul|ol|h[2-4])>)\s*<\/p>/g, "$1");
  h = h.replace(/<p>\s*<\/p>/g, "");
  return h;
}

// ── Data plan card ────────────────────────────────────────────────────────────

function renderDataPlan(plan) {
  const sourceLabel = { ooi_api: "OOI API", earthscope: "EarthScope", pi_html: "PI Portal" };

  const instruments = (plan.instruments || []).map(inst => `
    <div class="plan-instrument${inst.priority === "primary" ? " plan-primary-instr" : ""}">
      <div class="plan-instr-header">
        <span class="plan-instr-name">${inst.name || inst.id}</span>
        <span class="instrument-card-badge badge-${inst.source}">${sourceLabel[inst.source] || inst.source}</span>
      </div>
      ${inst.rationale ? `<div class="plan-rationale">${inst.rationale}</div>` : ""}
    </div>`).join("");

  const tr = plan.time_range;
  const timeHtml = tr ? `
    <div class="plan-time">
      <span class="plan-time-dates">${tr.start?.slice(0, 10) ?? "?"} → ${tr.end?.slice(0, 10) ?? "?"}</span>
      ${tr.notes ? `<span class="plan-time-notes">${tr.notes}</span>` : ""}
    </div>` : "";

  return `
    <div class="data-plan-card">
      <div class="plan-card-label">Data Plan</div>
      ${plan.summary ? `<p class="plan-summary">${plan.summary}</p>` : ""}
      ${timeHtml}
      <div class="plan-instruments-list">${instruments}</div>
    </div>`;
}

// ── Related papers (after data delivery) ─────────────────────────────────────

function showRelatedPapers(context) {
  const papers = (context || []).filter(c => c.type === "paper").slice(0, 3);
  if (!papers.length) return;

  const items = papers.map(p => {
    const doi     = p.id.replace(/^paper::/, "");
    const doiLink = doi.startsWith("10.")
      ? `<a class="paper-doi-link" href="https://doi.org/${doi}" target="_blank" rel="noopener">doi.org/${doi}</a>`
      : "";
    const byline  = [p.first_author, p.year].filter(Boolean).join(" · ");
    return `
      <div class="related-paper-item">
        <div class="related-paper-title">${p.title}</div>
        <div class="related-paper-byline">${byline}${doiLink ? ` <span class="related-paper-doi">${doiLink}</span>` : ""}</div>
      </div>`;
  }).join("");

  const el = addMessage("assistant", "");
  el.innerHTML = `
    <div class="related-papers-block">
      <div class="related-papers-label">For more context on how similar questions have been addressed with RCA data:</div>
      <div class="related-papers-list">${items}</div>
    </div>`;
}

// ── Download modal ────────────────────────────────────────────────────────────

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

  if (metadata.downloadUrl) { dl.href = metadata.downloadUrl; dl.hidden = false; }
  else dl.hidden = true;

  document.getElementById("modalOverlay").hidden = false;
}

// ── URL helpers ───────────────────────────────────────────────────────────────

function getInstrumentUrl(chunk) {
  const id = chunk.id.replace(/::.*$/, "");
  if (chunk.source === "pi_html") {
    // PI-MASSP-ASHES → class "massp"; PI-OVRSRA101 → fall back to piweb
    const seg = id.split("-")[1] || "";
    const isAlpha = /^[A-Za-z]+$/.test(seg);
    return isAlpha
      ? `https://oceanobservatories.org/instrument-class/${seg.toLowerCase()}/`
      : "http://piweb.ooirsn.uw.edu/marum/";
  }
  // OOI / EarthScope reference designator: RS01SBPD-DP01A-01-CTDPFL104
  // Instrument class = first 5 alpha chars of last segment
  const segs = id.split("-");
  if (segs.length >= 4) {
    const m = segs[segs.length - 1].match(/^([A-Z]{5})/);
    if (m) return `https://oceanobservatories.org/instrument-class/${m[1].toLowerCase()}/`;
  }
  return null;
}

function getPaperUrl(chunk) {
  const doi = chunk.id.replace(/^paper::/, "");
  return doi.startsWith("10.") ? `https://doi.org/${doi}` : null;
}

// ── Instrument / paper panel ──────────────────────────────────────────────────

function renderInstruments(chunks) {
  const instrList     = document.getElementById("instrumentList");
  const paperList     = document.getElementById("paperList");
  const hint          = document.getElementById("instrumentHint");
  const papersDivider = document.getElementById("papersDivider");
  const papersTitle   = document.getElementById("papersTitle");

  instrList.innerHTML = "";
  paperList.innerHTML = "";

  const instruments = chunks.filter(c => c.type !== "paper" && c.type !== "site-context");
  const papers      = chunks.filter(c => c.type === "paper");

  if (hint) hint.style.display = instruments.length ? "none" : "";

  const seenInstr = new Set();
  instruments.forEach((c, i) => {
    if (seenInstr.has(c.id)) return;
    seenInstr.add(c.id);
    const url  = getInstrumentUrl(c);
    const card = document.createElement(url ? "a" : "div");
    card.className = `instrument-card ${i < 2 ? "primary" : "secondary"}`;
    if (url) { card.href = url; card.target = "_blank"; card.rel = "noopener noreferrer"; }
    card.innerHTML = `
      <div class="instrument-card-name">${c.title}</div>
      <div class="instrument-card-meta">${c.location || ""}</div>
      <span class="instrument-card-badge badge-${c.source}">${c.source}</span>
    `;
    instrList.appendChild(card);
  });

  if (papers.length) {
    papersDivider.hidden = false;
    papersTitle.hidden   = false;
    const seenPaper = new Set();
    papers.forEach(c => {
      if (seenPaper.has(c.id)) return;
      seenPaper.add(c.id);
      const url  = getPaperUrl(c);
      const card = document.createElement(url ? "a" : "div");
      card.className = "paper-card";
      if (url) { card.href = url; card.target = "_blank"; card.rel = "noopener noreferrer"; }
      card.innerHTML = `
        <div class="paper-card-title">${c.title}</div>
        <span class="paper-card-citation">${buildCitation(c)}</span>
      `;
      paperList.appendChild(card);
    });
  } else {
    papersDivider.hidden = true;
    papersTitle.hidden   = true;
  }
}

function buildCitation(chunk) {
  const parts = [];
  if (chunk.first_author) parts.push(chunk.first_author);
  if (chunk.journal)      parts.push(abbreviateJournal(chunk.journal));
  if (chunk.year)         parts.push(chunk.year);
  if (parts.length)       return parts.join(" · ");
  const firstLine    = (chunk.text || "").split("\n")[0];
  const yearMatch    = firstLine.match(/\((\d{4})\)/);
  const journalMatch = firstLine.match(/—\s*(.+)$/);
  const fb = [];
  if (journalMatch) fb.push(abbreviateJournal(journalMatch[1].trim()));
  if (yearMatch)    fb.push(yearMatch[1]);
  return fb.join(" · ") || "Zotero";
}

const JOURNAL_ABBREVS = {
  "journal of geophysical research": "JGR",
  "journal of geophysical research: solid earth": "JGR Solid Earth",
  "journal of geophysical research: oceans": "JGR Oceans",
  "geophysical research letters": "GRL",
  "earth and planetary science letters": "EPSL",
  "annual review of earth and planetary sciences": "Annu. Rev. Earth Planet. Sci.",
  "perspectives of earth and space scientists": "Persp. Earth Space Sci.",
  "jasa express letters": "JASA Express Lett.",
  "the journal of the acoustical society of america": "J. Acoust. Soc. Am.",
  "journal of the acoustical society of america": "J. Acoust. Soc. Am.",
  "frontiers in marine science": "Front. Mar. Sci.",
  "visual intelligence": "Vis. Intell.",
  "nature communications": "Nat. Commun.",
  "nature geoscience": "Nat. Geosci.",
  "science": "Science",
  "nature": "Nature",
  "seismological research letters": "Seismol. Res. Lett.",
  "bulletin of the seismological society of america": "Bull. Seismol. Soc. Am.",
  "geochemistry, geophysics, geosystems": "G-Cubed",
  "oceanography": "Oceanography",
  "deep-sea research": "Deep-Sea Res.",
  "deep-sea research part i": "Deep-Sea Res. I",
  "deep-sea research part ii": "Deep-Sea Res. II",
  "journal of marine systems": "J. Mar. Syst.",
  "earth science reviews": "Earth-Sci. Rev.",
};

function abbreviateJournal(journal) {
  return JOURNAL_ABBREVS[journal.toLowerCase().trim()] || journal;
}

// ── Status bar ────────────────────────────────────────────────────────────────

function setStatus(text) {
  const bar = document.getElementById("statusBar");
  document.getElementById("statusText").textContent = text;
  bar.hidden = !text;
}

// ── UI: add message bubble ────────────────────────────────────────────────────

function addMessage(role, content, { html = false } = {}) {
  const list  = document.getElementById("messageList");
  const div   = document.createElement("div");
  div.className = `message ${role}`;
  const label = role === "user" ? "You" : role === "assistant" ? "aRCADA" : "Error";
  div.innerHTML = `<div class="message-label">${label}</div><div class="message-content"></div>`;
  const contentEl = div.querySelector(".message-content");
  if (html)         contentEl.innerHTML  = content;
  else if (content) contentEl.textContent = content;
  list.appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return contentEl;
}

// ── Intent classification (client-side heuristic) ─────────────────────────────

function classifyIntent(query) {
  const q = query.trim().toLowerCase();

  // Strong data-request signals
  const dataPatterns = [
    /^(get|fetch|pull|download|retrieve|give me|show me data|i need data)\b/,
    /\bdata\s+(from|for|at|between|since|after|before)\b/,
    /\b(time series|records|measurements|observations|readings)\s+(from|at|for|between)\b/,
    /\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\b/,
    /\b\d{4}[-/]\d{2}[-/]\d{2}\b/,
    /\b(last|past)\s+\d+\s+(day|week|month|year)s?\b/,
  ];
  if (dataPatterns.some(p => p.test(q))) return "DATA_REQUEST";

  // Literature / paper questions — before general QUESTION check so they get paper-boosted retrieval
  const literaturePatterns = [
    /\b(paper|papers|publication|publications|article|articles|study|studies|literature|journal)\b/,
    /what (has been|have been|was|were) (published|written|found|studied)/,
    /what (research|work|studies) (exist|are there|has been done)/,
    /\b(cite|citation|reference|bibliography|findings|results)\b/,
  ];
  if (literaturePatterns.some(p => p.test(q))) return "LITERATURE";

  // Capability questions
  const capabilityPatterns = /^(what can you|what do you|what are you|what is arcada|tell me what you|how do you work|what are your capabilities)/;
  if (capabilityPatterns.test(q)) return "CAPABILITY";

  // Strong question signals
  if (q.endsWith("?")) return "QUESTION";
  const questionStarters = /^(what|why|how|when|where|who|which|is |are |can you|could you|explain|tell me about|what's|what is|does |do )/;
  if (questionStarters.test(q)) return "QUESTION";

  return "AMBIGUOUS";
}

function responseHasQuestion(text) {
  // Check if the AI's response ends with a question (last 300 chars)
  return text.slice(-300).trim().includes("?");
}

// ── Streaming helper ──────────────────────────────────────────────────────────

async function streamChatToElement(query, context, contentEl, history = []) {
  const resp = await streamChat(query, context, history);
  if (!resp.ok) throw new Error(`Chat stream failed: ${resp.status}`);

  const reader  = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer   = "";
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
        const text  = chunk?.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
        fullText   += text;
        contentEl.innerHTML = renderMarkdown(fullText);
      } catch { /* partial JSON — skip */ }
    }
  }

  contentEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return fullText;
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

  try {
    if (CONV_STATE === "awaiting_clarification") {
      await handleClarificationReply(query);
    } else {
      await handleNewQuery(query);
    }
  } catch (err) {
    setStatus("");
    addMessage("error", `Error: ${err.message}`);
    console.error(err);
  } finally {
    submitBtn.disabled = false;
  }
}

async function handleNewQuery(query) {
  const intent = classifyIntent(query);

  let context = [];

  if (intent !== "CAPABILITY") {
    // Capability questions need no retrieval — system prompt has all the info
    setStatus("Searching catalog…");
    const [bm25, semantic] = await Promise.all([
      bm25Search(query, 12),
      semanticSearch(query, 12),
    ]);
    context = (intent === "LITERATURE")
      ? paperBoostedContext(bm25, semantic)
      : hybridFuse(bm25, semantic);
    renderInstruments(context);
  }

  // Stream conversational response
  setStatus("Generating response…");
  const contentEl = addMessage("assistant", "");
  const historySnapshot = [...HISTORY];
  const fullText = await streamChatToElement(query, context, contentEl, historySnapshot);
  setStatus("");

  // Fallback if model returned empty (safety filter, quota, etc.)
  if (!fullText.trim()) {
    contentEl.textContent = "I'm aRCADA, your Regional Cabled Array data assistant. I can help you access seismic, pressure, hydrophone, CTD, and geochemical data from the Cascadia margin — or answer questions about the published research. What would you like to explore?";
  }

  // Update history (cap at 8 turns = 16 entries)
  HISTORY.push({ role: "user",  parts: [{ text: query    }] });
  HISTORY.push({ role: "model", parts: [{ text: fullText }] });
  if (HISTORY.length > 16) HISTORY = HISTORY.slice(-16);

  // Non-data intents → answered, done
  if (intent === "QUESTION" || intent === "LITERATURE" || intent === "CAPABILITY") return;

  // AI asked a clarifying question → wait for user response
  if (responseHasQuestion(fullText)) {
    CONV_STATE      = "awaiting_clarification";
    PENDING_QUERY   = query;
    PENDING_CONTEXT = context;
    return;
  }

  // Data request with no clarification needed → pull data
  await proceedDataPull(query, context);
}

async function handleClarificationReply(reply) {
  setStatus("Generating response…");
  const contentEl = addMessage("assistant", "");
  const historySnapshot = [...HISTORY];
  const fullText = await streamChatToElement(reply, PENDING_CONTEXT || [], contentEl, historySnapshot);
  setStatus("");

  HISTORY.push({ role: "user",  parts: [{ text: reply    }] });
  HISTORY.push({ role: "model", parts: [{ text: fullText }] });
  if (HISTORY.length > 16) HISTORY = HISTORY.slice(-16);

  // AI still asking questions → stay in clarification mode
  if (responseHasQuestion(fullText)) return;

  // Clarification complete — merge and pull
  CONV_STATE = "idle";
  const combinedQuery = `${PENDING_QUERY}. User clarified: ${reply}`;
  const context       = PENDING_CONTEXT;
  PENDING_QUERY   = null;
  PENDING_CONTEXT = null;
  await proceedDataPull(combinedQuery, context);
}

async function proceedDataPull(query, context) {
  setStatus("Building data plan…");

  // Deduplicate context to base IDs before sending to /plan
  const seenPlan = new Set();
  const planContext = context.filter(c => {
    const base = c.id.replace(/::.*$/, "");
    if (seenPlan.has(base)) return false;
    seenPlan.add(base);
    return true;
  });

  const { plan } = await workerPost("/plan", { query, context: planContext, catalog: INSTRUMENT_CATALOG });

  if (!plan?.instruments?.length) {
    setStatus("");
    addMessage("assistant", "I couldn't identify specific instruments to pull for that request. Try rephrasing, or check the example queries below.");
    return;
  }

  // Render plan as formatted card
  const planEl = addMessage("assistant", "");
  planEl.innerHTML = renderDataPlan(plan);

  // Dispatch GitHub Actions job
  setStatus("Dispatching data pull job…");
  const { runId } = await workerPost("/dispatch", { plan });

  if (runId) {
    setStatus(`Job running (ID ${runId}) — this may take several minutes…`);
    await pollJob(runId, plan, context);
  } else {
    setStatus("");
    addMessage("error", "Could not start data pull job. Check Worker configuration.");
  }
}

async function pollJob(runId, plan, context = []) {
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
          showRelatedPapers(context);
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

// ── Theme ─────────────────────────────────────────────────────────────────────

function initTheme() {
  const saved = localStorage.getItem('arcada-theme');
  if (saved === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
  updateThemeIcon();
  document.getElementById('themeToggle')?.addEventListener('click', toggleTheme);
}

function toggleTheme() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  if (isDark) {
    document.documentElement.removeAttribute('data-theme');
    localStorage.setItem('arcada-theme', 'light');
  } else {
    document.documentElement.setAttribute('data-theme', 'dark');
    localStorage.setItem('arcada-theme', 'dark');
  }
  updateThemeIcon();
}

function updateThemeIcon() {
  const btn = document.getElementById('themeToggle');
  if (!btn) return;
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  btn.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
  btn.querySelector('.icon-sun').hidden = !isDark;
  btn.querySelector('.icon-moon').hidden = isDark;
}

boot().catch(console.error);
