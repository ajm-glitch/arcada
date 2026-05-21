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

// ── Mode ──────────────────────────────────────────────────────────────────────
let CURRENT_MODE = "ask"; // "ask" | "literature" | "data"

// ── Rate limiter ──────────────────────────────────────────────────────────────
// gemini-2.0-flash:      15 RPM free tier → 4s min between calls
// gemini-2.0-flash-lite: 30 RPM free tier → 2s min between calls
const RL = {
  flash:     { ms: 4200, last: 0 },
  flashLite: { ms: 2200, last: 0 },
};

async function throttle(model) {
  const r = RL[model];
  const wait = r.ms - (Date.now() - r.last);
  if (wait > 0) {
    setStatus(`Rate limiting — waiting ${(wait / 1000).toFixed(1)}s…`);
    await new Promise(res => setTimeout(res, wait));
    setStatus("");
  }
  r.last = Date.now();
}

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
  document.querySelectorAll(".mode-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      CURRENT_MODE = btn.dataset.mode;
      document.querySelectorAll(".mode-btn").forEach(b => b.classList.toggle("active", b === btn));
      const placeholders = {
        ask:        "e.g. What instruments are at Axial Seamount? What does BOTPT measure?",
        literature: "e.g. What papers have been published on Axial seismicity? Summarize methane seep research.",
        data:       "e.g. Show me seismic and pressure data near Axial Seamount for two weeks after the 2015 eruption",
      };
      document.getElementById("queryInput").placeholder = placeholders[CURRENT_MODE] || "";
    });
  });
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
    await throttle("flashLite");
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
  await throttle("flash");
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
  const sourceLabel = { ooi_api: "OOI M2M API", earthscope: "EarthScope FDSN", pi_html: "PI Portal" };
  const tr = plan.time_range;
  const period = tr ? `${tr.start?.slice(0, 10) ?? "?"} → ${tr.end?.slice(0, 10) ?? "?"}` : "—";
  const sizeEst = estimateDataSize(plan);

  const instrRows = (plan.instruments || []).map(inst =>
    `<div class="script-instr-row">
      <span class="script-instr-name">${inst.name || inst.id}</span>
      <span class="script-instr-source">${sourceLabel[inst.source] || inst.source}</span>
    </div>`
  ).join("");

  const notesHtml = tr?.notes
    ? `<div style="font-size:0.78rem;color:var(--text-muted);margin-bottom:0.5rem;font-style:italic;">${tr.notes}</div>`
    : "";

  return `
    <div class="script-card">
      <div class="script-card-label">Data Pull Script</div>
      ${plan.summary ? `<p class="script-summary">${plan.summary}</p>` : ""}
      ${notesHtml}
      <div class="script-instr-list">${instrRows}</div>
      <div class="script-meta">
        <span><strong>Period</strong> ${period}</span>
        ${sizeEst ? `<span><strong>Est. size</strong> ${sizeEst}</span>` : ""}
      </div>
      <button class="btn-download-script">
        <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true"><path d="M6.5 1v8M3 7l3.5 3.5L10 7M1 11.5h11" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
        Download Script (.py)
      </button>
    </div>`;
}

// ── Data size estimator ───────────────────────────────────────────────────────
function estimateDataSize(plan) {
  const tr = plan.time_range;
  if (!tr?.start || !tr?.end) return null;
  const duration_s = (new Date(tr.end) - new Date(tr.start)) / 1000;
  if (duration_s <= 0) return null;

  // bytes/sample * channels * format_overhead
  const SIZE_TABLE = {
    seismometer:      { hz: 100,  ch: 3,  bps: 4,  fmt: 1.0 },
    hydrophone:       { hz: 200,  ch: 1,  bps: 4,  fmt: 1.0 },
    pressure:         { hz: 1,    ch: 3,  bps: 8,  fmt: 2.0 },
    ctd:              { hz: 1,    ch: 5,  bps: 8,  fmt: 2.0 },
    pco2:             { hz: 0.5,  ch: 3,  bps: 8,  fmt: 2.0 },
    adcp:             { hz: 1,    ch: 8,  bps: 8,  fmt: 2.0 },
    fluorometer:      { hz: 1,    ch: 3,  bps: 8,  fmt: 2.0 },
    thermistor:       { hz: 1,    ch: 4,  bps: 8,  fmt: 2.0 },
    thermistor_array: { hz: 1,    ch: 24, bps: 8,  fmt: 2.0 },
    dissolved_oxygen: { hz: 1,    ch: 2,  bps: 8,  fmt: 2.0 },
    nitrate:          { hz: 0.1,  ch: 2,  bps: 8,  fmt: 2.0 },
    ph:               { hz: 0.1,  ch: 2,  bps: 8,  fmt: 2.0 },
    velocimeter:      { hz: 1,    ch: 3,  bps: 8,  fmt: 2.0 },
    hpies:            { hz: 1,    ch: 3,  bps: 8,  fmt: 2.0 },
  };

  let total = 0;
  for (const inst of (plan.instruments || [])) {
    const cat = INSTRUMENT_CATALOG.find(c => c.id === inst.id);
    const t   = SIZE_TABLE[inst.type] || { hz: 1, ch: 2, bps: 8, fmt: 2.0 };
    const hz  = cat?.sample_rate_hz || t.hz;
    total += hz * t.ch * t.bps * t.fmt * duration_s;
  }

  if (total < 1024)            return `~${Math.round(total)} B`;
  if (total < 1024 ** 2)       return `~${(total / 1024).toFixed(0)} KB`;
  if (total < 1024 ** 3)       return `~${(total / 1024 ** 2).toFixed(0)} MB`;
  return `~${(total / 1024 ** 3).toFixed(1)} GB`;
}

// ── Python script generator ───────────────────────────────────────────────────

const STREAM_OVERRIDES = {
  PRESTA: ["prest_real_time",          "streamed"],
  PRESTB: ["prest_real_time",          "streamed"],
  BOTPTA: ["botpt_nano_sample",        "streamed"],
  FLORDD: ["flord_d_data_record",      "streamed"],
  FLORTD: ["flort_d_data_record",      "streamed"],
  FLCDRA: ["flcd_r_dcl_instrument",    "recovered_inst"],
  FLNTUA: ["flntu_a_dcl_instrument",   "recovered_inst"],
  CTDPFL: ["ctdpf_optode_sample",      "recovered_inst"],
  VEL3DA: ["vel3d_b_sample",           "recovered_inst"],
  DOSTAD: ["do_stable_sample",         "streamed"],
  VADCPA: ["adcp_velocity_beam",       "streamed"],
  VADCPB: ["adcp_velocity_beam",       "streamed"],
  ADCPTD: ["adcp_velocity_beam",       "streamed"],
  ADCPTE: ["adcp_velocity_beam",       "streamed"],
  ADCPSK: ["adcp_velocity_beam",       "streamed"],
  VEL3DB: ["vel3d_b_sample",           "streamed"],
  VELPTD: ["velpt_velocity_data",      "streamed"],
  THSPHA: ["thsph_a_dcl_instrument",   "streamed"],
  TRHPHA: ["trhph_sample",             "streamed"],
  TMPSFA: ["tmpsf_sample",             "streamed"],
  HYDLFA: ["hydlf_a_dcl_instrument",   "streamed"],
  HYDBBA: ["hydbba_dcl_data",          "streamed"],
};

const STREAM_BY_TYPE = {
  pressure:         ["botpt_nano_sample",         "streamed"],
  ctd:              ["ctdpf_optode_sample",        "streamed"],
  dissolved_oxygen: ["do_stable_sample",           "streamed"],
  ph:               ["phsen_data_record",          "streamed"],
  fluorometer:      ["flort_d_data_record",        "streamed"],
  nitrate:          ["nutnr_a_sample",             "streamed"],
  adcp:             ["adcp_velocity_beam",         "streamed"],
  velocimeter:      ["vel3d_b_sample",             "streamed"],
  pco2:             ["pco2w_a_sami_data_record",   "streamed"],
  hpies:            ["horizontal_electric_field",  "streamed"],
  thermistor_array: ["tmpsf_sample",               "streamed"],
  thermistor:       ["trhph_sample",               "streamed"],
};

const PI_BASE_URLS = {
  "PI-OVRSRA101":    "http://piweb.ooirsn.uw.edu/marum/data/OVRSRA101/",
  "PI-QNTSRA101":    "http://piweb.ooirsn.uw.edu/marum/data/QNTSRA101/",
  "PI-MASSP-ASHES":  "http://piweb.ooirsn.uw.edu/marum/data/MASSP/",
  "PI-RASSP":        "http://piweb.ooirsn.uw.edu/marum/data/RASSP/",
  "PI-CTDPFA110":    "http://piweb.ooirsn.uw.edu/marum/data/CTDPFA110/",
  "PI-SCPRAA301":    "http://piweb.ooirsn.uw.edu/scpr/data/",
  "PI-A0ABPA301":    "http://piweb.ooirsn.uw.edu/a0a/data/A0ABPA301_data/",
  "PI-COVIS":        "http://piweb.ooirsn.uw.edu/covis/data/COVIS/",
  "PI-DAS-OPTASENSE":"http://piweb.ooirsn.uw.edu/das/data/Optasense/",
  "PI-DAS24":        "http://piweb.ooirsn.uw.edu/das24/data/",
  "PI-DAS25":        "http://piweb.ooirsn.uw.edu/das25/data/",
};

function resolveOOIStream(instId, instType, catEntry) {
  if (catEntry?.stream) {
    const method = (catEntry.node || "").startsWith("DP") ? "recovered_inst" : "streamed";
    return [catEntry.stream, method];
  }
  const parts   = instId.split("-");
  const lastSeg = parts[parts.length - 1] || "";
  const cls     = (lastSeg.match(/^([A-Z][A-Z0-9]{4,5})/) || [])[1] || "";
  if (STREAM_OVERRIDES[cls]) return STREAM_OVERRIDES[cls];
  if (STREAM_BY_TYPE[instType]) return STREAM_BY_TYPE[instType];
  return ["unknown_stream", "streamed"];
}

function pyDate(iso) {
  const d = new Date(iso);
  return [d.getUTCFullYear(), d.getUTCMonth() + 1, d.getUTCDate(),
          d.getUTCHours(), d.getUTCMinutes(), d.getUTCSeconds()].join(", ");
}

function generateDataScript(plan) {
  const { summary, time_range, instruments } = plan;
  const start = time_range?.start || "1970-01-01T00:00:00Z";
  const end   = time_range?.end   || "1970-01-02T00:00:00Z";

  const needsObspy    = instruments.some(i => i.source === "earthscope");
  const needsRequests = instruments.some(i => i.source === "ooi_api" || i.source === "pi_html");

  const fns   = [];
  const calls = [];

  for (const inst of instruments) {
    const cat     = INSTRUMENT_CATALOG.find(c => c.id === inst.id) || {};
    const safeId  = inst.id.replace(/[^a-zA-Z0-9]/g, "_").toLowerCase();
    const fnName  = "fetch_" + safeId;

    if (inst.source === "ooi_api") {
      const parts  = inst.id.split("-");
      const site   = parts[0] || "";
      const node   = parts[1] || "";
      const instr  = parts.slice(2).join("-") || "";
      const [stream, method] = resolveOOIStream(inst.id, inst.type, cat);

      fns.push([
        `def ${fnName}():`,
        `    """${inst.name} (${inst.type}) — OOI M2M API"""`,
        `    begin = START.strftime('%Y-%m-%dT%H:%M:%S.000Z')`,
        `    end_dt = END.strftime('%Y-%m-%dT%H:%M:%S.000Z')`,
        `    url = (`,
        `        "https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv"`,
        `        f"/${site}/${node}/${instr}/${method}/${stream}"`,
        `        f"?beginDT={begin}&endDT={end_dt}&format=application/json&limit=20000"`,
        `    )`,
        `    auth = (OOI_USERNAME, OOI_TOKEN) if OOI_USERNAME else None`,
        `    r = requests.get(url, auth=auth, timeout=120)`,
        `    r.raise_for_status()`,
        `    data = r.json()`,
        `    out = OUT_DIR / "${safeId}.json"`,
        `    with open(out, "w") as f:`,
        `        json.dump(data, f)`,
        `    print(f"  ${inst.name}: {len(data)} records → {out.name}")`,
        ``,
      ].join("\n"));

    } else if (inst.source === "earthscope") {
      const esId    = inst.id.replace(/^EARTHSCOPE-/, "");
      const esParts = esId.split("-");
      const network = esParts[0] || "OO";
      const station = esParts[1] || esId;
      const channel = inst.type === "hydrophone" ? "HDH,LDH" : "BH*,HH*";

      fns.push([
        `def ${fnName}():`,
        `    """${inst.name} (${inst.type}) — EarthScope FDSN"""`,
        `    client = FDSNClient("IRIS")`,
        `    st = client.get_waveforms(`,
        `        network="${network}", station="${station}",`,
        `        location="*", channel="${channel}",`,
        `        starttime=UTCDateTime(START), endtime=UTCDateTime(END),`,
        `    )`,
        `    st.merge(method=1, fill_value=0)`,
        `    out = OUT_DIR / "${safeId}.mseed"`,
        `    st.write(str(out), format="MSEED")`,
        `    print(f"  ${inst.name}: {len(st)} traces → {out.name}")`,
        ``,
      ].join("\n"));

    } else if (inst.source === "pi_html") {
      const baseUrl = PI_BASE_URLS[inst.id] || "#";
      fns.push([
        `def ${fnName}():`,
        `    """${inst.name} — PI Portal (manual download)`,
        `    Data available at: ${baseUrl}`,
        `    Navigate to the relevant date folder to download files."""`,
        `    print("  ${inst.name}: PI portal — download manually from:")`,
        `    print("    ${baseUrl}")`,
        ``,
      ].join("\n"));
    }

    calls.push(`    ${fnName}()`);
  }

  const instrList = instruments.map(i => `  - ${i.name} (${i.source})`).join("\n");
  const obspyLine = needsObspy    ? "\nfrom obspy.clients.fdsn import Client as FDSNClient\nfrom obspy import UTCDateTime" : "";
  const reqLine   = needsRequests ? "\nimport requests" : "";

  return [
    `"""`,
    `aRCADA Data Pull Script`,
    `Generated: ${new Date().toISOString()}`,
    `Summary:   ${summary || "Data request"}`,
    ``,
    `Time range: ${start.slice(0, 10)} → ${end.slice(0, 10)}`,
    `Instruments:`,
    instrList,
    ``,
    `Requirements:`,
    `  pip install${needsRequests ? " requests" : ""}${needsObspy ? " obspy" : ""}`,
    ``,
    `OOI credentials (if using OOI M2M):`,
    `  export OOI_USERNAME=your_username`,
    `  export OOI_TOKEN=your_api_token`,
    `  Register: https://ooinet.oceanobservatories.org`,
    `"""`,
    ``,
    `import json, os`,
    `from datetime import datetime, timezone`,
    `from pathlib import Path`,
    reqLine,
    obspyLine,
    ``,
    `START        = datetime(${pyDate(start)}, tzinfo=timezone.utc)`,
    `END          = datetime(${pyDate(end)}, tzinfo=timezone.utc)`,
    `OUT_DIR      = Path("./arcada_data")`,
    `OUT_DIR.mkdir(exist_ok=True)`,
    `OOI_USERNAME = os.getenv("OOI_USERNAME", "")`,
    `OOI_TOKEN    = os.getenv("OOI_TOKEN", "")`,
    ``,
    ...fns,
    `if __name__ == "__main__":`,
    `    print(f"Fetching: {START.date()} → {END.date()}")`,
    ...calls,
    `    print(f"\\nDone. Saved to: {OUT_DIR.resolve()}")`,
    ``,
  ].join("\n");
}

function downloadFile(content, filename, mime) {
  const blob = new Blob([content], { type: mime });
  const url  = URL.createObjectURL(blob);
  const a    = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
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

// PI instrument data pages — exact URLs from catalog/pi-pages.json
const PI_PAGES = {
  "PI-OVRSRA101":    "http://piweb.ooirsn.uw.edu/marum/data/OVRSRA101/",
  "PI-QNTSRA101":    "http://piweb.ooirsn.uw.edu/marum/data/QNTSRA101/",
  "PI-MASSP-ASHES":  "http://piweb.ooirsn.uw.edu/marum/data/MASSP/",
  "PI-RASSP":        "http://piweb.ooirsn.uw.edu/marum/data/RASSP/",
  "PI-CTDPFA110":    "http://piweb.ooirsn.uw.edu/marum/data/CTDPFA110/",
  "PI-SCPRAA301":    "http://piweb.ooirsn.uw.edu/scpr/data/",
  "PI-A0ABPA301":    "http://piweb.ooirsn.uw.edu/a0a/data/A0ABPA301_data/",
  "PI-COVIS":        "http://piweb.ooirsn.uw.edu/covis/data/COVIS/",
  "PI-DAS-OPTASENSE":"http://piweb.ooirsn.uw.edu/das/data/Optasense/",
  "PI-DAS24":        "http://piweb.ooirsn.uw.edu/das24/data/",
  "PI-DAS25":        "http://piweb.ooirsn.uw.edu/das25/data/",
};

// OOI instrument class page by instrument type.
// Only map types where the generic class slug matches all variants.
// Types with diverse class codes (adcp, fluorometer, hpies, thermistor_array)
// are intentionally omitted — the ref-designator fallback extracts the exact class.
const OOI_CLASS_BY_TYPE = {
  dissolved_oxygen: "dosta",
  pco2:             "pco2w",
  thermistor:       "thsph",
  nitrate:          "nutnr",
  ph:               "phsen",
  pressure:         "botpt",
};

function getInstrumentUrl(chunk) {
  const id = chunk.id.replace(/::.*$/, "");

  // PI instruments: exact lookup
  if (chunk.source === "pi_html") return PI_PAGES[id] || null;

  // EarthScope instruments: FDSN OO network
  if (chunk.source === "earthscope") return "https://www.fdsn.org/networks/detail/OO/";

  // OOI API: type-based mapping first
  const typeSlug = OOI_CLASS_BY_TYPE[chunk.type];
  if (typeSlug) return `https://oceanobservatories.org/instrument-class/${typeSlug}/`;

  // Fallback: extract class code from ref designator (5–6 alphanumeric chars, e.g. VEL3DA, CTDPFL)
  const segs = id.split("-");
  if (segs.length >= 4) {
    const m = segs[segs.length - 1].match(/^([A-Z][A-Z0-9]{4,5})/);
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
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.error || `Chat stream failed: ${resp.status}`);
  }

  contentEl.innerHTML = '<span class="thinking-indicator">Thinking<span class="thinking-dots"></span></span>';

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
        const parts = chunk?.candidates?.[0]?.content?.parts ?? [];
        const text  = parts.filter(p => !p.thought).map(p => p.text ?? "").join("");
        fullText   += text;
        if (fullText) contentEl.innerHTML = renderMarkdown(fullText);
      } catch { /* partial JSON — skip */ }
    }
  }

  if (!fullText) contentEl.innerHTML = '<span class="error-text">No response received. Please try again.</span>';
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

const CAPABILITY_RESPONSE = `**aRCADA** is your data assistant for the OOI Regional Cabled Array and EarthScope seafloor networks on the Cascadia margin. Here's what I can do:

**Answer questions** about instruments, sites, and research — ask me anything about Axial Seamount, Hydrate Ridge, the ASHES vent field, or the Endurance Array.

**Summarize published research** — ask "What papers have been published on Axial seismicity?" or "What research exists on methane seeps?" and I'll draw from a curated literature library.

**Pull time-series data** — describe what you need in plain language (e.g. "hydrophone and pCO2 data from Hydrate Ridge, January–March 2021") and I'll identify the right instruments, build a data plan, and trigger a fetch job. Data is returned as Zarr archives.

**Instruments I can access:**
- Ocean bottom seismometers (OBS) at Axial Seamount and Hydrate Ridge
- Seafloor pressure sensors (BOTPT) for deformation and slow-slip
- CTD profilers (temperature, salinity, pressure)
- Hydrophones (acoustic signals, whale calls, bubble plumes)
- pCO2 sensors near methane seep fields
- PI-operated sonar and mass spectrometers via the UW piweb portal

What would you like to explore?`;

async function handleNewQuery(query) {
  let intent = classifyIntent(query);

  // Mode overrides intent
  if (CURRENT_MODE === "literature") intent = "LITERATURE";
  else if (CURRENT_MODE === "data")  intent = "DATA_REQUEST";

  // Capability questions: skip the API call entirely, render static response
  if (intent === "CAPABILITY") {
    const contentEl = addMessage("assistant", "");
    contentEl.innerHTML = renderMarkdown(CAPABILITY_RESPONSE);
    HISTORY.push({ role: "user",  parts: [{ text: query }] });
    HISTORY.push({ role: "model", parts: [{ text: CAPABILITY_RESPONSE }] });
    if (HISTORY.length > 16) HISTORY = HISTORY.slice(-16);
    return;
  }

  // All other intents: run retrieval
  setStatus("Searching catalog…");
  const [bm25, semantic] = await Promise.all([
    bm25Search(query, 12),
    semanticSearch(query, 12),
  ]);
  const context = (intent === "LITERATURE")
    ? paperBoostedContext(bm25, semantic)
    : hybridFuse(bm25, semantic);
  renderInstruments(context);

  // DATA_REQUEST: get AI acknowledgment (cheap /ack call) then build plan
  if (intent === "DATA_REQUEST") {
    const seen = new Set();
    const instrMatches = context
      .filter(c => !c.id.startsWith("paper::"))
      .filter(c => { const base = c.id.replace(/::.*$/, ""); return !seen.has(base) && seen.add(base); })
      .slice(0, 6);

    if (!instrMatches.length) {
      addMessage("assistant", "No matching instruments found for this request. Try rephrasing or check the example queries.");
      setStatus("");
      return;
    }

    // Fire /ack immediately; render placeholder while it loads
    const affirmEl = addMessage("assistant", "");
    affirmEl.innerHTML = '<span class="thinking-indicator">Thinking<span class="thinking-dots"></span></span>';

    setStatus("Building data plan…");

    // Get AI acknowledgment — fast flash-lite call
    await throttle("flashLite");
    const ack = await workerPost("/ack", {
      query,
      instruments: instrMatches.map(c => ({ name: c.name || c.title, location: c.location, type: c.type })),
    }).then(r => r.ack || "").catch(() => "");
    affirmEl.innerHTML = renderMarkdown(ack || instrMatches.map(c => `- ${c.name || c.title}`).join("\n"));

    const affirmText = ack || `Found instruments: ${instrMatches.map(c => c.name || c.title).join(", ")}.`;
    HISTORY.push({ role: "user",  parts: [{ text: query      }] });
    HISTORY.push({ role: "model", parts: [{ text: affirmText }] });
    if (HISTORY.length > 16) HISTORY = HISTORY.slice(-16);

    await proceedDataPull(query, context);
    return;
  }

  // All other intents: stream conversational response
  setStatus("Generating response…");
  const contentEl = addMessage("assistant", "");
  const historySnapshot = [...HISTORY];
  const fullText = await streamChatToElement(query, context, contentEl, historySnapshot);
  setStatus("");

  // Update history (cap at 8 turns = 16 entries)
  HISTORY.push({ role: "user",  parts: [{ text: query    }] });
  HISTORY.push({ role: "model", parts: [{ text: fullText }] });
  if (HISTORY.length > 16) HISTORY = HISTORY.slice(-16);

  // Non-data intents → answered, done
  if (intent === "QUESTION" || intent === "LITERATURE") return;

  // AMBIGUOUS: AI asked a clarifying question → wait for user response
  if (responseHasQuestion(fullText)) {
    CONV_STATE      = "awaiting_clarification";
    PENDING_QUERY   = query;
    PENDING_CONTEXT = context;
    return;
  }

  // AMBIGUOUS with no clarifying question → pull data
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

  const seenPlan = new Set();
  const planContext = context.filter(c => {
    const base = c.id.replace(/::.*$/, "");
    if (seenPlan.has(base)) return false;
    seenPlan.add(base);
    return true;
  });

  await throttle("flashLite");
  const { plan, debug } = await workerPost("/plan", { query, context: planContext });

  if (!plan?.instruments?.length) {
    setStatus("");
    addMessage("assistant", "I couldn't identify specific instruments for that request. Try rephrasing, or check the example queries.");
    return;
  }

  setStatus("");
  const planEl = addMessage("assistant", "");
  planEl.innerHTML = renderDataPlan(plan);

  // Attach download handler
  const btn = planEl.querySelector(".btn-download-script");
  if (btn) {
    btn.addEventListener("click", () => {
      const script = generateDataScript(plan);
      downloadFile(script, "arcada_data_pull.py", "text/x-python");
    });
  }

  showRelatedPapers(context);
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
