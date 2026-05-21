/**
 * aRCADA RAG index builder.
 * Reads catalog/instruments.json → chunks text → embeds via Gemini →
 * writes public/embeddings.bin, public/chunks.json, public/search-index.json
 *
 * Adapted from vaultnotes/scripts/index-notes.mjs
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dir = path.dirname(fileURLToPath(import.meta.url));
const ROOT  = path.resolve(__dir, "..");

const CONFIG       = JSON.parse(fs.readFileSync(path.join(ROOT, "rag-config.json"), "utf8"));
const CATALOG      = JSON.parse(fs.readFileSync(path.join(ROOT, "catalog", "instruments.json"), "utf8"));
const PI_PAGES     = JSON.parse(fs.readFileSync(path.join(ROOT, "catalog", "pi-pages.json"), "utf8"));
const PAPERS       = loadOptional(path.join(ROOT, "catalog", "papers.json"));
const M2M_META     = loadOptional(path.join(ROOT, "catalog", "m2m-metadata.json"));
const RCA_CONTEXT  = loadOptional(path.join(ROOT, "catalog", "rca-context.json"));
const GEMINI_KEY   = process.env.GEMINI_API_KEY;

function loadOptional(p) {
  try { return JSON.parse(fs.readFileSync(p, "utf8")); }
  catch { console.warn(`  (optional) ${path.basename(p)} not found — skipping`); return null; }
}
const EMBED_URL    = `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents?key=${GEMINI_KEY}`;
const OUT_DIR      = path.join(ROOT, "public");

if (!GEMINI_KEY) { console.error("GEMINI_API_KEY not set"); process.exit(1); }
fs.mkdirSync(OUT_DIR, { recursive: true });

// ── Chunk instruments ─────────────────────────────────────────────────────────

function instrumentToText(inst) {
  const lines = [
    `Name: ${inst.name}`,
    `Type: ${inst.type}`,
    `Location: ${inst.location}`,
    `Source: ${inst.source}`,
    `Description: ${inst.description}`,
    `Keywords: ${(inst.keywords || []).join(", ")}`,
    inst.latitude  != null ? `Latitude: ${inst.latitude}` : null,
    inst.longitude != null ? `Longitude: ${inst.longitude}` : null,
    inst.depth_m   != null ? `Depth: ${inst.depth_m} m` : null,
    inst.start_date ? `Data available from: ${inst.start_date}` : null,
    inst.units      ? `Units: ${JSON.stringify(inst.units)}` : null,
    inst.sample_rate_hz ? `Sample rate: ${inst.sample_rate_hz} Hz` : null,
    inst.channels   ? `Channels: ${inst.channels.join(", ")}` : null,
    inst.site       ? `OOI site: ${inst.site} / node: ${inst.node}` : null,
    inst.network    ? `EarthScope network: ${inst.network} / station: ${inst.station}` : null,
    inst.pi_base_url ? `PI data URL: ${inst.pi_base_url}` : null,
  ].filter(Boolean);
  return lines.join("\n");
}

function makeChunks() {
  const chunks = [];

  // ── Instrument chunks ──────────────────────────────────────────────────────
  for (const inst of [...CATALOG.instruments, ...PI_PAGES.instruments]) {
    const m2m    = M2M_META?.instruments?.[inst.id];
    const params = m2m?.parameters ?? [];

    // Augment instrument text with M2M parameter names
    const paramLine = params.length
      ? `Parameters measured: ${params.map(p => p.display_name || p.name).join(", ")}`
      : null;

    const text = [instrumentToText(inst), paramLine].filter(Boolean).join("\n");
    const embedText = `Instrument: ${inst.name}\n\n${text}`;

    chunks.push({
      id:       inst.id,
      title:    inst.name,
      type:     inst.type,
      source:   inst.source,
      location: inst.location || null,
      keywords: inst.keywords || [],
      text,
      embedText,
    });

    // Description chunk
    if (inst.description && inst.description.length > 100) {
      chunks.push({
        id:       `${inst.id}::desc`,
        title:    `${inst.name} (description)`,
        type:     inst.type,
        source:   inst.source,
        location: inst.location || null,
        keywords: inst.keywords || [],
        text:     inst.description,
        embedText: `${inst.name}: ${inst.description}`,
      });
    }

    // M2M parameters chunk — what the instrument actually measures
    if (params.length > 0) {
      const paramText = params
        .filter(p => p.name)
        .map(p => `${p.display_name || p.name}${p.units ? ` (${p.units})` : ""}${p.description ? ": " + p.description : ""}`)
        .join("\n");
      chunks.push({
        id:       `${inst.id}::params`,
        title:    `${inst.name} — measured parameters`,
        type:     inst.type,
        source:   inst.source,
        location: inst.location || null,
        keywords: inst.keywords || [],
        text:     `Parameters for ${inst.name}:\n${paramText}`,
        embedText: `What does ${inst.name} measure?\n${paramText}`,
      });
    }
  }

  // ── RCA site context chunks ────────────────────────────────────────────────
  if (RCA_CONTEXT) {
    for (const page of RCA_CONTEXT.pages) {
      if (!page.description) continue;
      chunks.push({
        id:       `rca::${page.id}`,
        title:    page.title,
        type:     "site-context",
        source:   "rca-website",
        location: page.location || null,
        keywords: ["regional cabled array", "OOI", "RCA", page.location || ""].filter(Boolean),
        text:     page.description,
        embedText: `About the ${page.title}:\n${page.description}`,
        linked_instruments: page.instruments || [],
      });
    }
  }

  // ── Zotero paper chunks ────────────────────────────────────────────────────
  if (PAPERS) {
    for (const paper of PAPERS.papers) {
      if (!paper.abstract || paper.abstract.length < 80) continue;
      const yearStr = paper.year ? ` (${paper.year})` : "";
      const journalStr = paper.journal ? ` — ${paper.journal}` : "";
      const linkStr = paper.linked_instruments?.length
        ? `\nRelevant instruments: ${paper.linked_instruments.join(", ")}`
        : "";
      const text = `${paper.title}${yearStr}${journalStr}\n\n${paper.abstract}${linkStr}`;
      chunks.push({
        id:           `paper::${paper.doi || paper.title.slice(0, 40).replace(/\s+/g, "-")}`,
        title:        paper.title,
        type:         "paper",
        source:       "zotero",
        location:     null,
        keywords:     paper.tags || [],
        year:         paper.year || null,
        journal:      paper.journal || null,
        first_author: paper.first_author || null,
        text,
        embedText: `Research paper: ${paper.title}\n\n${paper.abstract}`,
        linked_instruments: paper.linked_instruments || [],
      });
    }
  }

  // ── Data-access script pattern chunks ────────────────────────────────────────
  const scriptPatterns = [
    {
      id:       "script::ooi-m2m",
      title:    "Accessing OOI M2M API data (Python)",
      type:     "script",
      source:   "arcada-scripts",
      location: "Oregon/Washington offshore",
      keywords: ["OOI", "M2M API", "data access", "Python", "requests", "download", "time series"],
      text: `How to download OOI Regional Cabled Array data via the M2M REST API using Python.
Base URL: https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/{site}/{node}/{instrument}/{method}/{stream}
Requires OOI_USERNAME and OOI_TOKEN (register at ooinet.oceanobservatories.org).
Supports beginDT and endDT query parameters in ISO8601 format.
Returns JSON array of time series records with timestamp, measured variables, and QC flags.
Instrument ref designators follow the pattern SITE-NODE-PP-CLASS+NUM (e.g. RS01SLBS-MJ01A-05-HYDLFA101).
Method is 'streamed' for most cabled instruments; 'recovered_inst' for deep profiler wire-following instruments (node prefix DP).
Stream names depend on instrument class: e.g. botpt_nano_sample (pressure), ctdpf_optode_sample (CTD), vel3d_b_sample (velocimeter).`,
      embedText: "How do I download OOI M2M API data? Python script for accessing Regional Cabled Array time series data.",
    },
    {
      id:       "script::earthscope-fdsn",
      title:    "Accessing EarthScope seismic data via FDSN (Python / ObsPy)",
      type:     "script",
      source:   "arcada-scripts",
      location: "Axial Seamount, Hydrate Ridge",
      keywords: ["EarthScope", "FDSN", "ObsPy", "seismic", "waveform", "MiniSEED", "Python", "download", "seismometer", "hydrophone"],
      text: `How to download RCA seismic and hydrophone waveforms using ObsPy and the EarthScope FDSN service.
Client: obspy.clients.fdsn.Client("IRIS") — network OO.
Seismometer stations: AXAS1, AXAS2 (Axial), AXCC1, AXEC2, AXEC3, HYSB1 (Hydrate Ridge), and others.
Seismic channels: BH* (broadband, 40 Hz), HH* (high-gain, 100 Hz), EH* (short-period).
Hydrophone channels: HDH (200 Hz) and LDH (1 Hz decimated).
Use get_waveforms(network, station, location, channel, starttime, endtime) returning an ObsPy Stream.
Output formats: MiniSEED (.mseed), SAC, or NumPy arrays. Use stream.write(path, format="MSEED").
Large time ranges should be fetched day-by-day to manage memory.`,
      embedText: "How do I download seismic waveform data? ObsPy EarthScope FDSN script for RCA ocean bottom seismometer data.",
    },
    {
      id:       "script::pi-portal",
      title:    "Accessing PI-operated instrument data from piweb.ooirsn.uw.edu",
      type:     "script",
      source:   "arcada-scripts",
      location: "Hydrate Ridge, ASHES vent field",
      keywords: ["PI portal", "piweb", "manual download", "scanning sonar", "MASSP", "RASSP", "DAS", "Python", "BeautifulSoup"],
      text: `How to access PI-operated instrument data from the UW piweb portal (piweb.ooirsn.uw.edu).
Data is served via Apache directory listings organized by date: {base_url}/{YYYY}/{MM}/{timestamp}/data/
Use requests + BeautifulSoup to scrape directory listings and download individual data files.
Instruments and their base URLs:
- Scanning Sonar (OVRSRA101): http://piweb.ooirsn.uw.edu/marum/data/OVRSRA101/
- Scanning Sonar (QNTSRA101): http://piweb.ooirsn.uw.edu/marum/data/QNTSRA101/
- Mass Spectrometer MASSP: http://piweb.ooirsn.uw.edu/marum/data/MASSP/
- RASSP fluid sampler: http://piweb.ooirsn.uw.edu/marum/data/RASSP/
- COVIS sonar: http://piweb.ooirsn.uw.edu/covis/data/COVIS/
- DAS (Optasense): http://piweb.ooirsn.uw.edu/das/data/Optasense/
File formats vary by instrument (binary, CSV, proprietary). Many require manual inspection.`,
      embedText: "How do I download PI instrument data from piweb? Scraping Apache directory listings for scanning sonar, MASSP, DAS data.",
    },
  ];

  for (const sp of scriptPatterns) {
    chunks.push({ ...sp, embedText: sp.embedText });
  }

  return chunks;
}

// ── Embed ─────────────────────────────────────────────────────────────────────

async function embedBatch(texts, retries = 6) {
  const requests = texts.map(t => ({
    model: "models/gemini-embedding-2",
    content: { parts: [{ text: t }] },
    taskType: "RETRIEVAL_DOCUMENT",
    outputDimensionality: 768,
  }));

  for (let attempt = 0; attempt < retries; attempt++) {
    const res = await fetch(EMBED_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ requests }),
    });
    if (res.ok) {
      const data = await res.json();
      return data.embeddings.map(e => e.values);
    }
    const errText = await res.text();
    const delay = res.status === 429 ? 30000 * (attempt + 1) : 2000 * (attempt + 1);
    console.warn(`Embed attempt ${attempt + 1} failed (${res.status}): ${errText.slice(0, 120)}`);
    if (attempt < retries - 1) {
      console.warn(`  Waiting ${delay / 1000}s before retry...`);
      await new Promise(r => setTimeout(r, delay));
    }
  }
  throw new Error("Embedding failed after retries");
}

function l2Normalize(vec) {
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  return norm > 0 ? vec.map(v => v / norm) : vec;
}

// ── MiniSearch index ──────────────────────────────────────────────────────────

function buildSearchIndex(chunks) {
  // Minimal BM25-ready index structure for MiniSearch
  return {
    version:   "1.0",
    fields:    ["title", "text", "keywords", "type", "location"],
    documents: chunks.map(c => ({
      id:       c.id,
      title:    c.title,
      text:     c.text,
      keywords: (c.keywords || []).join(" "),
      type:     c.type,
      location: c.location || "",
    })),
  };
}

// ── Incremental embedding cache ───────────────────────────────────────────────

function loadExistingEmbeddings() {
  const chunksPath = path.join(OUT_DIR, "chunks.json");
  const embedPath  = path.join(OUT_DIR, "embeddings.bin");
  try {
    const oldChunks = JSON.parse(fs.readFileSync(chunksPath, "utf8"));
    const buf       = fs.readFileSync(embedPath);
    const floats    = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
    const dim       = floats.length / oldChunks.length;
    const cache     = new Map();
    oldChunks.forEach((c, i) => {
      cache.set(c.id, Array.from(floats.subarray(i * dim, (i + 1) * dim)));
    });
    console.log(`  Loaded ${cache.size} cached embeddings (dim=${dim})`);
    return cache;
  } catch {
    console.log("  No existing index found — embedding everything from scratch");
    return new Map();
  }
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  console.log("Building aRCADA RAG index...");
  const chunks = makeChunks();
  const instrCount = CATALOG.instruments.length + PI_PAGES.instruments.length;
  const paperCount = PAPERS?.papers?.length ?? 0;
  const pageCount  = RCA_CONTEXT?.pages?.length ?? 0;
  console.log(`  ${chunks.length} chunks from ${instrCount} instruments, ${paperCount} papers, ${pageCount} RCA pages`);

  // Load existing embeddings cache to avoid re-embedding unchanged chunks
  const cache    = loadExistingEmbeddings();
  const toEmbed  = chunks.filter(c => !cache.has(c.id));
  const reused   = chunks.length - toEmbed.length;
  console.log(`  ${reused} chunks reused from cache, ${toEmbed.length} need embedding`);

  // Embed only new/changed chunks
  const newVecMap = new Map();
  if (toEmbed.length > 0) {
    const batchSize = CONFIG.batchSize || 25;
    for (let i = 0; i < toEmbed.length; i += batchSize) {
      const batch = toEmbed.slice(i, i + batchSize);
      const batchNum = Math.floor(i / batchSize) + 1;
      const totalBatches = Math.ceil(toEmbed.length / batchSize);
      console.log(`  Embedding batch ${batchNum}/${totalBatches} (${batch.length} chunks)...`);
      const vecs = await embedBatch(batch.map(c => c.embedText));
      batch.forEach((c, j) => newVecMap.set(c.id, l2Normalize(vecs[j])));
      if (i + batchSize < toEmbed.length) {
        await new Promise(r => setTimeout(r, 15000)); // stay under rate limit
      }
    }
  }

  // Assemble final embeddings in chunk order
  const allVecs = chunks.map(c => newVecMap.get(c.id) ?? cache.get(c.id));

  // Write embeddings.bin (Float32, row-major)
  const dim = allVecs[0].length;
  const bin = new Float32Array(allVecs.length * dim);
  allVecs.forEach((v, i) => bin.set(v, i * dim));
  fs.writeFileSync(path.join(OUT_DIR, "embeddings.bin"), Buffer.from(bin.buffer));
  console.log(`  embeddings.bin: ${allVecs.length} × ${dim}`);

  // Write chunks.json (strip embedText to save bandwidth)
  const chunksOut = chunks.map(({ embedText, ...rest }) => rest);
  fs.writeFileSync(path.join(OUT_DIR, "chunks.json"), JSON.stringify(chunksOut));
  console.log(`  chunks.json: ${chunksOut.length} entries`);

  // Write search-index.json
  const searchIdx = buildSearchIndex(chunks);
  fs.writeFileSync(path.join(OUT_DIR, "search-index.json"), JSON.stringify(searchIdx));
  console.log(`  search-index.json: ${searchIdx.documents.length} documents`);

  console.log(`Done. (${reused} cached, ${toEmbed.length} newly embedded)`);
}

main().catch(e => { console.error(e); process.exit(1); });
