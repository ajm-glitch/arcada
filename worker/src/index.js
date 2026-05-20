// aRCADA Cloudflare Worker
// Routes: POST /embed, POST /chat, POST /plan, POST /dispatch, GET /status/:runId

const GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent";
// gemini-embedding-001 returns 3072-dim vectors
const GEMINI_CHAT_URL  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse";
const GEMINI_JSON_URL  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent";
const GITHUB_API       = "https://api.github.com";

const SYSTEM_PROMPT = `You are aRCADA, an expert data assistant for the OOI Regional Cabled Array (RCA) and EarthScope seafloor observatory networks.

You help researchers access data from instruments on the Cascadia margin including:
- Ocean bottom seismometers (OBS) monitoring earthquakes, tremor, and volcanic eruptions
- Seafloor pressure sensors (BOTPT) tracking seafloor deformation, slow-slip events, and tidal loading
- CTD profilers measuring ocean temperature, salinity, and pressure through the water column
- Hydrophones recording acoustic signals from methane bubble plumes, earthquakes, and cetaceans
- pCO2 sensors measuring dissolved carbon dioxide near methane seeps
- PI-operated instruments (sonar, mass spectrometers) accessible via HTTP data portals

When a user requests data, identify:
1. Which instruments and sites are relevant
2. The time range they need
3. Any specific parameters or event context (e.g. "two weeks after the 2015 Axial eruption")

Respond clearly and concisely. When returning structured data plans, use JSON blocks.`;

function cors(env) {
  return {
    "Access-Control-Allow-Origin": env.ALLOWED_ORIGIN || "*",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
  };
}

async function retry(fn, attempts = 3, delayMs = 800) {
  for (let i = 0; i < attempts; i++) {
    try { return await fn(); }
    catch (e) {
      if (i === attempts - 1) throw e;
      await new Promise(r => setTimeout(r, delayMs * (i + 1)));
    }
  }
}

// ── /embed ────────────────────────────────────────────────────────────────────
async function handleEmbed(req, env) {
  const { text } = await req.json();
  if (!text) return new Response("Missing text", { status: 400 });

  const vec = await retry(() =>
    fetch(`${GEMINI_EMBED_URL}?key=${env.GEMINI_API_KEY}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "models/gemini-embedding-001", content: { parts: [{ text }] } }),
    }).then(r => r.json()).then(r => r.embedding.values)
  );

  return Response.json({ embedding: vec }, { headers: cors(env) });
}

// ── /chat ─────────────────────────────────────────────────────────────────────
async function handleChat(req, env) {
  const { query, context } = await req.json();
  if (!query) return new Response("Missing query", { status: 400 });

  const contextBlock = context?.length
    ? `\n\nRelevant context from the instrument catalog and research literature:\n${
        context.map(c => `[${c.type ?? "instrument"}] ${c.title ?? c.name}: ${c.text}`).join("\n\n")
      }`
    : "";

  const geminiReq = {
    system_instruction: { parts: [{ text: SYSTEM_PROMPT }] },
    contents: [{ role: "user", parts: [{ text: query + contextBlock }] }],
    generationConfig: { temperature: 0.3 },
  };

  const upstream = await retry(() =>
    fetch(`${GEMINI_CHAT_URL}&key=${env.GEMINI_API_KEY}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(geminiReq),
    })
  );

  return new Response(upstream.body, {
    headers: { ...cors(env), "Content-Type": "text/event-stream", "Cache-Control": "no-cache" },
  });
}

// ── /plan ─────────────────────────────────────────────────────────────────────
// Converts a plain-language query + instrument context into a structured data plan
async function handlePlan(req, env) {
  const { query, context, catalog } = await req.json();
  if (!query) return new Response("Missing query", { status: 400 });

  // Separate instrument chunks from reference material (papers, site context)
  const REFERENCE_TYPES = new Set(["paper", "site-context"]);
  const instrChunks = (context || []).filter(c => !REFERENCE_TYPES.has(c.type));
  const refChunks   = (context || []).filter(c =>  REFERENCE_TYPES.has(c.type));

  // Build a map of valid instrument IDs from the catalog for post-plan validation
  const validInstruments = new Map(
    (catalog || []).map(c => [c.id, c])
  );

  const instrSummary = instrChunks.length
    ? instrChunks.map(c =>
        `ID: ${c.id}\nName: ${c.title || c.name}\nType: ${c.type}\nSource: ${c.source}\nLocation: ${c.location || ""}\nContext: ${c.text}`
      ).join("\n\n")
    : "No instruments retrieved — use the catalog below.";

  const refSummary = refChunks.length
    ? refChunks.map(c => `- ${c.title}: ${c.text.slice(0, 300)}`).join("\n")
    : "";

  const catalogSummary = validInstruments.size
    ? [...validInstruments.values()].map(c =>
        `${c.id} | ${c.title || c.name} | type:${c.type} | source:${c.source}`
      ).join("\n")
    : "";

  const planPrompt = `${SYSTEM_PROMPT}

Based on the following user request, produce a structured JSON data plan.

User request: "${query}"

INSTRUMENTS RETRIEVED (ranked by relevance — pick from these first):
${instrSummary}

${refSummary ? `SCIENCE CONTEXT (papers and background — do NOT use these IDs as instruments):
${refSummary}

` : ""}${catalogSummary ? `FULL INSTRUMENT CATALOG (authoritative — all valid instrument IDs):
${catalogSummary}

` : ""}You MUST only use instrument IDs that appear in the catalog above. Do not invent IDs.

Return ONLY valid JSON with this exact structure:
{
  "summary": "One sentence describing what will be fetched",
  "time_range": {
    "start": "ISO8601 datetime",
    "end": "ISO8601 datetime",
    "notes": "any ambiguity or assumptions made"
  },
  "instruments": [
    {
      "id": "instrument id from catalog",
      "name": "human readable name",
      "type": "seismometer|pressure|ctd|hydrophone|pco2|thermistor|sonar|mass_spectrometer",
      "source": "ooi_api|earthscope|pi_html",
      "priority": "primary|supplementary",
      "rationale": "why this instrument is relevant to the request"
    }
  ],
  "output_format": "zarr",
  "metadata_requested": ["instrument_info", "coverage_dates", "gaps", "units", "provenance"]
}`;

  const result = await retry(() =>
    fetch(`${GEMINI_JSON_URL}?key=${env.GEMINI_API_KEY}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ role: "user", parts: [{ text: planPrompt }] }],
        generationConfig: { temperature: 0.1, responseMimeType: "application/json" },
      }),
    }).then(r => r.json())
  );

  const raw = result?.candidates?.[0]?.content?.parts?.[0]?.text ?? "{}";
  let plan;
  try { plan = JSON.parse(raw); }
  catch { return new Response(JSON.stringify({ error: "Failed to parse plan", raw }), { status: 500 }); }

  // Validate and fix instrument IDs against the catalog
  if (plan.instruments) {
    plan.instruments = plan.instruments
      .map(inst => {
        const known = validInstruments.get(inst.id);
        if (!known) return null; // drop hallucinated IDs
        // Correct source if Gemini got it wrong
        return { ...inst, source: known.source ?? inst.source };
      })
      .filter(Boolean);
  }

  return Response.json({ plan }, { headers: cors(env) });
}

// ── /dispatch ─────────────────────────────────────────────────────────────────
// Triggers a GitHub Actions workflow to run the Python data pull
async function handleDispatch(req, env) {
  const { plan } = await req.json();
  if (!plan) return new Response("Missing plan", { status: 400 });

  const owner = env.GITHUB_REPO_OWNER;
  const repo  = env.GITHUB_REPO_NAME;
  const token = env.GITHUB_PAT;

  const dispatchRes = await fetch(
    `${GITHUB_API}/repos/${owner}/${repo}/actions/workflows/fetch-data.yml/dispatches`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/vnd.github+json",
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
      },
      body: JSON.stringify({ ref: "main", inputs: { plan: JSON.stringify(plan) } }),
    }
  );

  if (!dispatchRes.ok) {
    const err = await dispatchRes.text();
    return new Response(JSON.stringify({ error: "GitHub dispatch failed", detail: err }), { status: 502 });
  }

  // GitHub Actions takes a moment to register the run — wait briefly then find its ID
  await new Promise(r => setTimeout(r, 3000));

  const runsRes = await fetch(
    `${GITHUB_API}/repos/${owner}/${repo}/actions/workflows/fetch-data.yml/runs?per_page=1`,
    {
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
      },
    }
  );
  const runsData = await runsRes.json();
  const runId = runsData?.workflow_runs?.[0]?.id;

  return Response.json({ runId, status: "queued" }, { headers: cors(env) });
}

// ── /status/:runId ─────────────────────────────────────────────────────────────
// Polls GitHub Actions run status and returns artifact download URL when complete
async function handleStatus(runId, env) {
  const owner = env.GITHUB_REPO_OWNER;
  const repo  = env.GITHUB_REPO_NAME;
  const token = env.GITHUB_PAT;
  const ghHeaders = {
    Authorization: `Bearer ${token}`,
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
  };

  const runRes = await fetch(
    `${GITHUB_API}/repos/${owner}/${repo}/actions/runs/${runId}`,
    { headers: ghHeaders }
  );
  const run = await runRes.json();

  if (run.status !== "completed") {
    return Response.json({ status: run.status, conclusion: null }, { headers: cors(env) });
  }

  if (run.conclusion !== "success") {
    return Response.json({ status: "completed", conclusion: run.conclusion, error: "Data pull job failed" }, { headers: cors(env) });
  }

  // Fetch artifact download URL
  const artRes = await fetch(
    `${GITHUB_API}/repos/${owner}/${repo}/actions/runs/${runId}/artifacts`,
    { headers: ghHeaders }
  );
  const artData = await artRes.json();
  const artifact = artData?.artifacts?.find(a => a.name === "arcada-data");

  return Response.json({
    status: "completed",
    conclusion: "success",
    artifactId: artifact?.id,
    artifactName: artifact?.name,
    artifactSize: artifact?.size_in_bytes,
    downloadUrl: artifact
      ? `${GITHUB_API}/repos/${owner}/${repo}/actions/artifacts/${artifact.id}/zip`
      : null,
    note: artifact ? "Use your GitHub PAT as a Bearer token to download the artifact." : "No artifact found.",
  }, { headers: cors(env) });
}

// ── Router ────────────────────────────────────────────────────────────────────
export default {
  async fetch(req, env) {
    const url = new URL(req.url);

    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: cors(env) });
    }

    // Optional password gate
    if (env.CHAT_PASSWORD) {
      const auth = req.headers.get("Authorization") ?? "";
      const provided = auth.replace(/^Bearer\s+/i, "");
      if (!timingSafeEqual(provided, env.CHAT_PASSWORD)) {
        return new Response("Unauthorized", { status: 401, headers: cors(env) });
      }
    }

    if (url.pathname === "/embed"    && req.method === "POST") return handleEmbed(req, env);
    if (url.pathname === "/chat"     && req.method === "POST") return handleChat(req, env);
    if (url.pathname === "/plan"     && req.method === "POST") return handlePlan(req, env);
    if (url.pathname === "/dispatch" && req.method === "POST") return handleDispatch(req, env);

    const statusMatch = url.pathname.match(/^\/status\/(\d+)$/);
    if (statusMatch && req.method === "GET") return handleStatus(statusMatch[1], env);

    return new Response("Not found", { status: 404 });
  },
};

function timingSafeEqual(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}
