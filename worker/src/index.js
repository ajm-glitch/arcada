// aRCADA Cloudflare Worker
// Routes: POST /embed, POST /chat, POST /plan, POST /dispatch, GET /status/:runId

const GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent";
const GEMINI_CHAT_URL  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:streamGenerateContent?alt=sse";
const GEMINI_JSON_URL  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent";
const GITHUB_API       = "https://api.github.com";

// Fallback model chain — tried in order on 429 for both streaming and JSON endpoints
const FALLBACK_MODELS = [
  "gemini-2.0-flash-lite",
  "gemini-2.0-flash",
];

const MAX_CONTEXT_LEN = 20000; // chars — hard cap on context block before sending to Gemini

// Shared helper: POST to Gemini generateContent with model fallback on 429
async function geminiJson(apiKey, body) {
  let resp;
  for (const model of FALLBACK_MODELS) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
    for (let attempt = 0; attempt < 3; attempt++) {
      resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (resp.status !== 503) break;
      const ra = resp.headers.get("Retry-After");
      const wait = ra ? Math.max(1000, parseInt(ra, 10) * 1000) : ([1000, 2000][attempt] ?? 2000);
      await new Promise(r => setTimeout(r, wait));
    }
    if (resp.status !== 429) break;
  }
  return resp;
}

const SYSTEM_PROMPT = `You are aRCADA, an expert data assistant for the OOI Regional Cabled Array (RCA) and EarthScope seafloor observatory networks on the Cascadia margin.

## Instruments you can access
- **Ocean bottom seismometers (OBS / seismometer)** — earthquakes, tremor, volcanic eruptions at Axial Seamount and Hydrate Ridge
- **Seafloor pressure sensors (BOTPT)** — seafloor deformation, slow-slip events, tidal loading
- **CTD profilers** — ocean temperature, salinity, and pressure through the water column
- **Hydrophones** — acoustic signals from methane bubble plumes, earthquakes, fin whales, and cetaceans
- **pCO2 sensors** — dissolved carbon dioxide near methane seeps
- **PI-operated instruments** — scanning sonar (Hydrate Ridge), mass spectrometers (ASHES vent field), and other portal-hosted data

## Key sites
Axial Seamount (active submarine volcano), Hydrate Ridge (methane seep field), ASHES hydrothermal vent field, Southern Hydrate Ridge, Endurance Array offshore Oregon.

## How to respond

**Capability questions** ("what can you do?", "what data is available?", "what instruments are there?"):
Describe the instrument types, sites, and research topics above. Mention that you can pull time-series data and link researchers to relevant published literature. Be concise and inviting.

**Literature / paper questions** ("what papers have been published on X?", "what research exists on Y?", "summarize the literature on Z?"):
Use the [paper] entries in the context below to answer. For each relevant paper, cite it as "Author et al. (Year) — Journal" and give a one-sentence summary of its finding. Group by sub-topic if helpful. If no papers are in context, say so honestly and suggest the user try a more specific query.

**Data requests** ("get me pressure data from...", "fetch seismic records for..."):
Identify which instruments and sites are relevant, clarify the time range, and confirm before pulling. Ask one focused clarifying question if the request is ambiguous.

**Follow-up / conversational turns**:
Use prior conversation context to give coherent, non-repetitive replies.

Respond clearly and concisely. Never invent instrument IDs or paper citations that are not in the provided context.`;

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
  if (!text) return new Response("Missing text", { status: 400, headers: cors(env) });

  const vec = await retry(async () => {
    const r = await fetch(`${GEMINI_EMBED_URL}?key=${env.GEMINI_API_KEY}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "models/gemini-embedding-2", content: { parts: [{ text }] }, outputDimensionality: 768 }),
    });
    if (!r.ok) throw new Error(`Gemini embed ${r.status}`);
    const data = await r.json();
    return data.embedding.values;
  });

  return Response.json({ embedding: vec }, { headers: cors(env) });
}

// ── /welcome ──────────────────────────────────────────────────────────────────
async function handleWelcome(req, env) {
  const { paperSamples } = await req.json().catch(() => ({}));

  const papersCtx = (paperSamples || []).length
    ? `\n\nRecent papers from the RCA research literature:\n${
        paperSamples.map(p => `- "${p.title}" (${p.first_author ?? "—"}, ${p.year ?? "n.d."})`).join("\n")
      }`
    : "";

  const prompt = `Generate a brief, friendly welcome message for a researcher who just opened the aRCADA data interface.${papersCtx}

Requirements:
- One welcoming sentence to open
- Mention 2–3 specific research directions (e.g. seismicity at Axial Seamount, methane flux at Hydrate Ridge, hydrothermal vent chemistry at ASHES)${papersCtx ? " — draw these from the paper topics listed above" : ""}
- End with something like "or did you have something else in mind?"
- Under 80 words, plain prose, no bullet points or markdown`;

  const result = await retry(() =>
    fetch(`${GEMINI_JSON_URL}?key=${env.GEMINI_API_KEY}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        system_instruction: { parts: [{ text: SYSTEM_PROMPT }] },
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.7 },
      }),
    }).then(r => r.json())
  );

  const text = result?.candidates?.[0]?.content?.parts?.[0]?.text?.trim()
    ?? "Hello! I'm aRCADA, your seafloor data assistant. I can help you access seismic, geochemical, and acoustic data from the Regional Cabled Array — or did you have something else in mind?";

  return Response.json({ text }, { headers: cors(env) });
}

// ── /chat ─────────────────────────────────────────────────────────────────────
async function handleChat(req, env) {
  const { query, context, history } = await req.json();
  if (!query) return new Response("Missing query", { status: 400, headers: cors(env) });

  const instruments = (context || []).filter(c => c.type !== "paper" && c.type !== "site-context");
  const papers      = (context || []).filter(c => c.type === "paper");
  const siteCtx     = (context || []).filter(c => c.type === "site-context");

  const instrBlock = instruments.length
    ? `\n\n## Relevant instruments\n${instruments.map(c => `- **${c.title}** (${c.location || c.source}): ${c.text}`).join("\n\n")}`
    : "";
  const paperBlock = papers.length
    ? `\n\n## Relevant papers from the literature\n${papers.map(c => {
        const citation = [c.first_author, c.journal, c.year].filter(Boolean).join(", ");
        return `- **${c.title}**${citation ? ` — ${citation}` : ""}\n  ${c.text.split("\n\n")[1] || c.text.slice(0, 400)}`;
      }).join("\n\n")}`
    : "";
  const siteBlock = siteCtx.length
    ? `\n\n## Background\n${siteCtx.map(c => c.text).join("\n\n")}`
    : "";

  const rawContextBlock = (instrBlock || paperBlock || siteBlock)
    ? instrBlock + paperBlock + siteBlock
    : "";
  const contextBlock = rawContextBlock.length > MAX_CONTEXT_LEN
    ? rawContextBlock.slice(0, MAX_CONTEXT_LEN) + "\n[context truncated]"
    : rawContextBlock;

  // Build multi-turn contents: history first, then the current user turn
  const contents = [];
  if (history?.length) {
    for (const msg of history) {
      contents.push({ role: msg.role, parts: msg.parts ?? [{ text: msg.text ?? "" }] });
    }
  }
  contents.push({ role: "user", parts: [{ text: query + contextBlock }] });

  const geminiReq = {
    system_instruction: { parts: [{ text: SYSTEM_PROMPT }] },
    contents,
    generationConfig: { temperature: 0.3 },
  };

  // Model fallback: try flash-lite first, then flash on 429.
  // Only retry 503 (transient overload) — never retry 429 (quota exhausted).
  // Retrying 429 multiplies consumption and deepens the rate-limit hole.
  let upstream;
  for (const model of FALLBACK_MODELS) {
    const chatUrl = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse`;
    for (let attempt = 0; attempt < 3; attempt++) {
      upstream = await fetch(`${chatUrl}&key=${env.GEMINI_API_KEY}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(geminiReq),
      });
      if (upstream.status !== 503) break;
      const ra = upstream.headers.get("Retry-After");
      const wait = ra ? Math.max(1000, parseInt(ra, 10) * 1000) : ([1000, 2000][attempt] ?? 2000);
      await new Promise(r => setTimeout(r, wait));
    }
    if (upstream.status !== 429) break; // success or non-retriable error — stop trying models
  }

  if (!upstream.ok) {
    const errBody = await upstream.text();
    const msg = upstream.status === 429
      ? "The AI model is rate-limited. Please wait a moment and try again."
      : upstream.status === 503
      ? "The AI model is temporarily unavailable. Please try again in a few seconds."
      : `Gemini error ${upstream.status}: ${errBody.slice(0, 200)}`;
    return new Response(JSON.stringify({ error: msg }), { status: upstream.status, headers: { ...cors(env), "Content-Type": "application/json" } });
  }

  return new Response(upstream.body, {
    headers: { ...cors(env), "Content-Type": "text/event-stream", "Cache-Control": "no-cache" },
  });
}

// ── /plan ─────────────────────────────────────────────────────────────────────
// Converts a plain-language query + instrument context into a structured data plan
async function handlePlan(req, env) {
  const { query, context } = await req.json();
  if (!query) return new Response("Missing query", { status: 400, headers: cors(env) });

  // Only use instrument chunks (drop papers / site-context — not needed for plan)
  const REFERENCE_TYPES = new Set(["paper", "site-context"]);
  const instrChunks = (context || []).filter(c => !REFERENCE_TYPES.has(c.type));

  // Valid IDs are derived from the retrieved context, not a client-sent catalog
  const validInstruments = new Map(
    instrChunks.map(c => [c.id.replace(/::.*$/, ""), c])
  );

  const instrSummary = instrChunks.length
    ? instrChunks.map(c =>
        `${c.id.replace(/::.*$/, "")} | ${c.title || c.name} | type:${c.type} | source:${c.source}${c.location ? ` | ${c.location}` : ""}`
      ).join("\n")
    : "none";

  const planPrompt = `Extract a structured data plan from this request.

User request: "${query}"

Available instruments (use IDs exactly as listed):
${instrSummary}

Return ONLY valid JSON:
{
  "summary": "one sentence describing what will be fetched",
  "time_range": { "start": "ISO8601", "end": "ISO8601", "notes": "assumptions if any" },
  "instruments": [
    { "id": "exact id from list above", "name": "human name", "type": "seismometer|pressure|ctd|hydrophone|pco2|thermistor|sonar|mass_spectrometer", "source": "ooi_api|earthscope|pi_html", "priority": "primary|supplementary", "rationale": "one phrase" }
  ],
  "output_format": "zarr",
  "metadata_requested": ["instrument_info", "coverage_dates", "gaps", "units", "provenance"]
}`;

  const planResp = await geminiJson(env.GEMINI_API_KEY, {
    contents: [{ role: "user", parts: [{ text: planPrompt }] }],
    generationConfig: { temperature: 0.1, responseMimeType: "application/json" },
  });

  if (!planResp.ok) {
    const errBody = await planResp.text();
    const msg = planResp.status === 429
      ? "The AI model is rate-limited. Please wait a moment and try again."
      : planResp.status === 503
      ? "The AI model is temporarily unavailable. Please try again in a few seconds."
      : `Gemini error ${planResp.status}: ${errBody.slice(0, 200)}`;
    return new Response(JSON.stringify({ error: msg }), { status: planResp.status, headers: { ...cors(env), "Content-Type": "application/json" } });
  }

  const result = await planResp.json();

  // Detect Gemini API-level errors (quota, bad key, safety, etc.)
  if (result?.error) {
    return new Response(JSON.stringify({ error: "Gemini API error", detail: result.error }), { status: 502, headers: cors(env) });
  }
  if (!result?.candidates?.length) {
    return new Response(JSON.stringify({ error: "Gemini returned no candidates", raw: result }), { status: 502, headers: cors(env) });
  }

  const raw = result.candidates[0]?.content?.parts?.[0]?.text ?? "{}";
  let plan;
  try { plan = JSON.parse(raw); }
  catch { return new Response(JSON.stringify({ error: "Failed to parse plan", raw }), { status: 500, headers: cors(env) }); }

  // Validate and fix instrument IDs against the catalog
  const geminiInstruments = plan.instruments ?? [];
  if (plan.instruments) {
    plan.instruments = plan.instruments
      .map(inst => {
        const baseId = inst.id.replace(/::.*$/, "");
        const known = validInstruments.get(baseId) ?? validInstruments.get(inst.id);
        if (!known) return null; // drop hallucinated IDs
        // Correct source if Gemini got it wrong; normalize to base ID
        return { ...inst, id: baseId, source: known.source ?? inst.source };
      })
      .filter(Boolean);
  }

  // Include debug info so the UI can surface validation failures
  const debug = {
    geminiInstrumentIds: geminiInstruments.map(i => i.id),
    validatedInstrumentIds: (plan.instruments || []).map(i => i.id),
    catalogIds: [...validInstruments.keys()],
  };

  return Response.json({ plan, debug }, { headers: cors(env) });
}

// ── /dispatch ─────────────────────────────────────────────────────────────────
// Triggers a GitHub Actions workflow to run the Python data pull
async function handleDispatch(req, env) {
  const { plan } = await req.json();
  if (!plan) return new Response("Missing plan", { status: 400, headers: cors(env) });

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
        "User-Agent": "arcada-data-assistant",
        "X-GitHub-Api-Version": "2022-11-28",
      },
      body: JSON.stringify({ ref: "main", inputs: { plan: JSON.stringify(plan) } }),
    }
  );

  if (!dispatchRes.ok) {
    const err = await dispatchRes.text();
    return new Response(JSON.stringify({ error: "GitHub dispatch failed", detail: err }), { status: 502, headers: cors(env) });
  }

  // GitHub Actions takes a moment to register the run — wait briefly then find its ID
  await new Promise(r => setTimeout(r, 3000));

  const runsRes = await fetch(
    `${GITHUB_API}/repos/${owner}/${repo}/actions/workflows/fetch-data.yml/runs?per_page=1`,
    {
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/vnd.github+json",
        "User-Agent": "arcada-data-assistant",
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
    "User-Agent": "arcada-data-assistant",
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

// ── /ack ──────────────────────────────────────────────────────────────────────
// Returns a short conversational acknowledgment. Accepts intent: "data" | "question" | "literature"
async function handleAck(req, env) {
  const { query, instruments, intent = "data" } = await req.json();
  const matches = (instruments || []).slice(0, 6);

  let prompt, fallback;

  if (intent === "literature") {
    const titles = matches.map(i => `"${i.name}"`).join(", ");
    prompt = `A researcher asked about research on: "${query}"${titles ? `\nRelevant papers found: ${titles}.` : ""}

In 1 sentence, acknowledge what literature topic you're about to summarize. Be direct and natural.`;
    fallback = `Searching the literature on ${query.slice(0, 60).trim()}…`;

  } else if (intent === "question") {
    const topics = matches.map(i => `${i.name}${i.location ? ` at ${i.location}` : ""}`).join(", ");
    prompt = `A researcher asked: "${query}"${topics ? `\nRelevant instruments/topics: ${topics}.` : ""}

In 1 sentence, acknowledge the question naturally. Do not answer it — just confirm you understood it.`;
    fallback = `Looking into that…`;

  } else {
    // data (default)
    const instrList = matches.map(i => `${i.name}${i.location ? ` at ${i.location}` : ""}`).join(", ");
    prompt = `A researcher submitted this data request: "${query}"
Matched instruments: ${instrList || "none found"}.

In 1–2 sentences, confirm what you understood they're asking for — mention the instrument type, site, and time period if specified. Be natural and conversational. Do not make promises about data availability or say what you "will" do.`;
    const names = matches.slice(0, 3).map(i => i.name).filter(Boolean);
    fallback = names.length
      ? `Got it — looking for ${names.join(" and ")} data${instrList.includes(" at ") ? ` from the matched sites` : ""}.`
      : `Got it — searching for relevant data based on your request.`;
  }

  const ackResp = await geminiJson(env.GEMINI_API_KEY, {
    contents: [{ role: "user", parts: [{ text: prompt }] }],
    generationConfig: { temperature: 0.5, maxOutputTokens: 100 },
  });

  const data = ackResp.ok ? await ackResp.json().catch(() => ({})) : {};
  const ack = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || fallback;

  return Response.json({ ack }, { headers: cors(env) });
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
    if (url.pathname === "/welcome"  && req.method === "POST") return handleWelcome(req, env);
    if (url.pathname === "/chat"     && req.method === "POST") return handleChat(req, env);
    if (url.pathname === "/ack"      && req.method === "POST") return handleAck(req, env);
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
