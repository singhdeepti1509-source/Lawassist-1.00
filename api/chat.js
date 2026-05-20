// api/chat.js  ← put this in your project root /api folder

export const config = { runtime: "edge" }; // ✅ No timeout on Edge!

const SPACE_URL = "https://deepti-singh-196-lawassit-version1-rag.hf.space";

export default async function handler(req) {
  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { "Content-Type": "application/json" },
    });
  }

  let body;
  try {
    body = await req.json();
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON body" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const { message, history = [] } = body;

  if (!message) {
    return new Response(JSON.stringify({ error: "No message provided" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  // ✅ Convert {role, content}[] → [[user, assistant], ...] for Gradio
  const gradioHistory = [];
  for (let i = 0; i < history.length - 1; i += 2) {
    const user = history[i];
    const assistant = history[i + 1];
    if (user?.role === "user" && assistant?.role === "assistant") {
      gradioHistory.push([user.content, assistant.content]);
    }
  }

  try {
    // ── Step 1: Join the Gradio queue ─────────────────────────────────
    const joinRes = await fetch(`${SPACE_URL}/queue/join`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        data: [message, gradioHistory],
        fn_index: 0,        // ← ChatInterface is always fn_index 0
        session_hash: Math.random().toString(36).slice(2),
      }),
    });

    if (!joinRes.ok) {
      const err = await joinRes.text();
      throw new Error(`Queue join failed: ${err}`);
    }

    const { event_id } = await joinRes.json();

    // ── Step 2: Poll the data stream for the result ───────────────────
    const dataRes = await fetch(
      `${SPACE_URL}/queue/data?session_hash=${event_id}`,
      { headers: { Accept: "text/event-stream" } }
    );

    if (!dataRes.ok) {
      throw new Error(`Stream failed: ${dataRes.status}`);
    }

    // ── Step 3: Read SSE stream until we get the final output ─────────
    const reader = dataRes.body.getReader();
    const decoder = new TextDecoder();
    let reply = null;
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        try {
          const json = JSON.parse(line.slice(5).trim());

          if (json.msg === "process_completed") {
            // ✅ Gradio returns data[0] as the assistant reply
            reply = json.output?.data?.[0] ?? null;
            break;
          }
          if (json.msg === "queue_full") {
            throw new Error("Space queue is full. Please try again.");
          }
        } catch (parseErr) {
          // skip malformed SSE lines
        }
      }
      if (reply !== null) break;
    }

    if (!reply) {
      throw new Error("No response from model. Space may be loading.");
    }

    return new Response(JSON.stringify({ reply }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });

  } catch (err) {
    console.error("HF Space error:", err.message);
    return new Response(
      JSON.stringify({ error: err.message || "Space unreachable" }),
      { status: 503, headers: { "Content-Type": "application/json" } }
    );
  }
}
