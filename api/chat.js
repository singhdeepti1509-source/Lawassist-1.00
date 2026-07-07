// api/chat.js  ← put this in your project's /api folder
//
// Uses the official @gradio/client package instead of hand-rolling Gradio's
// internal /queue/join + /queue/data SSE protocol. That protocol isn't a
// stable public API — its endpoint paths and message shapes have changed
// between Gradio major versions (e.g. /queue/join vs /gradio_api/queue/join),
// so hand-rolled fetch calls are fragile and hard to keep working.
//
// Run `npm install @gradio/client` and add it to package.json before deploying.
//
// NOTE: switched from `runtime: "edge"` to Node.js. @gradio/client relies on
// APIs (WebSocket/EventSource internals) that aren't guaranteed to work in
// the Edge runtime. If you're on Vercel Hobby, serverless functions have a
// hard execution cap (historically 10s, up to 60s on some plans) — a
// ZeroGPU cold start (30-60s+) can still exceed that. If you hit timeouts,
// either upgrade your Vercel plan for a higher `maxDuration`, or move to a
// pattern where the frontend polls a status endpoint instead of waiting on
// one long request.

import { Client } from "@gradio/client";

export const config = {
  maxDuration: 60, // requires a Vercel plan that allows this; see note above
};

const SPACE_URL = "https://deepti-singh-196-lawassit-version1-rag.hf.space";

// Reuse the connected client across warm invocations instead of reconnecting
// on every request.
let clientPromise = null;
function getClient() {
  if (!clientPromise) {
    clientPromise = Client.connect(SPACE_URL);
  }
  return clientPromise;
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const { message, history = [] } = req.body || {};
  if (!message) {
    res.status(400).json({ error: "No message provided" });
    return;
  }

  // Convert {role, content}[] -> [[user, assistant], ...] for Gradio's
  // ChatInterface history format.
  const gradioHistory = [];
  for (let i = 0; i < history.length - 1; i += 2) {
    const user = history[i];
    const assistant = history[i + 1];
    if (user?.role === "user" && assistant?.role === "assistant") {
      gradioHistory.push([user.content, assistant.content]);
    }
  }

  try {
    const client = await getClient();

    // ChatInterface exposes its predict function at the "/chat" api_name by
    // default. The fn signature is respond(message, history), so pass
    // positional args in that order.
    const result = await client.predict("/chat", [message, gradioHistory]);

    // result.data is an array matching the outputs of the Gradio fn.
    // ChatInterface's respond() returns a single string, so data[0] is it.
    const reply = Array.isArray(result?.data) ? result.data[0] : result?.data;

    if (!reply) {
      throw new Error("Empty response from model. Space may still be loading.");
    }

    res.status(200).json({ reply });
  } catch (err) {
    console.error("HF Space error:", err);
    res.status(503).json({ error: err?.message || "Space unreachable" });
  }
}
