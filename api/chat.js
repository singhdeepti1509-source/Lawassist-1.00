// api/chat.js  ← put this in your project's /api folder
//
// Uses the official @gradio/client package to call the deployed HF Space.
//
// IMPORTANT: the Space's exposed API endpoint is "/respond" and it accepts
// ONLY a `message` string — no `history` parameter. Gradio's ChatInterface
// keeps conversation history as an internal State component, which isn't
// exposed through the public API by default. That means the model itself
// only ever sees the latest message, not prior turns.
//
// If you want the model to have conversational context, you have to build
// that context into the `message` string yourself before sending it (see
// buildPromptWithHistory below) — there's no other way to pass it through
// this API as currently exposed.

import { Client } from "@gradio/client";

export const config = {
  maxDuration: 60, // requires a Vercel plan that allows this
};

const SPACE_URL = "Deepti-singh-196/LawAssit_Version1_RAG"; // HF Space repo id

let clientPromise = null;
function getClient() {
  if (!clientPromise) {
    clientPromise = Client.connect(SPACE_URL);
  }
  return clientPromise;
}

// Folds prior turns into a single prompt string, since the API endpoint
// doesn't accept history directly. Keeps only the last few turns to avoid
// blowing past the model's context window.
function buildPromptWithHistory(message, history, maxTurns = 4) {
  if (!history?.length) return message;

  const recent = history.slice(-maxTurns * 2); // history is {role, content}[]
  const turns = [];
  for (let i = 0; i < recent.length - 1; i += 2) {
    const user = recent[i];
    const assistant = recent[i + 1];
    if (user?.role === "user" && assistant?.role === "assistant") {
      turns.push(`User: ${user.content}\nAssistant: ${assistant.content}`);
    }
  }
  if (!turns.length) return message;

  return `${turns.join("\n\n")}\n\nUser: ${message}`;
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

  const fullMessage = buildPromptWithHistory(message, history);

  try {
    const client = await getClient();

    const result = await client.predict("/respond", {
      message: fullMessage,
    });

    // The Python client returns the raw value directly; the JS client
    // wraps it in { data: [...] }. Handle both shapes defensively.
    let reply = Array.isArray(result?.data) ? result.data[0] : result?.data;
    if (reply === undefined) reply = result;

    if (!reply) {
      throw new Error("Empty response from model. Space may still be loading.");
    }

    res.status(200).json({ reply });
  } catch (err) {
    console.error("HF Space error:", err);
    res.status(503).json({ error: err?.message || "Space unreachable" });
  }
}
