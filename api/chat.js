import { Client } from "@gradio/client";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { message, history = [] } = req.body;

  try {
    const client = await Client.connect(
      "Deepti-singh-196/LawAssit_Version1_RAG",
      { hf_token: process.env.HF_TOKEN }
    );

    const result = await client.predict("/respond", {
      message,
      history,
    });

    return res.status(200).json({ reply: result.data[0] });

  } catch (error) {
    console.error("HF error:", error);
    return res.status(500).json({ error: error.message || "Model error" });
  }
}
