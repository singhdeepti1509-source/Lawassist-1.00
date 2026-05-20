export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { message } = req.body;

  try {
    const response = await fetch(
      "https://deepti-singh-196-lawassit-version1-rag.hf.space/run/predict",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${process.env.HF_TOKEN}`,
        },
        body: JSON.stringify({
          data: [message],
          fn_index: 0,
        }),
      }
    );

    const result = await response.json();
    return res.status(200).json({ reply: result.data[0] });

  } catch (error) {
    return res.status(500).json({ error: "Failed to reach model" });
  }
}
