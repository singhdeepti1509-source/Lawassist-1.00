import os
import json
import pickle
import torch
import faiss
import numpy as np
import gradio as gr
import threading
import requests
import time
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from peft import PeftModel

# ── Config ───────────────────────────────────────────────────────────────
BASE_MODEL   = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_REPO = "Deepti-singh-196/Lawassit-v1"
RAG_REPO     = "Deepti-singh-196/indian-judiciary-rag-indexes"
HF_TOKEN     = os.getenv("HF_TOKEN")
SPACE_URL    = "https://deepti-singh-196-lawassit-version1-rag.hf.space"

EMBED_MODEL  = "BAAI/bge-large-en-v1.5"
CE_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_DENSE  = 20
TOP_K_RERANK = 5

if not HF_TOKEN:
    raise ValueError("HF_TOKEN secret is missing. Add it in Space settings.")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Running on: {device}")

# ── Keep-alive ping (prevents Space from sleeping) ────────────────────────
def keep_alive():
    while True:
        time.sleep(600)  # ping every 10 minutes
        try:
            requests.get(SPACE_URL, timeout=10)
            print("[keep-alive] Space pinged successfully.")
        except Exception as e:
            print(f"[keep-alive] Ping failed: {e}")

threading.Thread(target=keep_alive, daemon=True).start()
print("Keep-alive thread started.")

# ── Download RAG artifacts from HF Hub ───────────────────────────────────
print("Downloading RAG artifacts...")
rag_dir = Path(snapshot_download(
    repo_id=RAG_REPO,
    repo_type="dataset",
    token=HF_TOKEN
))
print(f"RAG artifacts downloaded to: {rag_dir}")

# Load FAISS index
print("Loading FAISS index...")
faiss_index = faiss.read_index(str(rag_dir / "faiss_hnsw.index"))

# Load BM25 index
print("Loading BM25 index...")
with open(rag_dir / "bm25.pkl", "rb") as f:
    bm25: BM25Okapi = pickle.load(f)

# Load chunks (JSONL format)
print("Loading chunks...")
chunks = []
with open(rag_dir / "chunks.jsonl", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))
print(f"Loaded {len(chunks):,} chunks")
if chunks:
    print(f"Chunk keys: {list(chunks[0].keys())}")

# ── Load retrieval models ─────────────────────────────────────────────────
print("Loading embedder (BGE-large)...")
embedder = SentenceTransformer(EMBED_MODEL, device=device)

print("Loading cross-encoder...")
cross_encoder = CrossEncoder(CE_MODEL, device=device)

# ── Load LLM ─────────────────────────────────────────────────────────────
print("Loading tokenizer from adapter repo...")
tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_REPO,
    token=HF_TOKEN,
    use_fast=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_REPO,
    token=HF_TOKEN,
    is_trainable=False,
)
model = model.to(device)
model.eval()
print(f"All models loaded on {device}.")

# ── RAG Retrieval ─────────────────────────────────────────────────────────
def retrieve(query: str, top_k_dense: int = TOP_K_DENSE, top_k_final: int = TOP_K_RERANK) -> list:
    # 1. Dense retrieval via FAISS
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    _, dense_ids = faiss_index.search(q_emb, top_k_dense)
    dense_chunks = [chunks[i] for i in dense_ids[0] if 0 <= i < len(chunks)]

    # 2. BM25 sparse retrieval
    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    bm25_top_ids = np.argsort(bm25_scores)[::-1][:top_k_dense]
    bm25_chunks = [chunks[i] for i in bm25_top_ids if 0 <= i < len(chunks)]

    # 3. RRF fusion (deduplicated)
    seen, candidates = set(), []
    for chunk in dense_chunks + bm25_chunks:
        key = chunk.get("chunk_id") or chunk.get("id") or chunk["text"][:80]
        if key not in seen:
            seen.add(key)
            candidates.append(chunk)

    # 4. Cross-encoder re-ranking
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    ce_scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(ce_scores, candidates), key=lambda x: -x[0])

    return [c for _, c in ranked[:top_k_final]]


def build_context(retrieved: list) -> str:
    parts = []
    for i, chunk in enumerate(retrieved, 1):
        src   = chunk.get("source", chunk.get("dataset", "Unknown"))
        court = chunk.get("court", "")
        year  = chunk.get("year", "")
        header = f"[{i}] {src}"
        if court:
            header += f" | {court}"
        if year:
            header += f" ({year})"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


# ── Prompting ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are LawAssist, a helpful Indian legal AI assistant. "
    "Use the provided legal context to answer accurately. "
    "Cite source numbers like [1], [2] when referencing them. "
    "Do not invent laws or cases. If uncertain, say so."
)

def normalize_history(history):
    normalized = []
    for item in history or []:
        if isinstance(item, dict):
            role    = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and content is not None:
                normalized.append({"role": role, "content": str(content)})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            u, a = item
            if u: normalized.append({"role": "user",      "content": str(u)})
            if a: normalized.append({"role": "assistant",  "content": str(a)})
    return normalized


# ── Main chat function ────────────────────────────────────────────────────
def respond(message: str, history: list) -> str:
    try:
        retrieved = retrieve(message)
        context   = build_context(retrieved)
    except Exception as e:
        print(f"Retrieval error: {e}")
        retrieved, context = [], ""

    system_with_ctx = SYSTEM_PROMPT
    if context:
        system_with_ctx += f"\n\nRelevant legal context:\n{context}"

    messages = [{"role": "system", "content": system_with_ctx}]
    messages.extend(normalize_history(history))
    messages.append({"role": "user", "content": message})

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = system_with_ctx + "\n\n"
        for m in messages[1:]:
            prompt += f"{m['role'].capitalize()}: {m['content']}\n"
        prompt += "Assistant:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    if not reply:
        reply = "I could not generate a response. Please try again."

    if retrieved:
        sources = "\n\n📚 **Sources:**\n" + "\n".join(
            f"[{i+1}] {c.get('source', c.get('dataset', 'Unknown'))}"
            + (f" — {c.get('court', '')}" if c.get('court') else "")
            + (f" ({c.get('year', '')})" if c.get('year') else "")
            for i, c in enumerate(retrieved)
        )
        reply += sources

    return reply


# ── Gradio UI ─────────────────────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=respond,
    title="⚖️ LawAssist v1 — RAG Augmented",
    description=(
        "Ask questions about Indian law and Supreme Court judgments. "
        "Powered by Hybrid KG-RAG (FAISS + BM25 + Cross-Encoder) and LawAssist-1B."
    ),
    examples=[
        "What is the right to privacy under the Indian Constitution?",
        "Explain the basic structure doctrine.",
        "What are the grounds for bail under BNSS?",
        "What did the Supreme Court hold in Maneka Gandhi v. Union of India?",
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
