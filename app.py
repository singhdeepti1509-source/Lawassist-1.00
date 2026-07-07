import os
import json
import pickle
import torch
import faiss
import numpy as np
import gradio as gr
import spaces
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
TOP_K_FUSED  = 20   # candidates kept after RRF fusion, before cross-encoder rerank
TOP_K_RERANK = 5
RRF_K        = 60   # standard RRF damping constant

if not HF_TOKEN:
    raise ValueError("HF_TOKEN secret is missing. Add it in Space settings.")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Running on: {device}")

# ── Keep-alive ping (prevents Space from sleeping) ────────────────────────
# NOTE: this just pings the Space's HTTP endpoint, not the @spaces.GPU
# function itself — so it keeps the container warm without burning ZeroGPU
# quota on an actual GPU allocation.
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
# NOTE: on ZeroGPU Spaces, no physical GPU is attached at module-import time —
# only inside an active call to a @spaces.GPU-decorated function. Loading
# straight onto "cuda" here would crash with "No CUDA GPUs are available".
# Load on CPU now; move to the real device lazily inside respond().
print("Loading embedder (BGE-large)...")
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

print("Loading cross-encoder...")
cross_encoder = CrossEncoder(CE_MODEL, device="cpu")

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
    dtype=dtype,
    low_cpu_mem_usage=True,
)

print("Loading LoRA adapter...")
# NOTE: peft's load_adapter() calls its own infer_device() internally if not
# told otherwise, which just checks torch.cuda.is_available(). On ZeroGPU
# that returns True even before any physical GPU is attached, so peft tries
# to load the adapter weights straight onto "cuda" and crashes with
# "No CUDA GPUs are available" — independent of where base_model itself
# lives. Force CPU explicitly here; the whole model gets moved to the real
# device later, inside respond(), where a GPU is actually guaranteed.
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_REPO,
    token=HF_TOKEN,
    is_trainable=False,
    torch_device="cpu",
)

# Sanity check: catch tokenizer/embedding-matrix mismatches early instead of
# generating silently garbled output at inference time.
vocab_size = model.get_input_embeddings().weight.shape[0]
if vocab_size != len(tokenizer):
    print(
        f"WARNING: model embedding size ({vocab_size}) != tokenizer vocab size "
        f"({len(tokenizer)}). Generation quality may be degraded."
    )

model.eval()
print("All models loaded on CPU; will move to GPU on first request (ZeroGPU).")

# Tracks whether models have been moved to the real device yet. Must only
# happen inside a @spaces.GPU-decorated call, where a physical GPU is
# actually attached — never at module scope.
_moved_to_gpu = False


def _ensure_on_device():
    """Move all models to `device` exactly once, on first real request."""
    global _moved_to_gpu
    if _moved_to_gpu or device != "cuda":
        return
    model.to(device)
    embedder.to(device)
    cross_encoder.model.to(device)
    _moved_to_gpu = True


# ── RAG Retrieval ─────────────────────────────────────────────────────────
def rrf_fuse(dense_ids, bm25_ids, k: int = RRF_K, top_n: int = TOP_K_FUSED) -> list:
    """Reciprocal Rank Fusion over two ranked id lists (higher score = more relevant)."""
    scores = {}
    for rank, idx in enumerate(dense_ids):
        if idx < 0:
            continue
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(bm25_ids):
        if idx < 0:
            continue
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [chunks[i] for i in ranked_ids if 0 <= i < len(chunks)]


def retrieve(query: str, top_k_dense: int = TOP_K_DENSE, top_k_final: int = TOP_K_RERANK) -> list:
    # 1. Dense retrieval via FAISS
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    _, dense_ids = faiss_index.search(q_emb, top_k_dense)
    dense_ids = list(dense_ids[0])

    # 2. BM25 sparse retrieval
    # NOTE: this must match whatever tokenization was used to build bm25.pkl
    # (lowercase + whitespace split here). If the index was built with a
    # different tokenizer, update this to match or BM25 scores will be
    # unreliable.
    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    bm25_ids = list(np.argsort(bm25_scores)[::-1][:top_k_dense])

    # 3. Reciprocal Rank Fusion (deduplicated by construction, since we fuse by id)
    fused_candidates = rrf_fuse(dense_ids, bm25_ids, k=RRF_K, top_n=top_k_dense)

    if not fused_candidates:
        return []

    # 4. Cross-encoder re-ranking
    pairs = [(query, c["text"]) for c in fused_candidates]
    ce_scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(ce_scores, fused_candidates), key=lambda x: -x[0])

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
# @spaces.GPU is required on ZeroGPU Spaces: it tells the Space which
# function should be allocated a GPU at call time. Without at least one
# decorated function, ZeroGPU Spaces fail at startup with
# "No @spaces.GPU function detected".
@spaces.GPU(duration=60)
def respond(message: str, history: list) -> str:
    _ensure_on_device()

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
        "Powered by Hybrid KG-RAG (FAISS + BM25 + RRF fusion + Cross-Encoder) and LawAssist-1B."
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
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        show_error=True,
    )
