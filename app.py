import gradio as gr
import torch
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Configuration - Replace with your actual paths
BASE_MODEL = "meta-llama/Llama-3.2-3B"
ADAPTER_ID = "singhdeepti1509-source/your-fine-tuned-adapter" # Replace with your HF adapter ID

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Load Base Model in 4-bit to fit in free RAM (Optional but recommended)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load your Fine-Tuned weights
try:
    model = PeftModel.from_pretrained(model, ADAPTER_ID)
    print("Fine-tuned adapter loaded successfully!")
except Exception as e:
    print(f"Running base model only. Error loading adapter: {e}")

# 2. RAG Retrieval Logic (Placeholder)
def get_legal_context(query):
    # This is where you would call Pinecone or Qdrant
    # For now, it returns a placeholder for your 30+ research papers
    return "Relevant Indian Judicial Precedent regarding the query..."

# 3. Inference Function with ZeroGPU
@spaces.GPU
def chat_with_lawassist(message, history):
    # Get context for RAG
    context = get_legal_context(message)
    
    # Format the prompt for Llama-3.2
    # We combine the context and the user query
    prompt = f"Context: {context}\n\nUser: {message}\nAssistant:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's part
    response = full_response.split("Assistant:")[-1].strip()
    return response

# 4. Gradio 6.13.0 UI Setup
demo = gr.ChatInterface(
    fn=chat_with_lawassist,
    type="messages", # Required for Gradio 6.x
    title="⚖️ LawAssist Version 1.0 (RAG)",
    description="Indian Judicial AI assistant fine-tuned on Llama-3.2. Provides citation-aware summaries.",
    examples=["What are the grounds for divorce under the Hindu Marriage Act?", "Summarize recent RAG trends in legal NLP."],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
