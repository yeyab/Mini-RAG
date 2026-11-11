# 🔎 RAG with ChromaDB & Ollama

A minimal, fully local **Retrieval-Augmented Generation (RAG)** pipeline in Python.
Documents are embedded and stored in **[ChromaDB](https://www.trychroma.com/)**, the most relevant passages are retrieved for each question, and a local **LLM served by [Ollama](https://ollama.com/)** answers using *only* that retrieved context.

No API key, no cloud: everything runs on your own machine. The LLM layer is easy to swap for any other provider (OpenAI-compatible servers, LM Studio, vLLM, etc.).

---

## ✨ Features

- **Persistent vector store** – ChromaDB saves the index to disk (`./ma_base`), so documents are only embedded once.
- **Local embeddings** – `qwen3-embedding:4b` via Ollama.
- **Local generation** – `qwen2.5-coder:7b` via Ollama (any chat model works).
- **Cosine similarity search** with HNSW indexing.
- **Instruction-aware queries** – questions are prefixed with a retrieval instruction, as recommended for Qwen3 embedding models.
- **Grounded answers** – the system prompt forces the model to answer from the retrieved context only, reducing hallucinations.
- **Transparent retrieval** – retrieved passages are printed with their similarity score.

---

## 🧠 How it works

```
            ┌──────────────┐   embed    ┌───────────────────┐
 Documents ─►  Ollama      ├───────────►  ChromaDB          │
            │  embeddings  │            │  (persistent,     │
            └──────────────┘            │   cosine / HNSW)  │
                                        └─────────┬─────────┘
                                                  │ top-k passages
            ┌──────────────┐   embed    ┌─────────▼─────────┐
 Question ──►  Ollama      ├───────────►  Similarity search │
            │  embeddings  │            └─────────┬─────────┘
            └──────────────┘                      │ context
                                        ┌─────────▼─────────┐
                                        │  LLM (Ollama)     │──► Answer
                                        │  context + question│
                                        └───────────────────┘
```

1. **Indexing** – on first run, each document is embedded and stored in the `cours_ml` collection.
2. **Retrieval** – the question is embedded (with an instruction prefix) and the `n_results` closest passages are returned.
3. **Generation** – the passages are injected into the prompt and the LLM produces the answer.

---

## 📋 Requirements

- **Python 3.11+** (developed with 3.11.9)
- **[Ollama](https://ollama.com/download)** installed and running
- Enough RAM/VRAM for the chosen models (~8 GB recommended for the defaults)

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/yeyab/rag-chroma.git
cd rag-chroma

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the models with Ollama
ollama pull qwen3-embedding:4b
ollama pull qwen2.5-coder:7b
```

---

## ▶️ Usage

Make sure Ollama is running (`ollama serve`, or the Ollama desktop app), then:

```bash
python test_rag.py
```

Example session (the demo knowledge base and prompts are in French):

```
Base indexée : 4 documents
Ta question : Comment corriger l'overfitting ?

Passages trouvés :
  - (0.78) L'overfitting se corrige avec plus de données variées et moins d'époques.
  - (0.41) Un batch est un groupe d'échantillons traités avant chaque correction des poids.

Réponse : Pour corriger l'overfitting, il faut utiliser plus de données variées et réduire le nombre d'époques.
```

The number in brackets is the cosine similarity (`1 - distance`); higher means more relevant.

> Scores above are illustrative and depend on the models used.

---

## ⚙️ Configuration

All settings are at the top of `test_rag.py`:

| Variable / parameter | Default | Description |
|---|---|---|
| `EMBED` | `qwen3-embedding:4b` | Ollama embedding model |
| `LLM` | `qwen2.5-coder:7b` | Ollama chat model used to generate the answer |
| `CONSIGNE` | `Instruct: … Query: ` | Instruction prefix added to queries (not to documents) |
| `path` | `./ma_base` | Folder where ChromaDB persists the index |
| collection name | `cours_ml` | ChromaDB collection |
| `hnsw:space` | `cosine` | Distance metric |
| `n_results` | `2` | Number of passages passed to the LLM |

### Adding your own documents

Edit the `docs` list in `test_rag.py`. Because documents are only indexed when the collection is empty, **delete the `ma_base/` folder** after changing them (or after changing the embedding model) so the index is rebuilt:

```bash
# Windows
rmdir /s /q ma_base
# macOS / Linux
rm -rf ma_base
```

> ⚠️ Always use the same embedding model for indexing and querying – vectors from different models are not comparable.

---

## 🔌 Using another LLM

Only the generation step depends on Ollama's chat API, so swapping the LLM is a small change.

**Another Ollama model** – just change the variable:

```python
LLM = "llama3.1:8b"   # or mistral, gemma3, qwen2.5:14b, ...
```

**Any OpenAI-compatible API** (OpenAI, LM Studio, vLLM, llama.cpp server, Groq, …) – `pip install openai`, then replace the `ollama.chat(...)` call:

```python
from openai import OpenAI

llm = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")  # e.g. LM Studio
rep = llm.chat.completions.create(
    model="your-model-name",
    messages=[
        {"role": "system", "content": "Answer only from the provided context."},
        {"role": "user", "content": f"Context:\n{contexte}\n\nQuestion: {question}"},
    ],
)
print(rep.choices[0].message.content)
```

Embeddings can be swapped the same way, as long as the `embed()` function returns a list of vectors – remember to rebuild `ma_base/` afterwards.

---

## 📁 Project structure

```
rag-chroma/
├── test_rag.py        # Full RAG pipeline: indexing, retrieval, generation
├── requirements.txt   # Python dependencies
├── .gitignore
└── ma_base/           # ChromaDB persistent storage (generated, git-ignored)
```

---

## 🗺️ Roadmap

- [ ] Load documents from files (`.txt`, `.md`, `.pdf`) instead of a hard-coded list
- [ ] Text chunking with overlap for long documents
- [ ] Metadata (source, page) and source citations in answers
- [ ] Interactive loop / chat history
- [ ] Simple web UI (Streamlit or Gradio)
- [ ] Configuration via `.env` or CLI arguments

---

## 🛠️ Built with

- [ChromaDB](https://github.com/chroma-core/chroma) – open-source vector database
- [Ollama](https://github.com/ollama/ollama) & [ollama-python](https://github.com/ollama/ollama-python) – local LLM runtime
- [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-4B) and [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) models

---

## 👤 Author

**Nolan Horner** – [GitHub @yeyab](https://github.com/yeyab) · [LinkedIn](https://www.linkedin.com/in/nolan-horner-114163339/)

---

## 📄 License

This project is released under the MIT License – see the `LICENSE` file for details.
