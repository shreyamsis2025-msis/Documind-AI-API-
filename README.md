# Documind AI

**Chat with your documents using natural language — powered by Groq (free & fast)**

Upload PDFs, Word files, CSVs, Excel sheets and ask questions about them. Answers stream in real time with source citations.

---

## ⚡ Quick Start

### 1. Get a FREE Groq API Key
1. Go to **[console.groq.com](https://console.groq.com)**
2. Sign up (no credit card required)
3. Click **API Keys → Create API Key**
4. Copy the key (starts with `gsk_...`)

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API key (choose one way)

**Option A — Environment variable (recommended):**
```bash
export GROQ_API_KEY="gsk_your_key_here"
streamlit run app.py
```

**Option B — In the app sidebar:**
Paste your key into the 🔑 field in the sidebar when the app starts.

**Option C — In config.py directly:**
```python
GROQ_API_KEY = "gsk_your_key_here"
```

---

## 📁 Supported File Types
| Format | Extension |
|--------|-----------|
| PDF | `.pdf` |
| Word | `.docx` |
| CSV | `.csv` |
| Excel | `.xlsx`, `.xls` |
| Text | `.txt` |

---

## 🏗️ Architecture

```
Upload → Load → Split into chunks → Embed (local, free)
                                          ↓
                                    FAISS vector index
                                          ↓
Query → Retrieve relevant chunks → Groq LLM → Streaming answer + Sources
```

- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 — runs locally, no API needed
- **LLM:** Groq llama-3.1-8b-instant — free, ~500 tokens/sec
- **Vector DB:** FAISS (in-memory + saved to disk)

---

## 🔧 Customise model in config.py

```python
GROQ_MODEL = "llama-3.1-8b-instant"      # fastest (default)
# GROQ_MODEL = "llama-3.3-70b-versatile"  # more powerful
# GROQ_MODEL = "gemma2-9b-it"             # Google Gemma
# GROQ_MODEL = "mixtral-8x7b-32768"       # large context window
```

---

<img width="1901" height="856" alt="image" src="assets\image.png" />

