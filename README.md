# RAG Document Q&A with LangChain

A hands-on learning project completed as part of a **5-day guided Generative AI training program**. The program covered the LangChain ecosystem from basic LLM API calls through a full Retrieval-Augmented Generation (RAG) pipeline with source attribution.

> **Note:** This is a structured training project, not an independent build. All notebooks were developed during instructor-guided sessions and document my learning progression through each concept.

---

## What This Covers

| Day | Notebook | Concepts Practised |
|-----|----------|--------------------|
| Day 1 | `day01_langchain_gemini_setup.ipynb` | LangChain setup, Gemini 2.5 Flash API calls, basic prompt-response flow |
| Day 2 | `day02_stateful_chatbot_gradio.ipynb` | Chat history with `SystemMessage` / `HumanMessage` / `AIMessage`, `ChatPromptTemplate` with domain variables, Gradio UI |
| Day 3 | `day03_pdf_loading_groq_llm.ipynb` | PDF ingestion with `PyPDFLoader`, metadata inspection, passing document content to Groq LLM |
| Day 4a | `day04a_text_splitting_chunking.ipynb` | `RecursiveCharacterTextSplitter` (chunk sizes 100 and 1000, overlap 0), summarisation and key-date extraction prompts |
| Day 4b | `day04b_wikipedia_chromadb_embeddings.ipynb` | `WikipediaRetriever`, ChromaDB vector store, `GoogleGenerativeAIEmbeddings` (text-embedding-004), similarity search |
| Day 5 | `day05_full_rag_pipeline_retrieval_qa.ipynb` | End-to-end RAG — custom `Document` objects → ChromaDB (gemini-embedding-001) → `as_retriever(k=2)` → Groq LLM (`gpt-oss-20b`) → `RetrievalQA` chain with source document attribution |

---

## RAG Architecture (Day 5)

```
Custom Documents
      │
      ▼
GoogleGenerativeAIEmbeddings        ← gemini-embedding-001
      │
      ▼
ChromaDB Vector Store               ← similarity search
      │
      ▼
as_retriever(search_kwargs={"k": 2})
      │
      ▼
RetrievalQA Chain                   ← return_source_documents=True
      │
      ▼
Groq LLM (gpt-oss-20b)             ← answer generation
      │
      ▼
Answer + Source Attribution
```

---

## Tech Stack

- **Orchestration:** LangChain
- **LLMs:** Google Gemini 2.5 Flash (via `langchain-google-genai`), Groq (`gpt-oss-20b`)
- **Embeddings:** Google Generative AI Embeddings (`text-embedding-004`, `gemini-embedding-001`)
- **Vector Store:** ChromaDB
- **Document Loaders:** `PyPDFLoader`, `WikipediaRetriever`, custom `Document` objects
- **Text Splitting:** `RecursiveCharacterTextSplitter`
- **UI:** Gradio (Day 2 experiment)
- **Environment:** Google Colab

---

## Setup

### Prerequisites
- Google Colab (recommended) or local Python 3.9+
- API keys: [Google AI Studio](https://aistudio.google.com/), [Groq Console](https://console.groq.com/)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure API keys

In each notebook, set your keys at the top cell:

```python
import os
os.environ["GOOGLE_API_KEY"] = "your-google-api-key"
os.environ["GROQ_API_KEY"] = "your-groq-api-key"
```

---

## Running the Notebooks

Open notebooks in order (Day 1 → Day 5). Each notebook is self-contained — you can run them independently, though the concepts build progressively.

**Recommended:** Open in Google Colab for zero local setup.

---

## Key Learnings

- How LangChain abstracts LLM API calls and chains prompt → model → output parser
- Maintaining conversation state using message history objects
- The document ingestion and chunking pipeline (load → split → embed → store)
- How vector similarity search works as a retriever
- Why RAG improves factual grounding compared to zero-shot prompting
- Source attribution — knowing *which* document chunks the LLM used to answer

---

## Acknowledgement

A hands-on learning project completed as part of a 5-day guided Generative AI training program (2026).

---

## Author

**Krishna K M**  
BCA Graduate 2026 — AI, ML, Robotics & IoT  
[LinkedIn](https://www.linkedin.com/in/krishna-k-m) · [GitHub](https://github.com/Krishnamadhu325)
