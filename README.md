# MedRag - Medical RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for mental health FAQs, built with LangChain, LangGraph, and FastAPI.

## Features

- **Vector Search**: ChromaDB with sentence-transformers embeddings
- **LLM**: GPT-2 model for answer generation
- **API**: FastAPI endpoint for querying
- **Workflow**: LangGraph orchestration for retrieve-then-generate pipeline

## Prerequisites

- Python 3.8+
- 4GB+ RAM recommended
- GPU optional (CPU fallback supported)

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd MedRag

# Create virtual environment
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
MedRag/
├── rag_components/
│   └── content/
│       ├── app.py                    # FastAPI application
│       ├── chroma_db/                # Vector store (gitignored)
│       ├── gpt2_model/               # Fine-tuned model (gitignored)
│       ├── prompt_template.pkl       # Prompt configuration
│       └── vector_store_config.pkl   # ChromaDB config
├── Mental_Health_FAQ.csv             # Training data
├── your_cleaned_data.csv             # Processed data
└── requirements.txt
```

## Usage

### Start the API Server

```bash
cd rag_components/content
python app.py
```

Server runs at `http://localhost:8000`

### Query the API

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are symptoms of anxiety?"}'
```

**Response:**
```json
{
  "question": "What are symptoms of anxiety?",
  "answer": "..."
}
```

## Setup Requirements

Before running, ensure these files exist in `rag_components/content/`:
- `gpt2_model/` - Pre-trained/fine-tuned GPT-2 model
- `chroma_db/` - Initialized ChromaDB vector store
- `prompt_template.pkl` - Serialized prompt template
- `vector_store_config.pkl` - Vector store configuration

## Configuration

Edit `app.py` to customize:
- Model path: `./gpt2_model`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Max tokens: `max_new_tokens=200`
- Temperature: `temperature=0.7`
- Retrieval count: `k=3`

## API Endpoints

### POST `/query`
Query the RAG system with a question.

**Request Body:**
```json
{
  "question": "string"
}
```

**Response:**
```json
{
  "question": "string",
  "answer": "string"
}
```

## Dependencies

Core packages:
- `torch` - PyTorch for model inference
- `transformers` - Hugging Face transformers
- `langchain` - RAG framework
- `langgraph` - Workflow orchestration
- `chromadb` - Vector database
- `sentence-transformers` - Embeddings
- `fastapi` - API framework
- `uvicorn` - ASGI server

See `requirements.txt` for full list.

## License

MIT

