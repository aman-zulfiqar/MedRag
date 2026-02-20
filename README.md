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

## Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd MedRag

# Create virtual environment
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup models and data (see Setup section below)
```

## Setup

### 1. Model Setup

Download or place your GPT-2 model in `rag_components/content/gpt2_model/`:

**Option A: Use base GPT-2**
```bash
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
model = AutoModelForCausalLM.from_pretrained('gpt2'); \
tokenizer = AutoTokenizer.from_pretrained('gpt2'); \
model.save_pretrained('./rag_components/content/gpt2_model'); \
tokenizer.save_pretrained('./rag_components/content/gpt2_model')"
```

**Option B: Use your fine-tuned model**
- Place your model files in `rag_components/content/gpt2_model/`
- Required files: `config.json`, `model.safetensors`, `tokenizer.json`, etc.

### 2. Data Setup

Place your FAQ data in the root directory:
- `Mental_Health_FAQ.csv` - Source FAQ data
- `your_cleaned_data.csv` - Preprocessed data

**Data format:**
```csv
question,answer
"What is anxiety?","Anxiety is..."
```

### 3. Vector Store Initialization

Create the ChromaDB vector store:

```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import pickle

# Load data
df = pd.read_csv("your_cleaned_data.csv")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
vector_store = Chroma.from_texts(
    texts=df['answer'].tolist(),
    embedding=embeddings,
    persist_directory="./rag_components/content/chroma_db"
)

# Save config
config = {"persist_directory": "./rag_components/content/chroma_db"}
with open("./rag_components/content/vector_store_config.pkl", "wb") as f:
    pickle.dump(config, f)
```

### 4. Prompt Template Setup

Create your prompt template:

```python
from langchain_core.prompts import ChatPromptTemplate
import pickle

prompt = ChatPromptTemplate.from_template(
    "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
)

with open("./rag_components/content/prompt_template.pkl", "wb") as f:
    pickle.dump(prompt, f)
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

## Configuration

### Model Parameters

Edit `rag_components/content/app.py`:

```python
# Model settings
model_path = "./gpt2_model"
torch_dtype = torch.float16  # Use float32 for CPU

# Generation settings
max_new_tokens = 200         # Max response length
temperature = 0.7            # Creativity (0.0-1.0)

# Retrieval settings
k = 3                        # Number of context documents
```

### Environment Variables

Create `.env` file (optional):
```bash
MODEL_PATH=./gpt2_model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
API_HOST=0.0.0.0
API_PORT=8000
```

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

## Troubleshooting

### Model Loading Issues
```bash
# If CUDA out of memory
# Edit app.py: torch_dtype=torch.float32, device_map="cpu"

# If model not found
# Verify path: ls rag_components/content/gpt2_model/
```

### Vector Store Issues
```bash
# Rebuild vector store
rm -rf rag_components/content/chroma_db
# Re-run vector store initialization script
```

### API Issues
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac

# Use different port
uvicorn app:app --port 8001
```

## Production Deployment

### Docker (Recommended)

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "rag_components.content.app:app", "--host", "0.0.0.0"]
```

Build and run:
```bash
docker build -t medrag .
docker run -p 8000:8000 medrag
```

### Environment Variables for Production
```bash
export MODEL_PATH=/path/to/model
export LOG_LEVEL=info
export WORKERS=4
```

## Performance Optimization

- **GPU**: Set `device_map="auto"` for automatic GPU usage
- **Batch Processing**: Implement batch inference for multiple queries
- **Caching**: Add Redis for response caching
- **Load Balancing**: Use nginx/gunicorn for multiple workers

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## License

MIT

