# Huberman Lab Podcast RAG Q&A Bot Design

## 1. Goals & Constraints
- Answer user questions using >300 Huberman Lab podcast transcripts.
- Entire pipeline runs locally on a 2019 Intel-based MacBook Pro (≤16GB RAM) with no network calls once models are downloaded.
- End-to-end latency ≤8s for short questions.
- Maintainability and extensibility: dropping new transcript files should require no code changes.

## 2. High-Level Architecture
```
transcripts/ --(ingest + normalise)--> cleaned text --(chunk)--> chunks
                                            |                         \
                                            v                          \
                                          metadata                      ---> embed --> embeddings + metadata
                                                                               |
                                                                               v
                                                         vector_db/index.faiss (FAISS IndexFlatIP)
```
At query time:
```
user query --> embed --> search FAISS --> top-k chunks --> prompt template --> local LLM --> answer
```
Modules communicate through well-defined functions to allow testing and reuse.

## 3. Module Designs & Pseudocode

### 3.1 `ingest.py`
**Responsibility:** Walk `transcripts/`, load `.txt` files, clean text, call `chunker` and `embed`, write out artifacts.
**Dependencies:** `pathlib`, `tqdm`, `faiss-cpu`, `numpy`, `psutil` (for resource checks), local modules `chunker`, `embed`.
**Pseudocode:**
```python
for path in transcripts_dir.rglob("*.txt"):
    raw = path.read_text()
    cleaned = normalise(raw)
    for chunk in chunk_text(cleaned):
        emb = embed_chunk(chunk.text)
        save_chunk_metadata(episode_id, chunk, emb)
```

### 3.2 Normalisation (`ingest.normalise`)
- Convert to UTF-8.
- Remove timestamps (`00:12:34`), sponsor blocks, repeated whitespace, non-speech artefacts.

```python
def normalise(raw: str) -> str:
    text = to_utf8(raw)
    text = remove_timestamps(text)
    text = strip_ads(text)
    text = collapse_whitespace(text)
    return text
```

### 3.3 `chunker.py`
- Split text into ~700-token segments with 20% overlap.
- Track `episode_id`, `chunk_id`, `start_token`, `end_token`.

```python
TOKENS = 700
OVERLAP = int(TOKENS * 0.2)

def chunk_text(text: str) -> list[Chunk]:
    tokens = tokenize(text)
    i = 0
    while i < len(tokens):
        window = tokens[i : i + TOKENS]
        yield Chunk(id=len(chunks), start=i, end=i+len(window), text=detokenize(window))
        i += TOKENS - OVERLAP
```

### 3.4 `embed.py`
- Generate embeddings locally using a GGUF/GGML model via `llama-cpp-python` or `ctransformers`.
- Support 4-bit QLoRA compression.

```python
init_embedding_model(path: str)

def embed_chunk(text: str) -> np.ndarray:
    return model.embed(text)

def embed_query(text: str) -> np.ndarray:
    return model.embed(text)
```

### 3.5 `build_index.py`
- Read stored chunk embeddings + metadata.
- Build FAISS `IndexFlatIP` and memory-map to `vector_db/index.faiss`.

```python
embeddings, metadata = load_embeddings()
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
faiss.write_index(index, index_path)
write_metadata(metadata, meta_path)
```

### 3.6 `local_llm.py` and `serve_llama.sh`
- Wrapper around llama.cpp or `ollama` runtime.
- Exposes `generate(prompt: str, max_tokens: int) -> str`.
- `serve_llama.sh` downloads weights and starts server.

```bash
# serve_llama.sh
if [ ! -f models/model.gguf ]; then
  download_model
fi
llama.cpp --model models/model.gguf --port 11435
```

```python
class LocalLLM:
    def __enter__(self):
        start_server_if_needed()
    def generate(self, prompt: str) -> str:
        return call_llama_cpp(prompt)
```

### 3.7 `rag_utils.py` and `answer.py`
- `rag_utils.py` houses retrieval and prompt assembly.

```python
PROMPT_TEMPLATE = """{system}\n{context}\nUser: {user}\nAssistant:"""

def retrieve(query: str, k: int = 5) -> list[Chunk]:
    q_emb = embed_query(query)
    D, I = index.search(q_emb, k)
    return [metadata[i] for i in I]


def compose_prompt(query: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(c.text for c in chunks)
    return PROMPT_TEMPLATE.format(system=SYSTEM_PROMPT, context=context, user=query)
```

- `answer.py` orchestrates query → retrieve → generate → format citations.

```python
def answer(question: str, k: int = 5) -> dict:
    chunks = retrieve(question, k)
    prompt = compose_prompt(question, chunks)
    output = llm.generate(prompt)
    return {"answer": output, "sources": [(c.episode_id, c.start_token) for c in chunks]}
```

### 3.8 CLI (`main.py`/`llama_query.py`)
- Argument parsing (`--question`, `--k`).
- Calls `answer()` and prints formatted response with provenance.

```python
if __name__ == "__main__":
    args = parse_args()
    result = answer(args.question, args.k)
    print(result["answer"])  # followed by episode IDs
```

### 3.9 Web UI (`app.py`, `templates/index.html`, `static/`)
- Flask app serving a single-page HTMX + Tailwind interface.
- `/` serves page; `/ask` accepts POST question, streams response tokens.

```python
@app.post("/ask")
def ask():
    q = request.form["question"]
    for token in stream_answer(q):
        yield token
```

HTML uses Tailwind for responsive design and dark-mode toggle, with citations shown beneath answers.

### 3.10 Logging & Metrics
- Use `logging` with `TimedRotatingFileHandler` writing to `logs/`.
- Log query text, latency, token counts, retrieved chunk IDs.

```python
start = time.time()
result = answer(...)
log.info({"query": q, "latency": time.time()-start, "chunks": result["sources"]})
```

### 3.11 Resource Safeguards
- Check RAM usage via `psutil`; if >85% switch to smaller model or reduce batch size.

```python
if psutil.virtual_memory().percent > 85:
    select_small_model()
```

### 3.12 Tests (`tests/`)
- PyTest suite covering:
  - Normalisation edge cases.
  - Chunk overlap boundaries.
  - Retrieval correctness using small fixture transcripts.
  - CLI regression: invoking `main.py` with sample question.

### 3.13 Makefile & Requirements
- `make setup` → install Python deps, download models, build `llama.cpp`.
- `make run` → start web server and LLM service.

```
setup:
pip install -r requirements.txt
download_models.sh
run:
python app.py
```

## 4. Dependency Rationale
- **faiss-cpu**: efficient similarity search without GPU.
- **llama-cpp-python / ctransformers**: CPU-friendly inference and embeddings.
- **Flask + HTMX + Tailwind**: minimal, mobile-friendly web stack.
- **psutil**: monitor system resources for safeguards.
- **pytest**: testing framework.

## 5. End-to-End Pseudocode
```python
# Build phase
ingest()        # produces cleaned chunks + embeddings
build_index()   # writes FAISS index

# Query phase
def query_loop():
    while True:
        q = input("Question: ")
        chunks = retrieve(q)
        prompt = compose_prompt(q, chunks)
        answer = llm.generate(prompt)
        print(answer)
```

## 6. Future Extensions
- Add GPU support when available.
- Support multi-lingual transcripts.
- Implement web-based administration dashboard for monitoring.

