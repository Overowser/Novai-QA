# Novai QA: Webnovel Question Answering System

**Novai QA** is a Retrieval-Augmented Generation (RAG) system designed to help readers of long webnovels get spoiler-free, context-aware answers about characters, events, and plot points. The system scrapes novels from [novelfull.com](https://novelfull.com), processes them into searchable chunks, and provides intelligent responses using local LLMs.

---

## Overview

This system provides:
- **Smart Web Scraping**: Automated novel content extraction from novelfull.com
- **Intelligent Chunking**: Paragraph-based text segmentation with overlap for context preservation
- **Hybrid Retrieval**: Combines semantic search (ChromaDB) with keyword-based search (BM25)
- **Spoiler Protection**: Filters responses based on user's reading progress
- **Local LLM Integration**: Uses Ollama for response generation (currently DeepSeek-R1:7b)
- **Interactive Web Interface**: Gradio-based chat interface for user interaction

---

## System Architecture

### Data Flow Pipeline

```
Web Scraping → Database Storage → Text Chunking → Embedding/Indexing → Retrieval → Response Generation
```

### 1. Data Collection & Storage
- **Source**: novelfull.com via automated scraper
- **Database**: PostgreSQL with three main tables:
  - `novels`: Novel metadata (title, image)
  - `chapters`: Chapter content and metadata
  - `chunks`: Processed text segments with preprocessing
- **Content Processing**: Handles Cloudflare protection and content cleaning

### 2. Text Processing & Chunking
- **Segmentation Strategy**: Paragraph-based with intelligent fallback to sentence-level
- **Chunk Configuration**: 512 tokens max size, 200 token overlap
- **Adaptive Processing**: Handles varying paragraph sizes automatically
- **Context Preservation**: Overlap ensures continuity across chunk boundaries

### 3. Dual Indexing System
- **Semantic Search**: 
  - Model: `mxbai-embed-large-v1` 
  - Storage: ChromaDB with normalized embeddings
  - CUDA acceleration for encoding
- **Keyword Search**:
  - Algorithm: BM25 with NLTK preprocessing
  - Features: Lemmatization, stopword removal, POS tagging

### 4. Intelligent Retrieval
- **Hybrid Approach**: Combines semantic and keyword-based results
- **Spoiler Filtering**: Chapter-based content filtering
- **Query Processing**: Multiple retrieval strategies merged for comprehensive results
- **Reranking Pipeline**: (Currently basic, expandable for advanced reranking)

### 5. Response Generation
- **LLM Backend**: Ollama integration (DeepSeek-R1:7b)
- **Grounded Responses**: Strict adherence to retrieved content
- **Natural Language**: Contextual answers without system prompt exposure
- **Reasoning Chain Handling**: Processes models with thinking tokens

---

## Getting Started

### Prerequisites
- Python 3.12.0
- PostgreSQL database
- CUDA-compatible GPU (recommended)
- Ollama installed with DeepSeek-R1:7b model

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd novai-qa
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file with:
```env
PG_HOST=your_postgres_host
PG_DB=your_database_name
PG_USER=your_username
PG_PASSWORD=your_password
```

4. **Initialize the database**
```bash
python App/1_setup.py
```

5. **Prepare your first novel**
```bash
python App/2_prepare_novel.py
```

6. **Launch the application**
```bash
python App/3_app.py
```

---

## Usage Workflow

### 1. Novel Preparation
Run the preparation script and follow the interactive prompts:
- Enter search keywords for your desired novel
- Confirm the correct novel selection
- Wait for scraping, chunking, and indexing to complete

### 2. Querying the System
Access the Gradio interface and:
- Enter the novel name
- Set your current chapter (optional, for spoiler protection)
- Ask questions about characters, plot, or events

### 3. Example Queries
- "Who is [character name]?"
- "What happened in [specific event]?"
- "Describe [character]'s abilities"
- "What is the relationship between [character A] and [character B]?"

---

## Technical Implementation Details

### Database Schema
```sql
-- Core novel metadata
CREATE TABLE novels (
    id SERIAL PRIMARY KEY,
    novel_title TEXT NOT NULL UNIQUE,
    novel_image TEXT
);

-- Chapter storage with content
CREATE TABLE chapters (
    id SERIAL PRIMARY KEY,
    novel_id INTEGER REFERENCES novels(id),
    chapter_number INT NOT NULL,
    chapter_title TEXT,
    chapter_url TEXT,
    chapter_content TEXT
);

-- Processed text chunks
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    chapter_id INTEGER REFERENCES chapters(id),
    novel_id INTEGER REFERENCES novels(id),
    chunk_number INT NOT NULL,
    chunk_content TEXT,
    preprocessed_chunk_content TEXT[]  -- Tokenized for BM25
);
```

### Chunking Algorithm
- **Primary**: Paragraph-level segmentation
- **Fallback**: Sentence-level for oversized paragraphs
- **Overlap Strategy**: Contextual bridging between chunks
- **Token Counting**: Precise tokenization using SentenceTransformer tokenizer

### Retrieval Strategy
- **ChromaDB Query**: Semantic similarity with cosine distance
- **BM25 Search**: Keyword matching with preprocessing pipeline
- **Result Fusion**: Combines and deduplicates results from both methods
- **Spoiler Filter**: Applied at database query level for efficiency

---

## Configuration Options

### Chunking Parameters
```python
max_chunk_size = 512      # Maximum tokens per chunk
overlap = 200             # Overlap tokens between chunks
embedding_model = "mixedbread-ai/mxbai-embed-large-v1"
```

### Retrieval Settings
```python
k = 10                    # Number of chunks to retrieve
chroma_batch_size = 1024  # Batch size for embedding
```

### Model Configuration
```python
llm_model = "deepseek-r1:7b"  # Ollama model for generation
```

---

## Performance Considerations

### Optimization Features
- **GPU Acceleration**: CUDA support for embedding generation
- **Batch Processing**: Efficient bulk operations for large novels
- **Connection Pooling**: Database connection management
- **Async Operations**: Non-blocking web scraping with aiohttp
- **Normalized Embeddings**: Faster similarity computations

### Scalability Notes
- **Memory Usage**: Embedding models require significant GPU memory
- **Storage**: Large novels generate thousands of chunks
- **Processing Time**: Initial preparation can take up to an hour on GPU and several hours on CPU for long novels

---

## System Capabilities

### Strengths
- **Spoiler-Free**: Intelligent content filtering based on reading progress
- **Comprehensive Coverage**: Handles very long novels (3000+ chapters)
- **Robust Retrieval**: Dual search strategy catches both semantic and exact matches
- **Local Processing**: No external API dependencies for core functionality

### Current Limitations
- **Single Novel Focus**: Designed for one novel at a time
- **Manual Preparation**: Requires user input for novel selection
- **Basic Reranking**: Currently shallow (just placeholder)
- **Slow Query preprocessing**: Slow NLTK preprocessing for BM25 during inference
- **No Chat History**: Stateless conversation model
- **Streaming output**: Currently not enabled (model sends full response at once)
- **Inference speed**: Tied to local ressources

---

## Development Setup

### Project Structure
```
App/
├── 1_setup.py          # Database initialization
├── 2_prepare_novel.py  # Novel preparation pipeline
├── 3_app.py           # Gradio web interface
├── scraper.py         # Web scraping functionality
├── chunker.py         # Text segmentation logic
├── indexer.py         # Embedding and BM25 indexing
├── retriever.py       # Hybrid retrieval system
├── generator.py       # LLM response generation
├── utils.py           # Database and utility functions
└── logger_config.py   # Logging configuration
```

### Key Dependencies
- **Web Interface**: `gradio`
- **Embeddings**: `sentence-transformers`
- **Vector Store**: `chromadb`
- **Search**: `rank-bm25`
- **Database**: `psycopg2`
- **LLM**: `ollama`
- **Scraping**: `aiohttp`, `beautifulsoup4`
- **NLP**: `nltk`

---

## Logging & Monitoring

The system includes comprehensive logging across all modules:
- **Scraping Progress**: Chapter download and processing status
- **Chunking Operations**: Text segmentation statistics
- **Indexing Progress**: Embedding generation and storage
- **Query Processing**: Retrieval and generation metrics
- **Error Handling**: Detailed error tracking and recovery

Log files are stored in `./logs/app.log` with module-specific prefixes.

---

## Future Enhancements

### Planned Features
- **Multi-Novel Support**: Handle multiple novels simultaneously
- **Reranking**: LLM-based relevance scoring 
- **Query Translation**: Rephrase user queries for improved retrieval
- **Chat Memory**: Maintain conversation context across interactions
- **Auto-Update**: Automatically update novel chapters via DAG
- **Enhanced UI**: Upgrade web interface for improved UX

### Technical Improvements
- **Reduced Retrieval Delay**: Minimize latency from query preprocessing and related steps 
- **Caching Layer**: Redis integration for faster responses
- **API Endpoints**: REST API for external integrations
- **Containerization**: Docker deployment configuration
- **Monitoring**: Performance metrics and health checks

---

## License & Disclaimer

This project is for educational and portfolio demonstration purposes. It is not intended for commercial use. Users should respect copyright laws and website terms of service when scraping content.

---

## Contributing

This is currently a personal portfolio project. Feel free to fork and adapt for your own use cases. Future collaboration opportunities may be considered.

---

## Support

For questions or feedback regarding **Novai QA**, open an issue on GitHub or contact [ben-ghali@hotmail.com](mailto:ben-ghali@hotmail.com).

For complete details about the **Novai** project, including frontend implementation and integration with **Novai QA**, visit [overowser.github.io](https://overowser.github.io/projects)