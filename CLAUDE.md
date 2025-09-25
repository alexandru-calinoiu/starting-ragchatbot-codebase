# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
```bash
# Quick start (uses run.sh script)
./run.sh

# Manual start
cd backend
uv run uvicorn app:app --reload --port 8000

# The application will be available at:
# - Web Interface: http://localhost:8000
# - API Documentation: http://localhost:8000/docs
```

### Dependency Management and Running Python Files
```bash
# Install dependencies
uv sync

# Add new dependency
uv add <package-name>

# Run any Python file or test with uv (always use this for Python execution)
uv run python <file.py>
uv run pytest <test_file.py>

# Run command in virtual environment
uv run <command>
```

### Environment Setup
```bash
# Copy example env file and add your Anthropic API key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your-key-here
```

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation) system** for course materials Q&A. The system follows a pipeline architecture:

### Request Flow
1. **User Query** → Frontend (http://localhost:8000)
2. **API Request** → FastAPI endpoint (`/api/query`)
3. **RAG Pipeline**:
   - `RAGSystem.query()` orchestrates the entire process
   - `VectorStore` performs semantic search in ChromaDB
   - Retrieved chunks provide context
   - `AIGenerator` calls Anthropic Claude with context + query
   - `SessionManager` maintains conversation history
4. **Response** → Returns answer + sources to frontend

### Component Interactions

**RAGSystem** (`backend/rag_system.py`) is the main orchestrator that coordinates:
- **DocumentProcessor**: Chunks course documents (800 chars with 100 char overlap)
- **VectorStore**: ChromaDB wrapper using sentence-transformers embeddings
- **AIGenerator**: Anthropic Claude integration (claude-sonnet-4-20250514)
- **SessionManager**: Tracks conversation history (last 2 messages)
- **ToolManager/CourseSearchTool**: Additional search capabilities

### Data Flow for Document Processing
1. Course documents (`.txt` files) in `docs/` are loaded on startup
2. `DocumentProcessor.process_course_document()` extracts metadata and creates chunks
3. `VectorStore.add_course_content()` stores embeddings in ChromaDB
4. ChromaDB persists to `backend/chroma_db/`

### Key Configuration
All settings in `backend/config.py`:
- `CHUNK_SIZE`: 800 characters
- `CHUNK_OVERLAP`: 100 characters
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation turns
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2

### API Endpoints
- `POST /api/query`: Main query endpoint (expects `query` and optional `session_id`)
- `GET /api/courses`: Returns course statistics
- Static files served from `frontend/` at root path

### Database
- **ChromaDB** stored in `backend/chroma_db/` (gitignored)
- Automatically initializes on first run
- Embeddings use sentence-transformers model

### Frontend
- Single-page application (vanilla JS)
- Files: `frontend/index.html`, `script.js`, `style.css`
- No build step required

# Individual Preferences
- @local-notes.md