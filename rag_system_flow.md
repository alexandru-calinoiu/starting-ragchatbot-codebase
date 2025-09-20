# RAG Chatbot System Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Frontend<br/>(script.js)
    participant API as FastAPI<br/>(app.py)
    participant RAG as RAGSystem<br/>(rag_system.py)
    participant Tools as ToolManager<br/>(search_tools.py)
    participant Vector as VectorStore<br/>(ChromaDB)
    participant AI as AIGenerator<br/>(Claude API)
    participant Session as SessionManager

    Note over User, Session: User Query Flow

    User->>Frontend: Types query & clicks send
    Frontend->>Frontend: Show loading animation
    Frontend->>API: POST /api/query<br/>{"query": "...", "session_id": "..."}

    API->>API: Create session_id if null
    API->>RAG: rag_system.query(query, session_id)

    RAG->>Session: Get conversation history
    Session-->>RAG: Returns last 2 exchanges

    RAG->>AI: Generate response with tools
    Note over AI: Claude processes query with tool definitions

    AI->>Tools: Uses CourseSearchTool
    Tools->>Vector: Semantic search in ChromaDB
    Vector->>Vector: Query embeddings<br/>(sentence-transformers)
    Vector-->>Tools: Top 5 relevant chunks
    Tools-->>AI: Context + sources

    AI-->>RAG: Generated response
    RAG->>Tools: Get sources from last search
    Tools-->>RAG: Source list

    RAG->>Session: Add exchange to history
    RAG-->>API: (response, sources)

    API-->>Frontend: JSON response<br/>{"answer": "...", "sources": [...], "session_id": "..."}

    Frontend->>Frontend: Remove loading animation
    Frontend->>Frontend: Render markdown response
    Frontend->>Frontend: Display collapsible sources
    Frontend->>User: Show complete answer

    Note over User, Session: Component Architecture

    Note over Vector: ChromaDB Storage<br/>• Document chunks (800 chars)<br/>• Embeddings (all-MiniLM-L6-v2)<br/>• Course metadata

    Note over Session: Session Management<br/>• Maintains last 2 conversations<br/>• Provides context for follow-ups

    Note over Tools: Search Tools<br/>• CourseSearchTool<br/>• Vector similarity search<br/>• Source attribution
```

## Key Components

### Frontend (Vanilla JS)
- **Input Handling**: Captures user queries and manages UI state
- **API Communication**: POST requests to `/api/query` endpoint
- **Response Rendering**: Markdown parsing and source display

### Backend API (FastAPI)
- **Route Handler**: `/api/query` endpoint processes requests
- **Session Management**: Creates/maintains session IDs
- **Error Handling**: HTTP exceptions and response formatting

### RAG System Core
- **Query Orchestration**: Coordinates all components
- **Tool Integration**: Manages search tools for context retrieval
- **Response Generation**: Interfaces with Claude API

### Vector Store (ChromaDB)
- **Document Storage**: 800-character chunks with 100-char overlap
- **Embeddings**: sentence-transformers model (all-MiniLM-L6-v2)
- **Semantic Search**: Returns top 5 most relevant chunks

### AI Generator (Anthropic Claude)
- **Model**: claude-sonnet-4-20250514
- **Tool Usage**: Executes CourseSearchTool for context
- **Context Integration**: Combines query + history + retrieved content

### Session Manager
- **Conversation History**: Stores last 2 message exchanges
- **Context Continuity**: Enables follow-up questions
- **Memory Management**: Prevents context overflow

## Data Flow Summary

1. **User Input** → Frontend captures and validates
2. **API Request** → JSON payload with query and session
3. **RAG Processing** → Orchestrates search and generation
4. **Vector Search** → Finds relevant document chunks
5. **AI Generation** → Claude processes with context
6. **Response Assembly** → Combines answer with sources
7. **Frontend Display** → Renders markdown with attribution