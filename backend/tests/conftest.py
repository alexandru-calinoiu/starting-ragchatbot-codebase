import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
import tempfile
from fastapi.testclient import TestClient
from typing import Generator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from config import Config


@pytest.fixture
def test_config():
    import tempfile
    config = Config()
    config.MAX_RESULTS = 5
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.CHROMA_PATH = tempfile.mkdtemp(prefix="test_chroma_")
    return config


@pytest.fixture
def sample_course():
    return Course(
        title="Building Towards Computer Use with Anthropic",
        course_link="https://www.deeplearning.ai/short-courses/test/",
        instructor="Test Instructor",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Introduction",
                lesson_link="https://learn.deeplearning.ai/test/lesson/1"
            ),
            Lesson(
                lesson_number=1,
                title="Overview",
                lesson_link="https://learn.deeplearning.ai/test/lesson/2"
            )
        ]
    )


@pytest.fixture
def sample_chunks(sample_course):
    return [
        CourseChunk(
            content="Lesson 0 content: This is introduction to computer use with Anthropic models.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="Computer use allows models to interact with computers through screenshots and actions.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=1
        ),
        CourseChunk(
            content="Lesson 1 content: Overview of Anthropic's approach to AI safety and alignment.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=2
        )
    ]


@pytest.fixture
def sample_search_results():
    return SearchResults(
        documents=[
            "Lesson 0 content: This is introduction to computer use with Anthropic models.",
            "Computer use allows models to interact with computers through screenshots and actions."
        ],
        metadata=[
            {
                "course_title": "Building Towards Computer Use with Anthropic",
                "lesson_number": 0
            },
            {
                "course_title": "Building Towards Computer Use with Anthropic",
                "lesson_number": 0
            }
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def empty_search_results():
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def error_search_results():
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="Search error: Connection timeout"
    )


def search_results_for_query(query: str) -> SearchResults:
    """Returns query-specific mock search results. Add more queries as needed."""
    query_results = {
        "computer use": SearchResults(
            documents=[
                "Computer use allows models to interact with computers through screenshots and actions.",
                "The computer use capability combines vision, tool use, and agentic workflows."
            ],
            metadata=[
                {"course_title": "Building Towards Computer Use with Anthropic", "lesson_number": 0},
                {"course_title": "Building Towards Computer Use with Anthropic", "lesson_number": 0}
            ],
            distances=[0.1, 0.15]
        ),
        "AI safety": SearchResults(
            documents=[
                "AI safety is a core principle at Anthropic focusing on alignment and interpretability."
            ],
            metadata=[
                {"course_title": "Building Towards Computer Use with Anthropic", "lesson_number": 1}
            ],
            distances=[0.12]
        ),
        "prompting tips": SearchResults(
            documents=[
                "Effective prompting includes chain of thought and few-shot examples."
            ],
            metadata=[
                {"course_title": "Building Towards Computer Use with Anthropic", "lesson_number": 3}
            ],
            distances=[0.08]
        ),
    }

    return query_results.get(
        query,
        SearchResults([], [], [], error=f"No mock registered for query: '{query}'")
    )


@pytest.fixture
def mock_vector_store_with_query_support():
    """Mock vector store that returns query-specific results."""
    mock_store = Mock()
    mock_store.search.side_effect = lambda **kwargs: search_results_for_query(kwargs.get('query', ''))
    mock_store.get_lesson_link = Mock(return_value="https://test.com/lesson/1")
    return mock_store


@pytest.fixture
def mock_vector_store():
    """Basic mock vector store for manual configuration in tests."""
    mock_store = Mock()
    mock_store.search = Mock()
    mock_store.get_lesson_link = Mock(return_value="https://test.com/lesson/1")
    return mock_store


@pytest.fixture
def mock_anthropic_client():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="This is a test response")]
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_tool_use_response():
    mock_response = MagicMock()

    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Let me search for that information."

    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.id = "tool_123"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.input = {"query": "computer use"}

    mock_response.content = [mock_text_block, mock_tool_block]
    mock_response.stop_reason = "tool_use"

    return mock_response


@pytest.fixture
def mock_final_response():
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Based on the search results, computer use allows models to interact with computers.")]
    mock_response.stop_reason = "end_turn"
    return mock_response


@pytest.fixture
def mock_rag_system():
    """Mock RAG system for API testing."""
    mock_rag = Mock()
    mock_rag.query.return_value = (
        "This is a test response based on course materials.",
        ["Course: Building Towards Computer Use with Anthropic - Lesson 1"]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": [
            "Building Towards Computer Use with Anthropic",
            "AI Safety Fundamentals",
            "Prompt Engineering Best Practices"
        ]
    }
    mock_rag.session_manager = Mock()
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    mock_rag.session_manager.clear_session.return_value = None
    mock_rag.add_course_folder.return_value = (1, 10)
    return mock_rag


@pytest.fixture
def test_app(mock_rag_system):
    """Create test FastAPI app that mirrors the actual app structure."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Create test app with same configuration as real app
    app = FastAPI(title="Course Materials RAG System", root_path="")
    
    # Add trusted host middleware (same as real app)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    # Add CORS middleware (same as real app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models (same as real app)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str
    
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # API endpoints (same as real app)
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/sessions/{session_id}/clear")
    async def clear_session(session_id: str):
        try:
            mock_rag_system.session_manager.clear_session(session_id)
            return {"message": "Session cleared successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


@pytest.fixture
def test_client(test_app) -> Generator:
    """Create test client for API testing."""
    client = TestClient(test_app)
    yield client


@pytest.fixture
def temp_frontend_dir():
    """Create temporary frontend directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal frontend files
        index_path = os.path.join(tmpdir, "index.html")
        with open(index_path, "w") as f:
            f.write("<html><body>Test</body></html>")
        yield tmpdir


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client."""
    mock_client = Mock()
    mock_collection = Mock()
    mock_collection.query.return_value = {
        "documents": [["Test document content"]],
        "metadatas": [[{"course_title": "Test Course", "lesson_number": 0}]],
        "distances": [[0.1]]
    }
    mock_collection.add.return_value = None
    mock_collection.delete.return_value = None
    mock_client.get_or_create_collection.return_value = mock_collection
    return mock_client


@pytest.fixture
def sample_query_request():
    """Sample query request data."""
    return {
        "query": "What is computer use in Anthropic models?",
        "session_id": None
    }


@pytest.fixture
def sample_query_request_with_session():
    """Sample query request with existing session."""
    return {
        "query": "Tell me more about safety measures.",
        "session_id": "existing-session-456"
    }