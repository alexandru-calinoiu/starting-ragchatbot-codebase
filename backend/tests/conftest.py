import os
import sys
from unittest.mock import MagicMock, Mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from models import Course, CourseChunk, Lesson
from vector_store import SearchResults


@pytest.fixture
def test_config():
    config = Config()
    config.MAX_RESULTS = 5
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.CHROMA_PATH = "./test_chroma_db"
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
                lesson_link="https://learn.deeplearning.ai/test/lesson/1",
            ),
            Lesson(
                lesson_number=1,
                title="Overview",
                lesson_link="https://learn.deeplearning.ai/test/lesson/2",
            ),
        ],
    )


@pytest.fixture
def sample_chunks(sample_course):
    return [
        CourseChunk(
            content="Lesson 0 content: This is introduction to computer use with Anthropic models.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=0,
        ),
        CourseChunk(
            content="Computer use allows models to interact with computers through screenshots and actions.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=1,
        ),
        CourseChunk(
            content="Lesson 1 content: Overview of Anthropic's approach to AI safety and alignment.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def sample_search_results():
    return SearchResults(
        documents=[
            "Lesson 0 content: This is introduction to computer use with Anthropic models.",
            "Computer use allows models to interact with computers through screenshots and actions.",
        ],
        metadata=[
            {
                "course_title": "Building Towards Computer Use with Anthropic",
                "lesson_number": 0,
            },
            {
                "course_title": "Building Towards Computer Use with Anthropic",
                "lesson_number": 0,
            },
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def empty_search_results():
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="Search error: Connection timeout",
    )


def search_results_for_query(query: str) -> SearchResults:
    """Returns query-specific mock search results. Add more queries as needed."""
    query_results = {
        "computer use": SearchResults(
            documents=[
                "Computer use allows models to interact with computers through screenshots and actions.",
                "The computer use capability combines vision, tool use, and agentic workflows.",
            ],
            metadata=[
                {
                    "course_title": "Building Towards Computer Use with Anthropic",
                    "lesson_number": 0,
                },
                {
                    "course_title": "Building Towards Computer Use with Anthropic",
                    "lesson_number": 0,
                },
            ],
            distances=[0.1, 0.15],
        ),
        "AI safety": SearchResults(
            documents=[
                "AI safety is a core principle at Anthropic focusing on alignment and interpretability."
            ],
            metadata=[
                {
                    "course_title": "Building Towards Computer Use with Anthropic",
                    "lesson_number": 1,
                }
            ],
            distances=[0.12],
        ),
        "prompting tips": SearchResults(
            documents=[
                "Effective prompting includes chain of thought and few-shot examples."
            ],
            metadata=[
                {
                    "course_title": "Building Towards Computer Use with Anthropic",
                    "lesson_number": 3,
                }
            ],
            distances=[0.08],
        ),
    }

    return query_results.get(
        query,
        SearchResults([], [], [], error=f"No mock registered for query: '{query}'"),
    )


@pytest.fixture
def mock_vector_store_with_query_support():
    """Mock vector store that returns query-specific results."""
    mock_store = Mock()
    mock_store.search.side_effect = lambda **kwargs: search_results_for_query(
        kwargs.get("query", "")
    )
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
    mock_response.content = [
        MagicMock(
            text="Based on the search results, computer use allows models to interact with computers."
        )
    ]
    mock_response.stop_reason = "end_turn"
    return mock_response
