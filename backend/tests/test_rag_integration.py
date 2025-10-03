import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_system import RAGSystem
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestRAGSystemQuery:

    def test_query_returns_response_and_sources_tuple(
        self, test_config, mock_vector_store_with_query_support
    ):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text="Computer use is a capability that allows AI models to interact with computers."
            )
        ]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        with (
            patch("ai_generator.anthropic.Anthropic", return_value=mock_client),
            patch(
                "rag_system.VectorStore",
                return_value=mock_vector_store_with_query_support,
            ),
        ):
            rag = RAGSystem(test_config)

            response, sources = rag.query("What is computer use?")

            assert isinstance(response, str)
            assert isinstance(sources, list)

    def test_query_with_content_question_uses_search_tool(
        self, test_config, mock_vector_store_with_query_support
    ):
        mock_client = MagicMock()

        tool_use_response = MagicMock()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_123"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "computer use"}
        tool_use_response.content = [tool_block]
        tool_use_response.stop_reason = "tool_use"

        final_response = MagicMock()
        final_response.content = [
            MagicMock(
                text="Computer use allows models to interact with computers through screenshots."
            )
        ]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        with (
            patch("ai_generator.anthropic.Anthropic", return_value=mock_client),
            patch(
                "rag_system.VectorStore",
                return_value=mock_vector_store_with_query_support,
            ),
        ):
            rag = RAGSystem(test_config)

            response, sources = rag.query("What is computer use in the course?")

            assert "Computer use allows models" in response

    def test_query_returns_sources_from_search_tool(
        self, test_config, mock_vector_store_with_query_support
    ):
        mock_client = MagicMock()

        tool_use_response = MagicMock()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_123"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "computer use"}
        tool_use_response.content = [tool_block]
        tool_use_response.stop_reason = "tool_use"

        final_response = MagicMock()
        final_response.content = [MagicMock(text="Response")]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        with (
            patch("ai_generator.anthropic.Anthropic", return_value=mock_client),
            patch(
                "rag_system.VectorStore",
                return_value=mock_vector_store_with_query_support,
            ),
        ):
            rag = RAGSystem(test_config)

            # Recreate search tool with new vector store
            rag.search_tool = CourseSearchTool(mock_vector_store_with_query_support)
            rag.tool_manager.register_tool(rag.search_tool)

            response, sources = rag.query("What is computer use?")

            assert len(sources) > 0

    def test_query_with_session_id_includes_conversation_history(
        self, test_config, mock_vector_store_with_query_support
    ):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        with (
            patch("ai_generator.anthropic.Anthropic", return_value=mock_client),
            patch(
                "rag_system.VectorStore",
                return_value=mock_vector_store_with_query_support,
            ),
        ):
            rag = RAGSystem(test_config)
            session_id = rag.session_manager.create_session()
            rag.session_manager.add_exchange(
                session_id, "Previous question", "Previous answer"
            )

            response, sources = rag.query("Follow-up question", session_id=session_id)

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert "Previous question" in call_kwargs["system"]
            assert "Previous answer" in call_kwargs["system"]

    def test_query_updates_session_history_after_response(
        self, test_config, mock_vector_store_with_query_support
    ):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is the answer")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        with (
            patch("ai_generator.anthropic.Anthropic", return_value=mock_client),
            patch(
                "rag_system.VectorStore",
                return_value=mock_vector_store_with_query_support,
            ),
        ):
            rag = RAGSystem(test_config)
            session_id = rag.session_manager.create_session()

            response, sources = rag.query("What is AI?", session_id=session_id)

            history = rag.session_manager.get_conversation_history(session_id)
            assert "What is AI?" in history
            assert "This is the answer" in history


class TestRAGSystemCourseManagement:

    def test_get_course_analytics_returns_dict_with_total_courses(self, test_config):
        mock_vector_store = Mock()
        mock_vector_store.get_course_count = Mock(return_value=3)
        mock_vector_store.get_existing_course_titles = Mock(
            return_value=["Course1", "Course2", "Course3"]
        )

        with (
            patch("ai_generator.anthropic.Anthropic"),
            patch("rag_system.VectorStore", return_value=mock_vector_store),
        ):
            rag = RAGSystem(test_config)

            analytics = rag.get_course_analytics()

            assert analytics["total_courses"] == 3

    def test_get_course_analytics_returns_dict_with_course_titles(self, test_config):
        mock_vector_store = Mock()
        mock_vector_store.get_course_count = Mock(return_value=2)
        mock_vector_store.get_existing_course_titles = Mock(
            return_value=["CourseA", "CourseB"]
        )

        with (
            patch("ai_generator.anthropic.Anthropic"),
            patch("rag_system.VectorStore", return_value=mock_vector_store),
        ):
            rag = RAGSystem(test_config)

            analytics = rag.get_course_analytics()

            assert "CourseA" in analytics["course_titles"]
            assert "CourseB" in analytics["course_titles"]


class TestRAGSystemWithEmptyVectorStore:

    def test_query_with_empty_vector_store_returns_no_results_message(
        self, test_config
    ):
        mock_client = MagicMock()

        tool_use_response = MagicMock()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_123"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "test query"}
        tool_use_response.content = [tool_block]
        tool_use_response.stop_reason = "tool_use"

        final_response = MagicMock()
        final_response.content = [
            MagicMock(
                text="I couldn't find relevant information in the course materials."
            )
        ]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        empty_mock_store = Mock()
        empty_mock_store.search.side_effect = lambda **kwargs: (
            SearchResults(
                [],
                [],
                [],
                error=f"No mock registered for query: '{kwargs.get('query', '')}'",
            )
        )
        empty_mock_store.get_lesson_link = Mock(return_value=None)

        with (
            patch("ai_generator.anthropic.Anthropic", return_value=mock_client),
            patch("rag_system.VectorStore", return_value=empty_mock_store),
        ):
            rag = RAGSystem(test_config)

            response, sources = rag.query("What is computer use?")

            assert len(sources) == 0
