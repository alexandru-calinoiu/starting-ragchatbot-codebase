import os
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolDefinition:

    def test_tool_definition_has_name_field(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)

        definition = tool.get_tool_definition()

        assert "name" in definition

    def test_tool_definition_has_description_field(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)

        definition = tool.get_tool_definition()

        assert "description" in definition

    def test_tool_definition_has_input_schema_field(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)

        definition = tool.get_tool_definition()

        assert "input_schema" in definition

    def test_tool_definition_name_is_search_course_content(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)

        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"

    def test_tool_definition_query_is_required(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)

        definition = tool.get_tool_definition()

        assert "query" in definition["input_schema"]["required"]


class TestCourseSearchToolExecuteSuccess:

    def test_execute_returns_formatted_results_for_computer_use_query(
        self, mock_vector_store_with_query_support
    ):
        tool = CourseSearchTool(mock_vector_store_with_query_support)

        result = tool.execute(query="computer use")

        assert "Computer use allows models" in result

    def test_execute_includes_course_title_in_output(
        self, mock_vector_store_with_query_support
    ):
        tool = CourseSearchTool(mock_vector_store_with_query_support)

        result = tool.execute(query="computer use")

        assert "Building Towards Computer Use with Anthropic" in result

    def test_execute_includes_lesson_number_in_output(
        self, mock_vector_store_with_query_support
    ):
        tool = CourseSearchTool(mock_vector_store_with_query_support)

        result = tool.execute(query="computer use")

        assert "Lesson 0" in result

    def test_execute_returns_multiple_documents_joined(
        self, mock_vector_store_with_query_support
    ):
        tool = CourseSearchTool(mock_vector_store_with_query_support)

        result = tool.execute(query="computer use")

        assert "Computer use allows models" in result
        assert "agentic workflows" in result


class TestCourseSearchToolExecuteWithFilters:

    def test_execute_with_course_name_passes_filter_to_store(self, mock_vector_store):
        mock_vector_store.search.return_value = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
        )
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="test", course_name="Test Course")

        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["course_name"] == "Test Course"

    def test_execute_with_lesson_number_passes_filter_to_store(self, mock_vector_store):
        mock_vector_store.search.return_value = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 2}],
            distances=[0.1],
        )
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="test", lesson_number=2)

        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["lesson_number"] == 2


class TestCourseSearchToolExecuteErrors:

    def test_execute_returns_error_message_when_search_has_error(
        self, mock_vector_store
    ):
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error="Database connection failed"
        )
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="test")

        assert result == "Database connection failed"

    def test_execute_returns_no_content_message_when_empty_results(
        self, mock_vector_store
    ):
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="test")

        assert "No relevant content found" in result

    def test_execute_includes_course_filter_in_empty_message(self, mock_vector_store):
        mock_vector_store.search.return_value = SearchResults([], [], [])
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="test", course_name="Nonexistent Course")

        assert "Nonexistent Course" in result

    def test_execute_includes_lesson_filter_in_empty_message(self, mock_vector_store):
        mock_vector_store.search.return_value = SearchResults([], [], [])
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="test", lesson_number=99)

        assert "lesson 99" in result


class TestCourseSearchToolSourceTracking:

    def test_last_sources_is_empty_initially(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)

        assert tool.last_sources == []

    def test_last_sources_populated_after_successful_execute(
        self, mock_vector_store_with_query_support
    ):
        tool = CourseSearchTool(mock_vector_store_with_query_support)

        tool.execute(query="computer use")

        assert len(tool.last_sources) == 2

    def test_last_sources_includes_course_and_lesson_info(
        self, mock_vector_store_with_query_support
    ):
        tool = CourseSearchTool(mock_vector_store_with_query_support)

        tool.execute(query="AI safety")

        source = tool.last_sources[0]
        assert "Building Towards Computer Use with Anthropic" in source
        assert "Lesson 1" in source

    def test_last_sources_includes_lesson_link_when_available(
        self, mock_vector_store_with_query_support
    ):
        mock_vector_store_with_query_support.get_lesson_link.return_value = (
            "https://example.com/lesson/1"
        )
        tool = CourseSearchTool(mock_vector_store_with_query_support)

        tool.execute(query="AI safety")

        source = tool.last_sources[0]
        assert "https://example.com/lesson/1" in source


class TestToolManager:

    def test_register_tool_adds_tool_to_manager(self, mock_vector_store):
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_get_tool_definitions_returns_list_of_definitions(self, mock_vector_store):
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        definitions = manager.get_tool_definitions()

        assert isinstance(definitions, list)
        assert len(definitions) == 1

    def test_get_tool_definitions_includes_registered_tool_definition(
        self, mock_vector_store
    ):
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        definitions = manager.get_tool_definitions()

        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool_calls_correct_tool_with_params(
        self, mock_vector_store_with_query_support
    ):
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store_with_query_support)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="computer use")

        assert "Computer use allows models" in result

    def test_execute_tool_returns_error_for_unknown_tool(self):
        manager = ToolManager()

        result = manager.execute_tool("unknown_tool", query="test")

        assert "not found" in result

    def test_get_last_sources_returns_sources_from_executed_tool(
        self, mock_vector_store_with_query_support
    ):
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store_with_query_support)
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="computer use")

        sources = manager.get_last_sources()

        assert len(sources) == 2

    def test_get_last_sources_returns_empty_list_when_no_sources(self):
        manager = ToolManager()

        sources = manager.get_last_sources()

        assert sources == []

    def test_reset_sources_clears_all_tool_sources(
        self, mock_vector_store_with_query_support
    ):
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store_with_query_support)
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="computer use")

        manager.reset_sources()
        sources = manager.get_last_sources()

        assert len(sources) == 0
