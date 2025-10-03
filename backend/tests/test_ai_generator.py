import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class TestAIGeneratorBasicResponse:

    def test_generate_response_without_tools_returns_text(self, test_config):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is a simple response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            generator.client = mock_client

            result = generator.generate_response(query="What is 2+2?")

            assert result == "This is a simple response"

    def test_generate_response_includes_query_in_messages(self, test_config):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            generator.client = mock_client

            result = generator.generate_response(query="What is AI?")

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs["messages"][0]["content"] == "What is AI?"

    def test_generate_response_includes_conversation_history_in_system(self, test_config):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            generator.client = mock_client
            history = "User: Hello\nAssistant: Hi there!"

            result = generator.generate_response(query="New question", conversation_history=history)

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert "Hello" in call_kwargs["system"]
            assert "Hi there!" in call_kwargs["system"]


class TestAIGeneratorWithTools:

    def test_generate_response_includes_tools_in_api_call_when_provided(self, test_config):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            generator.client = mock_client
            tools = [{"name": "test_tool", "description": "A test tool"}]

            # Mock tool_manager that returns tools
            mock_tool_manager = MagicMock()
            mock_tool_manager.get_tool_definitions.return_value = tools

            result = generator.generate_response(query="Test", tool_manager=mock_tool_manager)

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert "tools" in call_kwargs
            assert call_kwargs["tools"] == tools

    def test_generate_response_sets_tool_choice_to_auto_when_tools_provided(self, test_config):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            generator.client = mock_client
            tools = [{"name": "test_tool"}]

            # Mock tool_manager that returns tools
            mock_tool_manager = MagicMock()
            mock_tool_manager.get_tool_definitions.return_value = tools

            result = generator.generate_response(query="Test", tool_manager=mock_tool_manager)

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs["tool_choice"]["type"] == "auto"


class TestAIGeneratorToolExecution:

    def test_generate_response_executes_tool_when_stop_reason_is_tool_use(self, test_config, mock_vector_store_with_query_support):
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
        final_response.content = [MagicMock(text="Based on the search, computer use enables AI interaction.")]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            generator.client = mock_client

            tool_manager = ToolManager()
            tool = CourseSearchTool(mock_vector_store_with_query_support)
            tool_manager.register_tool(tool)

            result = generator.generate_response(query="What is computer use?", tool_manager=tool_manager)

            assert "computer use enables AI interaction" in result

    def test_generate_response_makes_second_api_call_after_tool_execution(self, test_config, mock_vector_store_with_query_support):
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
        final_response.content = [MagicMock(text="Final response")]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            generator.client = mock_client

            tool_manager = ToolManager()
            tool = CourseSearchTool(mock_vector_store_with_query_support)
            tool_manager.register_tool(tool)

            result = generator.generate_response(query="Test", tool_manager=tool_manager)

            assert mock_client.messages.create.call_count == 2

    def test_handle_tool_execution_includes_tool_results_in_second_call(self, test_config, mock_vector_store_with_query_support):
        mock_client = MagicMock()

        tool_use_response = MagicMock()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_456"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "AI safety"}
        tool_use_response.content = [tool_block]
        tool_use_response.stop_reason = "tool_use"

        final_response = MagicMock()
        final_response.content = [MagicMock(text="Final response")]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            generator.client = mock_client

            tool_manager = ToolManager()
            tool = CourseSearchTool(mock_vector_store_with_query_support)
            tool_manager.register_tool(tool)

            result = generator.generate_response(query="Test", tool_manager=tool_manager)

            second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
            messages = second_call_kwargs["messages"]
            assert len(messages) == 3
            assert messages[2]["role"] == "user"
            assert messages[2]["content"][0]["type"] == "tool_result"
            assert "tool_456" == messages[2]["content"][0]["tool_use_id"]


class TestAIGeneratorConfiguration:

    def test_base_params_includes_model(self, test_config):
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)

            assert generator.base_params["model"] == test_config.ANTHROPIC_MODEL

    def test_base_params_includes_temperature(self, test_config):
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)

            assert "temperature" in generator.base_params

    def test_base_params_includes_max_tokens(self, test_config):
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)

            assert "max_tokens" in generator.base_params