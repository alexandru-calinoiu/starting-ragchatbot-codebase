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


class TestSequentialToolCalling:
    """Tests for sequential tool calling with up to 2 rounds"""

    def test_sequential_two_round_tool_calling(self, test_config, mock_vector_store_with_query_support):
        """Test that Claude can make 2 sequential tool calls"""
        mock_client = MagicMock()

        # Round 1: Tool use
        round1_tool_block = MagicMock()
        round1_tool_block.type = "tool_use"
        round1_tool_block.id = "tool_round1"
        round1_tool_block.name = "search_course_content"
        round1_tool_block.input = {"query": "Python course"}
        round1_response = MagicMock()
        round1_response.content = [round1_tool_block]
        round1_response.stop_reason = "tool_use"

        # Round 2: Tool use again
        round2_tool_block = MagicMock()
        round2_tool_block.type = "tool_use"
        round2_tool_block.id = "tool_round2"
        round2_tool_block.name = "search_course_content"
        round2_tool_block.input = {"query": "Advanced Python"}
        round2_response = MagicMock()
        round2_response.content = [round2_tool_block]
        round2_response.stop_reason = "tool_use"

        # Final response after max rounds
        final_response = MagicMock()
        final_response.content = [MagicMock(text="Based on both searches, here's the answer...")]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            generator.client = mock_client

            tool_manager = ToolManager()
            tool = CourseSearchTool(mock_vector_store_with_query_support)
            tool_manager.register_tool(tool)

            result = generator.generate_response(query="Compare Python courses", tool_manager=tool_manager)

            # Should make 3 API calls total (round 1, round 2, final)
            assert mock_client.messages.create.call_count == 3
            assert "Based on both searches" in result

    def test_max_rounds_enforced_at_two(self, test_config, mock_vector_store_with_query_support):
        """Test that system stops at 2 rounds even if Claude wants more"""
        mock_client = MagicMock()

        # Create 3 tool_use responses (but should only use 2)
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_123"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "test"}

        tool_response = MagicMock()
        tool_response.content = [tool_block]
        tool_response.stop_reason = "tool_use"

        # Final response without tools
        final_response = MagicMock()
        final_response.content = [MagicMock(text="Final answer")]
        final_response.stop_reason = "end_turn"

        # Mock will return tool_use twice, then final
        mock_client.messages.create.side_effect = [tool_response, tool_response, final_response]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            generator.client = mock_client

            tool_manager = ToolManager()
            tool = CourseSearchTool(mock_vector_store_with_query_support)
            tool_manager.register_tool(tool)

            result = generator.generate_response(query="Test", tool_manager=tool_manager, max_rounds=2)

            # Should make exactly 3 calls: round1 + round2 + final (no tools)
            assert mock_client.messages.create.call_count == 3

            # Third call should NOT include tools (forces synthesis)
            third_call_kwargs = mock_client.messages.create.call_args_list[2].kwargs
            assert "tools" not in third_call_kwargs

    def test_message_history_accumulates_across_rounds(self, test_config, mock_vector_store_with_query_support):
        """Test that message history grows correctly across rounds"""
        mock_client = MagicMock()

        # Round 1 tool use
        tool_block1 = MagicMock()
        tool_block1.type = "tool_use"
        tool_block1.id = "tool_1"
        tool_block1.name = "search_course_content"
        tool_block1.input = {"query": "first search"}
        response1 = MagicMock()
        response1.content = [tool_block1]
        response1.stop_reason = "tool_use"

        # Round 2 ends naturally
        response2 = MagicMock()
        response2.content = [MagicMock(text="Answer")]
        response2.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [response1, response2]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            generator.client = mock_client

            tool_manager = ToolManager()
            tool = CourseSearchTool(mock_vector_store_with_query_support)
            tool_manager.register_tool(tool)

            result = generator.generate_response(query="Test", tool_manager=tool_manager)

            # Check second API call has accumulated messages
            second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
            messages = second_call_kwargs["messages"]

            # Should have: [user query, assistant tool_use, user tool_results]
            assert len(messages) == 3
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"


class TestToolExecutionErrorResilience:
    """Tests for graceful error handling when tools fail"""

    def test_tool_execution_error_passed_to_claude(self, test_config):
        """When tool fails, error should be passed to Claude as tool_result"""
        mock_client = MagicMock()

        # Tool use response
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_error"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "test"}
        tool_response = MagicMock()
        tool_response.content = [tool_block]
        tool_response.stop_reason = "tool_use"

        # Second response after error
        final_response = MagicMock()
        final_response.content = [MagicMock(text="I cannot access that information right now")]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [tool_response, final_response]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            generator.client = mock_client

            # Mock tool_manager that raises error
            mock_tool_manager = MagicMock()
            mock_tool_manager.get_tool_definitions.return_value = [{"name": "search_course_content"}]
            mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")

            result = generator.generate_response(query="Test", tool_manager=mock_tool_manager)

            # Should still complete and return response
            assert result == "I cannot access that information right now"

            # Check that error was passed to second API call
            second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
            tool_results = second_call_kwargs["messages"][2]["content"]
            assert tool_results[0]["is_error"] == True
            assert "Error executing tool" in tool_results[0]["content"]

    def test_partial_tool_success_continues_execution(self, test_config, mock_vector_store_with_query_support):
        """Round 1 fails, Round 2 succeeds - should continue"""
        mock_client = MagicMock()

        # Round 1: tool use
        tool_block1 = MagicMock()
        tool_block1.type = "tool_use"
        tool_block1.id = "tool_1"
        tool_block1.name = "search_course_content"
        tool_block1.input = {"query": "first"}
        response1 = MagicMock()
        response1.content = [tool_block1]
        response1.stop_reason = "tool_use"

        # Round 2: tool use again
        tool_block2 = MagicMock()
        tool_block2.type = "tool_use"
        tool_block2.id = "tool_2"
        tool_block2.name = "search_course_content"
        tool_block2.input = {"query": "second"}
        response2 = MagicMock()
        response2.content = [tool_block2]
        response2.stop_reason = "tool_use"

        # Final
        final_response = MagicMock()
        final_response.content = [MagicMock(text="Here's what I found in the second search")]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [response1, response2, final_response]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            generator.client = mock_client

            # Tool manager that fails first, succeeds second
            tool_manager = ToolManager()
            tool = CourseSearchTool(mock_vector_store_with_query_support)
            tool_manager.register_tool(tool)

            # Mock execute_tool to fail first time, succeed second time
            original_execute = tool_manager.execute_tool
            call_count = {"count": 0}

            def mock_execute(name, **kwargs):
                call_count["count"] += 1
                if call_count["count"] == 1:
                    raise Exception("First search failed")
                return original_execute(name, **kwargs)

            tool_manager.execute_tool = mock_execute

            result = generator.generate_response(query="Test", tool_manager=tool_manager)

            # Should complete despite first tool failure
            assert mock_client.messages.create.call_count == 3
            assert "second search" in result

    def test_complete_failure_returns_helpful_message(self, test_config):
        """If final API call fails and no tools succeeded, return helpful error"""
        mock_client = MagicMock()

        # Tool use that will fail
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_1"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "test"}
        tool_response = MagicMock()
        tool_response.content = [tool_block]
        tool_response.stop_reason = "tool_use"

        # Another tool use
        tool_response2 = MagicMock()
        tool_response2.content = [tool_block]
        tool_response2.stop_reason = "tool_use"

        # Final call will raise exception
        mock_client.messages.create.side_effect = [
            tool_response,
            tool_response2,
            Exception("API Error")
        ]

        with patch('ai_generator.anthropic.Anthropic', return_value=mock_client):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            generator.client = mock_client

            # Tool manager that always fails
            mock_tool_manager = MagicMock()
            mock_tool_manager.get_tool_definitions.return_value = [{"name": "search_course_content"}]
            mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")

            result = generator.generate_response(query="Test", tool_manager=mock_tool_manager)

            # Should return user-friendly error message
            assert "having trouble accessing course information" in result