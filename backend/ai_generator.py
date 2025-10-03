import anthropic
from typing import Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **You may search up to 2 times** to gather information needed for complex questions
- Use sequential searches strategically:
  * First search: Get course outlines, lesson titles, or initial information
  * Second search (if needed): Get specific content based on first search results
- Synthesize all search results into accurate, fact-based responses
- If search yields no results or fails, provide the best answer possible with available information

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **Complex questions**: Use sequential searches to build complete context
- **If tools fail or return errors**: Acknowledge the limitation briefly and answer what you can
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tool_manager=None,
                         max_rounds: int = 2) -> str:
        """
        Generate AI response with sequential tool calling support (up to max_rounds).
        Implements graceful error handling - continues even if individual tool calls fail.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tool_manager: Manager to execute tools (provides tool definitions and execution)
            max_rounds: Maximum number of tool calling rounds (default: 2)

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Get tools from tool_manager if available
        tools = tool_manager.get_tool_definitions() if tool_manager else None

        # Initialize message list with user query
        messages = [{"role": "user", "content": query}]

        # Track rounds and success
        round_count = 0
        has_successful_tool_call = False

        # Sequential tool calling loop
        while round_count < max_rounds:
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }

            # Add tools if available (only when we can execute them)
            if tools and tool_manager:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Get response from Claude
            response = self.client.messages.create(**api_params)

            # Check if tool was used
            if response.stop_reason == "tool_use":
                # Execute tools and update messages for next round
                messages, tool_success = self._execute_tools_and_update_messages(
                    response, messages, tool_manager
                )
                if tool_success:
                    has_successful_tool_call = True
                round_count += 1
            else:
                # No tool use - we have our final response
                return response.content[0].text

        # Max rounds reached - make final call without tools to synthesize
        return self._get_final_response(messages, system_content, has_successful_tool_call)
    
    def _execute_tools_and_update_messages(self, response, messages, tool_manager):
        """
        Execute tools from Claude's response with error resilience.
        Even if tools fail, errors are passed to Claude for graceful handling.

        Args:
            response: Claude's response containing tool_use blocks
            messages: Current message list
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (updated_messages, success_flag)
            - updated_messages: Message list with assistant response and tool results
            - success_flag: True if at least one tool executed successfully
        """
        # Start with existing messages
        new_messages = messages.copy()

        # Add AI's tool use response
        new_messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls and collect results
        tool_results = []
        any_tool_succeeded = False

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    # Attempt tool execution
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                    any_tool_succeeded = True

                except Exception as e:
                    # Tool failed - pass error to Claude for graceful handling
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Error executing tool: {str(e)}",
                        "is_error": True
                    })

        # Add tool results as single message
        if tool_results:
            new_messages.append({"role": "user", "content": tool_results})

        return new_messages, any_tool_succeeded

    def _get_final_response(self, messages, system_content: str, has_successful_tool_call: bool) -> str:
        """
        Get final response after max rounds exhausted.
        Implements tiered fallback for error resilience.

        Args:
            messages: Complete message history
            system_content: System prompt
            has_successful_tool_call: Whether any tool call succeeded during execution

        Returns:
            Final response text
        """
        try:
            # Attempt final API call WITHOUT tools to force synthesis
            final_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
                # NOTE: No "tools" parameter - forces text response
            }

            final_response = self.client.messages.create(**final_params)
            return final_response.content[0].text

        except Exception:
            # Final API call failed - return appropriate error message
            if has_successful_tool_call:
                return "I found some information but encountered an error completing your request. Please try asking your question again."
            else:
                return "I'm having trouble accessing course information right now. Please try asking a general question I can answer directly, or try again later."