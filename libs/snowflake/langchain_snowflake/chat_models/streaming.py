"""Streaming functionality for Snowflake chat models."""

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk

from .._connection.rest_client import RestApiClient, RestApiRequestBuilder
from .._error_handling import SnowflakeErrorHandler

logger = logging.getLogger(__name__)


class SnowflakeStreaming:
    """Mixin class for Snowflake streaming functionality."""

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completions from Snowflake Cortex.

        Args:
            messages: List of messages to send to the model
            stop: List of stop sequences (not supported by Cortex)
            run_manager: Callback manager for the run
            **kwargs: Additional keyword arguments

        Yields:
            ChatGenerationChunk: Streaming chunks of the response

        Note: Uses Cortex COMPLETE's native streaming via REST API when tools
        are bound, otherwise falls back to simulated streaming for SQL function.
        """
        try:
            if self._should_use_rest_api():
                # Use native streaming via REST API
                for chunk in self._stream_via_rest_api(messages, run_manager, **kwargs):
                    yield chunk
            else:
                # Simulate streaming by chunking the complete response
                result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

                if result.generations:
                    content = result.generations[0].message.content

                    # Split content into chunks for streaming effect
                    chunk_size = max(1, len(content) // 20)  # Aim for ~20 chunks

                    for i in range(0, len(content), chunk_size):
                        chunk_content = content[i : i + chunk_size]

                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(
                                content=chunk_content,
                                usage_metadata=(result.generations[0].message.usage_metadata if i == 0 else None),
                                response_metadata=(result.generations[0].message.response_metadata if i == 0 else {}),
                            )
                        )

                        yield chunk

                        if run_manager:
                            run_manager.on_llm_new_token(chunk_content)

        except Exception as e:
            # Use centralized error handling to create consistent error response
            error_result = SnowflakeErrorHandler.create_chat_error_result(
                error=e,
                operation="stream chat completions",
                model=self.model,
                input_tokens=self._estimate_tokens(messages),
            )
            # Convert ChatResult to streaming chunk format
            error_content = error_result.generations[0].message.content
            yield ChatGenerationChunk(message=AIMessageChunk(content=error_content))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream response by delegating to sync _stream method.

        This eliminates code duplication by using the sync implementation
        with asyncio.to_thread() for non-blocking execution.

        IMPORTANT: Always yields at least one chunk to avoid "No generations found in stream"
        error when used with astream_events.
        """
        # Determine streaming method based on tool requirements
        has_yielded = False
        logger.info(f"_astream called with {len(messages)} messages")
        try:
            should_use_rest = self._should_use_rest_api()
            logger.info(f"_should_use_rest_api returned: {should_use_rest}")
            if should_use_rest:
                # Use native async REST API streaming with aiohttp
                async for chunk in self._astream_via_rest_api(messages, run_manager, **kwargs):
                    has_yielded = True
                    yield chunk
            else:
                # For SQL-based streaming, we currently delegate to sync method since
                # Snowflake doesn't support native SQL streaming, only batch results
                def sync_stream():
                    return list(self._stream(messages, stop, run_manager, **kwargs))

                chunks = await asyncio.to_thread(sync_stream)
                for chunk in chunks:
                    has_yielded = True
                    yield chunk

        except Exception as e:
            # Use centralized error handling for consistent async streaming errors
            error_result = SnowflakeErrorHandler.create_chat_error_result(
                error=e,
                operation="async stream chat completions",
                model=self.model,
                input_tokens=self._estimate_tokens(messages),
            )
            # Convert ChatResult to streaming chunk format
            error_content = error_result.generations[0].message.content
            error_chunk = ChatGenerationChunk(message=AIMessageChunk(content=error_content))
            has_yielded = True
            yield error_chunk

        # CRITICAL FIX: Always yield at least one chunk to avoid "No generations found in stream"
        # This is required for compatibility with LangChain's astream_events
        if not has_yielded:
            # Yield an empty chunk with minimal content
            empty_chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=""),
                generation_info={"fallback": True}
            )
            yield empty_chunk

    def _stream_via_rest_api(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completions via REST API with native streaming support."""
        try:
            # Build REST API payload with streaming enabled
            payload = self._build_rest_api_payload(messages)
            payload["stream"] = True  # Enable native streaming

            # Add generation parameters directly
            payload.update(
                {
                    "temperature": getattr(self, "temperature", 0.7),
                    "max_tokens": getattr(self, "max_tokens", 4096),
                    "top_p": getattr(self, "top_p", 1.0),
                }
            )

            # Get session for centralized REST API client
            session = self._get_session()

            # Use centralized REST API client for streaming
            request_config = RestApiRequestBuilder.cortex_complete_request(
                session=session,
                method="POST",
                payload=payload,
                request_timeout=self.request_timeout,
                verify_ssl=self.verify_ssl,
            )

            # Use centralized streaming
            for chunk_json in RestApiClient.make_sync_streaming_request(request_config, "streaming Cortex Complete"):
                if chunk_json:
                    # Parse JSON chunk and extract content
                    try:
                        chunk_data = json.loads(chunk_json)
                        # Extract content from Cortex Complete format
                        if isinstance(chunk_data, dict):
                            chunk_content = chunk_data.get("content", "")
                        else:
                            chunk_content = str(chunk_data)
                    except (json.JSONDecodeError, TypeError):
                        # Fallback: treat as plain text
                        chunk_content = chunk_json

                    if chunk_content:
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=chunk_content),
                            generation_info={"stream": True},
                        )
                        if run_manager:
                            run_manager.on_llm_new_token(chunk_content)
                        yield chunk

        except Exception as e:
            # Use centralized error handling
            error_content = f"Streaming error: {str(e)}"
            error_chunk = ChatGenerationChunk(message=AIMessageChunk(content=error_content))
            yield error_chunk

    async def _astream_via_rest_api(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream chat completions via REST API using aiohttp for true async."""
        try:
            # Build REST API payload with streaming enabled
            payload = self._build_rest_api_payload(messages)
            payload["stream"] = True  # Enable native streaming

            # Add generation parameters directly
            payload.update(
                {
                    "temperature": getattr(self, "temperature", 0.7),
                    "max_tokens": getattr(self, "max_tokens", 4096),
                    "top_p": getattr(self, "top_p", 1.0),
                }
            )

            # Get session for centralized REST API client
            session = self._get_session()

            # Use centralized REST API client for streaming
            from .._connection.rest_client import RestApiClient, RestApiRequestBuilder

            request_config = RestApiRequestBuilder.cortex_complete_request(
                session=session,
                method="POST",
                payload=payload,
                request_timeout=self.request_timeout,
                verify_ssl=self.verify_ssl,
            )

            # Use centralized async streaming
            logger.info("Starting async streaming from Cortex Complete REST API")
            chunk_count = 0
            async for chunk_json in RestApiClient.make_async_streaming_request(
                request_config, "async streaming Cortex Complete"
            ):
                chunk_count += 1
                logger.info(f"Received chunk #{chunk_count}: {repr(chunk_json)[:200]}")
                if chunk_json:
                    # Parse JSON chunk and extract content
                    try:
                        import json
                        from langchain_core.messages.tool import tool_call_chunk

                        chunk_data = json.loads(chunk_json)
                        logger.info(f"Parsed chunk data: {chunk_data}")
                        # Extract content from Cortex Complete format
                        # The format is: {"choices": [{"delta": {"text": "...", "type": "text"|"tool_use"}}]}
                        if isinstance(chunk_data, dict):
                            choices = chunk_data.get("choices", [])
                            if choices and isinstance(choices, list):
                                delta = choices[0].get("delta", {})
                                delta_type = delta.get("type")

                                # Handle text chunks
                                chunk_content = delta.get("text", "")

                                # Handle tool_use chunks
                                tool_call_chunks = []
                                if delta_type == "tool_use":
                                    tool_call_id = delta.get("tool_use_id", "")
                                    tool_name = delta.get("name", "")
                                    tool_input = delta.get("input", "")

                                    if tool_call_id or tool_name or tool_input:
                                        # Create tool call chunk
                                        tc_chunk = {
                                            "index": 0,
                                        }
                                        if tool_call_id:
                                            tc_chunk["id"] = tool_call_id
                                        if tool_name:
                                            tc_chunk["name"] = tool_name
                                        if tool_input:
                                            tc_chunk["args"] = tool_input

                                        tool_call_chunks = [tc_chunk]
                                        logger.info(f"Tool call chunk: {tc_chunk}")
                            else:
                                # Fallback to top-level content
                                chunk_content = chunk_data.get("content", "")
                                tool_call_chunks = []
                        else:
                            chunk_content = str(chunk_data)
                            tool_call_chunks = []
                    except (json.JSONDecodeError, TypeError) as e:
                        # Fallback: treat as plain text
                        logger.warning(f"JSON decode failed: {e}, treating as plain text")
                        chunk_content = chunk_json
                        tool_call_chunks = []

                    logger.info(f"Chunk content: {repr(chunk_content)}, tool_call_chunks: {len(tool_call_chunks)}")
                    # Yield chunk if we have either content or tool calls
                    if chunk_content or tool_call_chunks:
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(
                                content=chunk_content,
                                tool_call_chunks=tool_call_chunks
                            ),
                            generation_info={"stream": True},
                        )
                        if run_manager and chunk_content:
                            await run_manager.on_llm_new_token(chunk_content)
                        logger.info(f"Yielding chunk: content_length={len(chunk_content)}, tool_calls={len(tool_call_chunks)}")
                        yield chunk
                    else:
                        logger.info("Skipping chunk with no content or tool calls")
            logger.info(f"Async streaming completed. Total chunks received: {chunk_count}")

        except Exception as e:
            # Use centralized error handling
            error_content = f"Async streaming error: {str(e)}"
            error_chunk = ChatGenerationChunk(message=AIMessageChunk(content=error_content))
            yield error_chunk
