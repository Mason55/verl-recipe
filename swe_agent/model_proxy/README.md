# Model Proxy Server for SWE Agent

This module provides a lightweight HTTP proxy server that intercepts OpenAI-compatible API calls from SWE-Agent and forwards them to VERL for processing.

## Overview

The proxy implements an "anti-call" mechanism similar to ROCK's ModelService:

1. SWE-Agent calls `/v1/chat/completions` → proxy suspends the request
2. VERL calls `get_request()` to retrieve the request
3. VERL generates a response and calls `send_response()`
4. Proxy returns the OpenAI-format response to SWE-Agent

## Usage

### Basic Example

```python
import asyncio
from recipe.swe_agent.model_proxy import ModelProxy

async def main():
    # Create proxy instance
    proxy = ModelProxy(port=8080)
    
    try:
        # Start the server
        await proxy.start_server()
        
        # In VERL loop:
        while True:
            # Get the next request from SWE-Agent
            request = await proxy.get_request()
            
            # Check if session ended
            if request.is_session_end():
                break
            
            # Generate response (using VERL's model)
            response_text = await generate_response(request.messages)
            
            # Send response back to SWE-Agent
            await proxy.send_response(response_text)
    
    finally:
        # Stop the server
        await proxy.stop_server()

asyncio.run(main())
```

### Configuration

Configure SWE-Agent to use the proxy by setting:

```bash
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=dummy_key  # Not used by proxy
```

## API Reference

### ModelRequest

Represents a model call request from SWE-Agent.

**Attributes:**
- `request_id` (str): Unique request identifier
- `messages` (list): OpenAI format messages list
- `model` (Optional[str]): Model name
- `temperature` (Optional[float]): Temperature parameter
- `max_tokens` (Optional[int]): Max tokens parameter
- `stream` (bool): Whether streaming is requested
- `extra_params` (Optional[dict]): Additional OpenAI API parameters

**Methods:**
- `is_session_end() -> bool`: Check if this is a session end marker

### ModelProxy

Main proxy server class.

**Methods:**

#### `__init__(port: int = 8080, host: str = "0.0.0.0")`

Initialize the model proxy.

#### `async start_server(port: Optional[int] = None) -> None`

Start the HTTP proxy server. If `port` is provided, overrides the port set in `__init__`.

#### `async stop_server() -> None`

Stop the HTTP proxy server and clean up resources.

#### `async get_request() -> ModelRequest`

Get the next model call request from the queue. Blocks until a request is available.

#### `async send_response(
    response: str,
    request: Optional[ModelRequest] = None,
    request_id: Optional[str] = None,
    finish_reason: str = "stop"
) -> None`

Send a response back to SWE-Agent. The `response` parameter is required. You can optionally provide:
- `request`: The ModelRequest object (uses its request_id)
- `request_id`: The request ID explicitly
- If neither is provided, uses the most recently retrieved request from `get_request()`

#### `async send_error_response(
    request_id: str,
    error_message: str,
    error_type: str = "server_error"
) -> None`

Send an error response back to SWE-Agent.

## Architecture

The proxy uses:
- `asyncio.Queue` to store incoming requests waiting for VERL processing
- `asyncio.Event` to synchronize responses (one event per request)
- Dictionary to map request IDs to response events and data

This design supports concurrent requests from SWE-Agent, with each request having its own response synchronization mechanism.

## Error Handling

The proxy handles:
- Missing or invalid request data
- Request cancellation
- Internal server errors
- Missing responses

All errors are returned to SWE-Agent in OpenAI error format.
