import json
import logging
import asyncio
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Dict, Set, List, Optional
import os
from dotenv import load_dotenv
from mcp.server import Server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel, Field

load_dotenv()

from . import tools

# Load environment variables

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-obsidian")

api_key = os.getenv("OBSIDIAN_API_KEY")
if not api_key:
    raise ValueError(f"OBSIDIAN_API_KEY environment variable required. Working directory: {os.getcwd()}")

# Get port from environment variable or use default
port = int(os.getenv("PORT", "8000"))
logger.info(f"Configured to run on port {port}")

app = Server("mcp-obsidian")

tool_handlers = {}
def add_tool_handler(tool_class: tools.ToolHandler):
    global tool_handlers

    tool_handlers[tool_class.name] = tool_class

def get_tool_handler(name: str) -> tools.ToolHandler | None:
    if name not in tool_handlers:
        return None
    
    return tool_handlers[name]

add_tool_handler(tools.ListFilesInDirToolHandler())
add_tool_handler(tools.ListFilesInVaultToolHandler())
add_tool_handler(tools.GetFileContentsToolHandler())
add_tool_handler(tools.SearchToolHandler())
add_tool_handler(tools.PatchContentToolHandler())
add_tool_handler(tools.AppendContentToolHandler())
add_tool_handler(tools.ComplexSearchToolHandler())
add_tool_handler(tools.BatchGetFileContentsToolHandler())

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""

    return [th.get_tool_description() for th in tool_handlers.values()]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls for command line run."""
    
    if not isinstance(arguments, dict):
        raise RuntimeError("arguments must be dictionary")

    tool_handler = get_tool_handler(name)
    if not tool_handler:
        raise ValueError(f"Unknown tool: {name}")

    try:
        result = tool_handler.run_tool(arguments)
        
        # Send event through SSE if client_id is provided
        if "client_id" in arguments:
            client_id = arguments["client_id"]
            event_data = {
                "tool": name,
                "arguments": arguments,
                "status": "success",
                "result": [
                    {"type": content.type, "value": content.value}
                    for content in result
                    if hasattr(content, "type") and hasattr(content, "value")
                ]
            }
            asyncio.create_task(sse_manager.send_event(client_id, "tool_result", event_data))
        
        return result
    except Exception as e:
        logger.error(str(e))
        # Send error event through SSE if client_id is provided
        if "client_id" in arguments:
            client_id = arguments["client_id"]
            event_data = {
                "tool": name,
                "arguments": arguments,
                "status": "error",
                "error": str(e)
            }
            asyncio.create_task(sse_manager.send_event(client_id, "tool_error", event_data))
        raise RuntimeError(f"Caught Exception. Error: {str(e)}")

class SSEManager:
    def __init__(self):
        self.connections: Dict[str, Set[asyncio.Queue]] = {}
        
    async def register(self, client_id: str) -> asyncio.Queue:
        if client_id not in self.connections:
            self.connections[client_id] = set()
        queue = asyncio.Queue()
        self.connections[client_id].add(queue)
        return queue
    
    async def unregister(self, client_id: str, queue: asyncio.Queue):
        if client_id in self.connections:
            self.connections[client_id].discard(queue)
            if not self.connections[client_id]:
                del self.connections[client_id]
    
    async def send_event(self, client_id: str, event_type: str, data: Any):
        if client_id in self.connections:
            event_data = json.dumps({"type": event_type, "data": data})
            for queue in self.connections[client_id]:
                await queue.put(f"event: {event_type}\ndata: {event_data}\n\n")

sse_manager = SSEManager()

class ToolParameter(BaseModel):
    name: str
    type: str
    description: str

class ToolDescription(BaseModel):
    name: str
    description: str
    parameters: List[ToolParameter]

class ToolRequest(BaseModel):
    arguments: Dict[str, Any] = Field(..., description="Arguments for the tool")
    client_id: Optional[str] = Field(None, description="Optional client ID for SSE updates")

class ToolResponse(BaseModel):
    status: str = Field(..., description="Status of the tool execution")
    result: Any = Field(..., description="Result of the tool execution")
    error: Optional[str] = Field(None, description="Error message if status is 'error'")

class ToolsListResponse(BaseModel):
    status: str = Field(..., description="Status of the request")
    tools: List[ToolDescription] = Field(..., description="List of available tools")

def create_tool_routes():
    """Create FastAPI routes for each tool handler"""
    routes = []
    for name, handler in tool_handlers.items():
        tool_desc = handler.get_tool_description()
        
        # Create a closure to capture the current tool_name and tool_handler
        def create_endpoint(tool_name: str, tool_handler: tools.ToolHandler):
            async def endpoint(request: ToolRequest):
                try:
                    result = tool_handler.run_tool(request.arguments)
                    return ToolResponse(
                        status="success",
                        result=json.loads(result[0].text)
                    )
                except Exception as e:
                    logger.error(str(e))
                    if request.client_id:
                        event_data = {
                            "tool": tool_name,
                            "arguments": request.arguments,
                            "status": "error",
                            "error": str(e)
                        }
                        asyncio.create_task(sse_manager.send_event(request.client_id, "tool_error", event_data))
                    return ToolResponse(
                        status="error",
                        error=str(e),
                        result=None
                    )
            return endpoint
        
        route = fastapi_app.post(
            f"/tool/{name}",
            tags=["Tools"],
            summary=tool_desc.description,
            description=f"Execute the {name} tool",
            response_model=ToolResponse
        )(create_endpoint(name, handler))
        routes.append(route)
    return routes

fastapi_app = FastAPI(
    title="MCP Obsidian",
    description="MCP server to work with Obsidian via the remote REST plugin",
    version="0.2.1",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create routes for all tools
tool_routes = create_tool_routes()

@fastapi_app.get("/sse/{client_id}")
async def sse_endpoint(client_id: str):
    async def event_generator():
        queue = await sse_manager.register(client_id)
        try:
            while True:
                data = await queue.get()
                yield data
        finally:
            await sse_manager.unregister(client_id, queue)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@fastapi_app.get("/test-event/{client_id}")
async def test_event(client_id: str):
    await sse_manager.send_event(
        client_id,
        "test",
        {"message": "Test event received!", "timestamp": str(asyncio.get_running_loop().time())}
    )
    return {"status": "Event sent"}

@fastapi_app.get("/tools", response_model=ToolsListResponse, tags=["Tools"])
async def list_tools_http():
    """List all available tools with their descriptions and parameters."""
    tools = await list_tools()
    return ToolsListResponse(
        status="success",
        tools=[
            ToolDescription(
                name=tool.name,
                description=tool.description,
                parameters=[
                    ToolParameter(
                        name=param.name,
                        type=param.type,
                        description=param.description
                    )
                    for param in tool.parameters
                ] if hasattr(tool, "parameters") else []
            )
            for tool in tools
        ]
    )

# Modify the main function to run both MCP and FastAPI servers
async def main():
    # Import here to avoid issues with event loops
    from mcp.server.stdio import stdio_server
    
    # Create tasks for both servers
    async def run_fastapi():
        config = uvicorn.Config(fastapi_app, host="0.0.0.0", port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    
    async def run_mcp():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    
    # Run both servers concurrently
    await asyncio.gather(
        run_fastapi(),
        run_mcp()
    )



