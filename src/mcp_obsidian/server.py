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
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

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

# Pydantic models for request/response
class ListFilesResponse(BaseModel):
    files: List[str] = Field(description="List of files in the vault or directory")

class FileContentResponse(BaseModel):
    content: str = Field(description="Content of the file")

class DirPathRequest(BaseModel):
    dirpath: str = Field(description="Path to list files from (relative to your vault root)")

class FilePathRequest(BaseModel):
    filepath: str = Field(description="Path to the file (relative to vault root)")

class SearchRequest(BaseModel):
    query: str = Field(description="Text to search for in the vault")
    context_length: Optional[int] = Field(default=100, description="How much context to return around the matching string")

class SearchMatch(BaseModel):
    context: str = Field(description="Context around the match")
    match_position: Dict[str, int] = Field(description="Start and end positions of the match")

class SearchResult(BaseModel):
    filename: str = Field(description="Name of the file containing matches")
    score: float = Field(description="Search relevance score")
    matches: List[SearchMatch] = Field(description="List of matches in this file")

class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(description="Search results across files")

class AppendContentRequest(BaseModel):
    filepath: str = Field(description="Path to the file (relative to vault root)")
    content: str = Field(description="Content to append to the file")

class PatchContentRequest(BaseModel):
    filepath: str = Field(description="Path to the file (relative to vault root)")
    operation: str = Field(description="Operation to perform (append, prepend, or replace)")
    target_type: str = Field(description="Type of target to patch")
    target: str = Field(description="Target identifier (heading path, block reference, or frontmatter field)")
    content: str = Field(description="Content to insert")

class BatchFileContentsRequest(BaseModel):
    filepaths: List[str] = Field(description="List of file paths to get contents for")

class BatchFileContentsResponse(BaseModel):
    contents: Dict[str, str] = Field(description="Map of filepath to file contents")

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

def get_openapi_schema():
    """Generate a custom OpenAPI schema with proper schema references and server URLs."""
    if not fastapi_app.openapi_schema:
        openapi_schema = fastapi_app.openapi()
        
        # Fix schema references by removing duplicate prefixes
        if "components" in openapi_schema and "schemas" in openapi_schema["components"]:
            new_schemas = {}
            for key, schema in openapi_schema["components"]["schemas"].items():
                # Remove duplicate obsidian_ prefixes
                new_key = key.replace("obsidian_" * 14, "obsidian_")
                new_schemas[new_key] = schema
            openapi_schema["components"]["schemas"] = new_schemas

            # Update references in paths
            for path in openapi_schema["paths"].values():
                for method in path.values():
                    if "requestBody" in method:
                        _fix_refs(method["requestBody"])
                    if "responses" in method:
                        for response in method["responses"].values():
                            if "content" in response:
                                _fix_refs(response["content"])

        # Set proper server URLs
        openapi_schema["servers"] = [
            {
                "url": f"http://localhost:{port}",
                "description": "Local development server"
            }
        ]
        
        fastapi_app.openapi_schema = openapi_schema
    
    return fastapi_app.openapi_schema

def _fix_refs(obj):
    """Recursively fix schema references by removing duplicate prefixes."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "$ref" and isinstance(value, str):
                if "obsidian_" * 14 in value:
                    obj[key] = value.replace("obsidian_" * 14, "obsidian_")
            elif isinstance(value, (dict, list)):
                _fix_refs(value)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                _fix_refs(item)

fastapi_app = FastAPI(
    title="MCP Obsidian",
    description="API server to work with Obsidian via the remote REST plugin",
    version="0.2.1",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Override the default OpenAPI schema with our custom one
fastapi_app.openapi = get_openapi_schema

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@fastapi_app.post("/tool/{tool_name}")
async def call_tool_http(tool_name: str, arguments: dict):
    try:
        # Get the tool handler directly
        tool_handler = get_tool_handler(tool_name)
        if not tool_handler:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        # Run the tool directly using the handler
        result = tool_handler.run_tool(arguments)
        
        # Pass through the raw response for all tools
        return json.loads(result[0].text)
        
    except Exception as e:
        logger.error(str(e))
        # Send error event through SSE if client_id is provided
        if "client_id" in arguments:
            client_id = arguments["client_id"]
            event_data = {
                "tool": tool_name,
                "arguments": arguments,
                "status": "error",
                "error": str(e)
            }
            asyncio.create_task(sse_manager.send_event(client_id, "tool_error", event_data))
        return {"error": str(e)}

@fastapi_app.get("/tools")
async def list_tools_http():
    tools = await list_tools()
    return {
        "status": "success",
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": [
                    {"name": param.name, "type": param.type, "description": param.description}
                    for param in tool.parameters
                ] if hasattr(tool, "parameters") else []
            }
            for tool in tools
        ]
    }

@fastapi_app.get("/vault/files", response_model=ListFilesResponse, tags=["Files"])
async def list_files_in_vault():
    """Lists all files and directories in the root directory of your Obsidian vault."""
    tool_handler = tools.ListFilesInVaultToolHandler()
    result = tool_handler.run_tool({})
    return {"files": json.loads(result[0].text)}

@fastapi_app.post("/dir/files", response_model=ListFilesResponse, tags=["Files"])
async def list_files_in_dir(request: DirPathRequest):
    """Lists all files and directories that exist in a specific Obsidian directory."""
    tool_handler = tools.ListFilesInDirToolHandler()
    result = tool_handler.run_tool({"dirpath": request.dirpath})
    return json.loads(result[0].text)

@fastapi_app.post("/file/contents", response_model=FileContentResponse, tags=["Files"])
async def get_file_contents(request: FilePathRequest):
    """Return the content of a single file in your vault."""
    tool_handler = tools.GetFileContentsToolHandler()
    result = tool_handler.run_tool({"filepath": request.filepath})
    return json.loads(result[0].text)

@fastapi_app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """Simple search for documents matching a specified text query across all files in the vault."""
    tool_handler = tools.SearchToolHandler()
    result = tool_handler.run_tool({
        "query": request.query,
        "context_length": request.context_length
    })
    return {"results": json.loads(result[0].text)}

@fastapi_app.post("/file/append", tags=["Files"])
async def append_content(request: AppendContentRequest):
    """Append content to a new or existing file in the vault."""
    tool_handler = tools.AppendContentToolHandler()
    result = tool_handler.run_tool({
        "filepath": request.filepath,
        "content": request.content
    })
    return {"message": result[0].text}

@fastapi_app.post("/file/patch", tags=["Files"])
async def patch_content(request: PatchContentRequest):
    """Insert content into an existing note relative to a heading, block reference, or frontmatter field."""
    tool_handler = tools.PatchContentToolHandler()
    result = tool_handler.run_tool({
        "filepath": request.filepath,
        "operation": request.operation,
        "target_type": request.target_type,
        "target": request.target,
        "content": request.content
    })
    return {"message": result[0].text}

@fastapi_app.post("/search/complex", response_model=SearchResponse, tags=["Search"])
async def complex_search(request: SearchRequest):
    """Complex search for documents matching a specified text query across all files in the vault."""
    tool_handler = tools.ComplexSearchToolHandler()
    result = tool_handler.run_tool({
        "query": request.query,
        "context_length": request.context_length
    })
    return {"results": json.loads(result[0].text)}

@fastapi_app.post("/file/batch-contents", response_model=BatchFileContentsResponse, tags=["Files"])
async def batch_get_file_contents(request: BatchFileContentsRequest):
    """Return the content of multiple files in your vault."""
    tool_handler = tools.BatchGetFileContentsToolHandler()
    result = tool_handler.run_tool({"filepaths": request.filepaths})
    return json.loads(result[0].text)

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



