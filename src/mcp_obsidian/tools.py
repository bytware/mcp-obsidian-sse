from collections.abc import Sequence
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
import json
import os
from . import obsidian

api_key = os.getenv("OBSIDIAN_API_KEY", "")
if api_key == "":
    raise ValueError(f"OBSIDIAN_API_KEY environment variable required. Working directory: {os.getcwd()}")

TOOL_LIST_FILES_IN_VAULT = "obsidian_list_files_in_vault"
TOOL_LIST_FILES_IN_DIR = "obsidian_list_files_in_dir"
TOOL_LIST_ALL_FILES = "obsidian_list_all_files"

class ToolHandler():
    def __init__(self, tool_name: str):
        self.name = tool_name

    def get_tool_description(self) -> Tool:
        raise NotImplementedError()

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        raise NotImplementedError()
    
class ListFilesInVaultToolHandler(ToolHandler):
    def __init__(self):
        super().__init__(TOOL_LIST_FILES_IN_VAULT)

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Lists all files and directories in the root directory of your Obsidian vault.",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }
        )

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        try:
            api = obsidian.Obsidian(api_key=api_key, protocol='https')
            result = api.list_files_in_vault()
            
            # Ensure result is a list, even if None is returned
            if result is None:
                result = []
                
            return [TextContent(type="text", text=json.dumps({"files": result}))]
        except Exception as e:
            # Log the error
            print(f"Error in ListFilesInVaultToolHandler: {str(e)}")
            # Return empty list instead of raising exception
            return [TextContent(type="text", text=json.dumps({"files": [], "error": str(e)}))]
    
class ListFilesInDirToolHandler(ToolHandler):
    def __init__(self):
        super().__init__(TOOL_LIST_FILES_IN_DIR)

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Lists all files and directories that exist in a specific Obsidian directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dirpath": {
                        "type": "string",
                        "description": "Path to list files from (relative to your vault root). Note that empty directories will not be returned."
                    },
                },
                "required": ["dirpath"]
            }
        )

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        try:
            if "dirpath" not in args:
                raise RuntimeError("dirpath argument missing in arguments")

            api = obsidian.Obsidian(api_key=api_key)
            result = api.list_files_in_dir(args["dirpath"])
            
            # Ensure result is a list, even if None is returned
            if result is None:
                result = []
                
            return [TextContent(type="text", text=json.dumps({"files": result}))]
        except Exception as e:
            # Log the error
            print(f"Error in ListFilesInDirToolHandler: {str(e)}")
            # Return empty list instead of raising exception
            return [TextContent(type="text", text=json.dumps({"files": [], "error": str(e)}))]
    
class GetFileContentsToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_get_file_contents")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Return the content of a single file in your vault.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the relevant file (relative to your vault root).",
                        "format": "path"
                    },
                },
                "required": ["filepath"]
            }
        )

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        try:
            if "filepath" not in args:
                raise RuntimeError("filepath argument missing in arguments")

            api = obsidian.Obsidian(api_key=api_key)
            content = api.get_file_contents(args["filepath"])
            
            # Handle empty content from errors
            if content == "":
                return [TextContent(type="text", text=json.dumps({
                    "content": "",
                    "error": f"File not found or unable to read: {args['filepath']}"
                }))]
                
            return [TextContent(type="text", text=json.dumps({"content": content}))]
        except Exception as e:
            # Log the error
            print(f"Error in GetFileContentsToolHandler: {str(e)}")
            # Return error information instead of raising exception
            return [TextContent(type="text", text=json.dumps({
                "content": "",
                "error": str(e)
            }))]
    
class SearchToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_simple_search")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="""Simple search for documents matching a specified text query across all files in the vault. 
            Use this tool when you want to do a simple text search""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to a simple search for in the vault."
                    },
                    "context_length": {
                        "type": "integer",
                        "description": "How much context to return around the matching string (default: 100)",
                        "default": 100
                    }
                },
                "required": ["query"]
            }
        )

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if "query" not in args:
            raise RuntimeError("query argument missing in arguments")

        context_length = args.get("context_length", 100)
        
        api = obsidian.Obsidian(api_key=api_key)
        results = api.search(args["query"], context_length)
        
        formatted_results = []
        for result in results:
            formatted_matches = []
            for match in result.get('matches', []):
                context = match.get('context', '')
                match_pos = match.get('match', {})
                start = match_pos.get('start', 0)
                end = match_pos.get('end', 0)
                
                formatted_matches.append({
                    'context': context,
                    'match_position': {'start': start, 'end': end}
                })
                
            formatted_results.append({
                'filename': result.get('filename', ''),
                'score': result.get('score', 0),
                'matches': formatted_matches
            })

        return [
            TextContent(
                type="text",
                text=json.dumps(formatted_results, indent=2)
            )
        ]
    
class AppendContentToolHandler(ToolHandler):
   def __init__(self):
       super().__init__("obsidian_append_content")

   def get_tool_description(self):
       return Tool(
           name=self.name,
           description="Append content to a new or existing file in the vault.",
           inputSchema={
               "type": "object",
               "properties": {
                   "filepath": {
                       "type": "string",
                       "description": "Path to the file (relative to vault root)",
                       "format": "path"
                   },
                   "content": {
                       "type": "string",
                       "description": "Content to append to the file"
                   }
               },
               "required": ["filepath", "content"]
           }
       )

   def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
       try:
           if "filepath" not in args or "content" not in args:
               raise RuntimeError("filepath and content arguments required")

           api = obsidian.Obsidian(api_key=api_key)
           result = api.append_content(args.get("filepath", ""), args["content"])
           
           # Check if there was an error
           if isinstance(result, dict) and "error" in result:
               return [
                   TextContent(
                       type="text",
                       text=json.dumps({
                           "success": False,
                           "error": result["error"],
                           "message": f"Failed to append content to {args['filepath']}"
                       })
                   )
               ]

           return [
               TextContent(
                   type="text",
                   text=json.dumps({
                       "success": True,
                       "message": f"Successfully appended content to {args['filepath']}"
                   })
               )
           ]
       except Exception as e:
           # Log the error
           print(f"Error in AppendContentToolHandler: {str(e)}")
           # Return error information instead of raising exception
           return [
               TextContent(
                   type="text",
                   text=json.dumps({
                       "success": False,
                       "error": str(e),
                       "message": f"Failed to append content to {args.get('filepath', '')}"
                   })
               )
           ]
   
class PatchContentToolHandler(ToolHandler):
   def __init__(self):
       super().__init__("obsidian_patch_content")

   def get_tool_description(self):
       return Tool(
           name=self.name,
           description="Insert content into an existing note relative to a heading, block reference, or frontmatter field.",
           inputSchema={
               "type": "object",
               "properties": {
                   "filepath": {
                       "type": "string",
                       "description": "Path to the file (relative to vault root)",
                       "format": "path"
                   },
                   "operation": {
                       "type": "string",
                       "description": "Operation to perform (append, prepend, or replace)",
                       "enum": ["append", "prepend", "replace"]
                   },
                   "target_type": {
                       "type": "string",
                       "description": "Type of target to patch",
                       "enum": ["heading", "block", "frontmatter"]
                   },
                   "target": {
                       "type": "string", 
                       "description": "Target identifier (heading path, block reference, or frontmatter field)"
                   },
                   "content": {
                       "type": "string",
                       "description": "Content to insert"
                   }
               },
               "required": ["filepath", "operation", "target_type", "target", "content"]
           }
       )

   def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
       try:
           required = ["filepath", "operation", "target_type", "target", "content"]
           if not all(key in args for key in required):
               raise RuntimeError(f"Missing required arguments: {', '.join(required)}")

           api = obsidian.Obsidian(api_key=api_key)
           result = api.patch_content(
               args.get("filepath", ""),
               args.get("operation", ""),
               args.get("target_type", ""),
               args.get("target", ""),
               args.get("content", "")
           )
           
           # Check if there was an error
           if isinstance(result, dict) and "error" in result:
               return [
                   TextContent(
                       type="text",
                       text=json.dumps({
                           "success": False,
                           "error": result["error"],
                           "message": f"Failed to patch content in {args['filepath']}"
                       })
                   )
               ]

           return [
               TextContent(
                   type="text",
                   text=json.dumps({
                       "success": True,
                       "message": f"Successfully patched content in {args['filepath']}"
                   })
               )
           ]
       except Exception as e:
           # Log the error
           print(f"Error in PatchContentToolHandler: {str(e)}")
           # Return error information instead of raising exception
           return [
               TextContent(
                   type="text",
                   text=json.dumps({
                       "success": False,
                       "error": str(e),
                       "message": f"Failed to patch content in {args.get('filepath', '')}"
                   })
               )
           ]
   
class ComplexSearchToolHandler(ToolHandler):
   def __init__(self):
       super().__init__("obsidian_complex_search")

   def get_tool_description(self):
       return Tool(
           name=self.name,
           description="""Complex search for documents using a JsonLogic query. 
           Supports standard JsonLogic operators plus 'glob' and 'regexp' for pattern matching. Results must be non-falsy.

           Use this tool when you want to do a complex search, e.g. for all documents with certain tags etc.
           """,
           inputSchema={
               "type": "object",
               "properties": {
                   "query": {
                       "type": "object",
                       "description": "JsonLogic query object. Example: {\"glob\": [\"*.md\", {\"var\": \"path\"}]} matches all markdown files"
                   }
               },
               "required": ["query"]
           }
       )

   def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
       if "query" not in args:
           raise RuntimeError("query argument missing in arguments")

       api = obsidian.Obsidian(api_key=api_key)
       results = api.search_json(args.get("query", ""))

       return [
           TextContent(
               type="text",
               text=json.dumps(results, indent=2)
           )
       ]

class BatchGetFileContentsToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_batch_get_file_contents")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Return the contents of multiple files in your vault, concatenated with headers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepaths": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Path to a file (relative to your vault root)",
                            "format": "path"
                        },
                        "description": "List of file paths to read"
                    },
                },
                "required": ["filepaths"]
            }
        )

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if "filepaths" not in args:
            raise RuntimeError("filepaths argument missing in arguments")

        api = obsidian.Obsidian(api_key=api_key)
        content = api.get_batch_file_contents(args["filepaths"])

        return [
            TextContent(
                type="text",
                text=content
            )
        ]

class ListAllFilesToolHandler(ToolHandler):
    def __init__(self):
        super().__init__(TOOL_LIST_ALL_FILES)

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Lists all files in the Obsidian vault with their complete paths, including those in nested directories.",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }
        )

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        try:
            api = obsidian.Obsidian(api_key=api_key)
            
            # Get all files from root and subdirectories recursively
            all_files = []
            
            # Helper function to recursively explore directories
            def explore_directory(path="", prefix=""):
                # Get all files in the current directory
                dir_entries = api.list_files_in_dir(path) if path else api.list_files_in_vault()
                
                # Process each entry
                for entry in dir_entries:
                    full_path = f"{prefix}{entry}"
                    
                    if entry.endswith('/'):
                        # This is a directory, explore it recursively
                        subdir_path = path + "/" + entry[:-1] if path else entry[:-1]
                        explore_directory(subdir_path, full_path)
                    else:
                        # This is a file, add it to the list
                        all_files.append(full_path)
            
            # Start recursive exploration from root
            explore_directory()
            
            # Sort the list for consistency
            all_files.sort()
            
            return [TextContent(type="text", text=json.dumps({"files": all_files}))]
        except Exception as e:
            # Log the error
            print(f"Error in ListAllFilesToolHandler: {str(e)}")
            # Return empty list instead of raising exception
            return [TextContent(type="text", text=json.dumps({"files": [], "error": str(e)}))]