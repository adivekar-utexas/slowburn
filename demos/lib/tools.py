"""
Pre-built tools for SlowBurn agent demos.

All file operations are sandboxed to a workspace directory.
The sandbox is enforced by resolving all paths against the workspace root
and rejecting any path that escapes it (via symlinks, .., etc.).

Tools:
- search_web: Real web search via DuckDuckGo (no API key needed)
- read_file: Read a file within the sandbox
- write_file: Write/create a file within the sandbox
- list_dir: List files in a directory within the sandbox

Each tool function returns a JSON string (the format litellm tool results expect).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SandboxViolation(PermissionError):
    """Raised when a tool tries to access a path outside the workspace."""


def _resolve_sandboxed_path(workspace: Path, relative_path: str) -> Path:
    """Resolve a path ensuring it stays within the workspace.

    Raises SandboxViolation if the resolved path escapes the workspace.
    """
    workspace = workspace.resolve()
    candidate = (workspace / relative_path).resolve()
    if not str(candidate).startswith(str(workspace)):
        raise SandboxViolation(
            f"Path '{relative_path}' resolves to '{candidate}' which is outside "
            f"workspace '{workspace}'. Refusing to proceed."
        )
    return candidate


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo. Returns JSON with search results."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return json.dumps(
                {"error": "Neither ddgs nor duckduckgo-search is installed. Run: pip install ddgs"}
            )

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        formatted = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
        return json.dumps({"results": formatted}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Search failed: {type(e).__name__}: {e}"})


def read_file(path: str, workspace: Path = None) -> str:
    """Read a file within the sandbox workspace."""
    if workspace is None:
        return json.dumps({"error": "No workspace configured."})
    try:
        resolved = _resolve_sandboxed_path(workspace, path)
    except SandboxViolation as e:
        return json.dumps({"error": str(e)})

    if not resolved.exists():
        return json.dumps({"error": f"File not found: {path}"})
    if not resolved.is_file():
        return json.dumps({"error": f"Not a file: {path}"})

    try:
        content = resolved.read_text(encoding="utf-8")
        if len(content) > 50_000:
            content = content[:50_000] + f"\n\n[TRUNCATED — file is {len(content)} chars]"
        return json.dumps({"path": path, "content": content})
    except Exception as e:
        return json.dumps({"error": f"Read failed: {e}"})


def write_file(path: str, content: str, workspace: Path = None) -> str:
    """Write (create or overwrite) a file within the sandbox workspace."""
    if workspace is None:
        return json.dumps({"error": "No workspace configured."})
    try:
        resolved = _resolve_sandboxed_path(workspace, path)
    except SandboxViolation as e:
        return json.dumps({"error": str(e)})

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return json.dumps({"path": path, "bytes_written": len(content)})
    except Exception as e:
        return json.dumps({"error": f"Write failed: {e}"})


def list_dir(path: str = ".", workspace: Path = None) -> str:
    """List files in a directory within the sandbox workspace."""
    if workspace is None:
        return json.dumps({"error": "No workspace configured."})
    try:
        resolved = _resolve_sandboxed_path(workspace, path)
    except SandboxViolation as e:
        return json.dumps({"error": str(e)})

    if not resolved.is_dir():
        return json.dumps({"error": f"Not a directory: {path}"})

    try:
        entries = []
        for item in sorted(resolved.iterdir()):
            rel = item.relative_to(workspace.resolve())
            entries.append(
                {
                    "name": item.name,
                    "path": str(rel),
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                }
            )
        return json.dumps({"path": path, "entries": entries})
    except Exception as e:
        return json.dumps({"error": f"List failed: {e}"})


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web using DuckDuckGo. Returns a JSON list "
                "of results with title, URL, and snippet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file. Returns the file content as a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "File path relative to workspace root, e.g. 'report.md' or 'src/main.py'."
                        ),
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write a file to disk. Takes exactly two arguments: 'path' and 'content'. "
                "The 'content' argument must be a SINGLE string containing the ENTIRE file content. "
                "Do NOT split the content across multiple arguments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root, e.g. 'report.md'.",
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "The complete file content as a single string. "
                            "Include all text, headings, and sections in "
                            "this one string."
                        ),
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and subdirectories at a path in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to workspace root. Use '.' for the root.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


TOOL_FUNCTIONS = {
    "search_web": search_web,
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
}


def execute_tool_call(
    tool_name: str,
    tool_args: Dict[str, Any],
    workspace: Path,
) -> str:
    """Execute a tool call, injecting the workspace for sandboxed tools.

    Args:
        tool_name: Name of the tool to execute.
        tool_args: Arguments from the LLM's tool_call.
        workspace: Sandbox directory for file operations.

    Returns:
        JSON string with the tool result.
    """
    if tool_name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    fn = TOOL_FUNCTIONS[tool_name]
    if tool_name in ("read_file", "write_file", "list_dir"):
        if not isinstance(tool_args, dict):
            return json.dumps(
                {"error": f"Invalid arguments type: expected dict, got {type(tool_args).__name__}"}
            )
        tool_args = {**tool_args, "workspace": workspace}

    try:
        return fn(**tool_args)
    except TypeError as e:
        return json.dumps({"error": f"Invalid arguments for {tool_name}: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
