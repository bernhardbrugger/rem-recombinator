"""Compatibility shim. Starts the STDIO server in :mod:`rem_recombinator.mcp_server`."""

from rem_recombinator.mcp_server import dispatch, serve, tool_specs, validate_arguments

if __name__ == "__main__":
    serve()
