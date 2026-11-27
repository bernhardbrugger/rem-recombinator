"""
MCP server that exposes the REM Recombinator as a tool — so Claude
(or any other MCP client) can call it.

Start (stdio, for local Claude Desktop integration):
    python mcp_server.py

Register in claude_desktop_config.json:
    {
      "mcpServers": {
        "rem-recombinator": {
          "command": "python",
          "args": ["/absolute/path/to/mcp_server.py"],
          "env": {
            "ANTHROPIC_API_KEY": "...",
            "VOYAGE_API_KEY": "..."
          }
        }
      }
    }
"""
import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import critic
import recombinator
from store import Store

app = Server("rem-recombinator")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="run_rem_cycle",
            description=(
                "Runs a REM recombination cycle: searches for thematically "
                "distant knowledge fragments in the personal store, checks "
                "for hidden connections between them, filters out noise, "
                "and returns only the genuinely interesting insights. Use "
                "this when the user asks about unexpected connections, new "
                "ideas, or patterns in their stored knowledge."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="list_accepted_insights",
            description="Shows previously accepted recombination insights from history.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "run_rem_cycle":
        raw_results = recombinator.run_recombination_cycle()
        if not raw_results:
            return [TextContent(type="text", text="Not enough fragments in the store for a cycle.")]

        accepted = critic.filter_results(raw_results)
        if not accepted:
            return [TextContent(
                type="text",
                text=f"Checked {len(raw_results)} candidate pairs, none passed the critic filter.",
            )]

        lines = [f"{len(accepted)} of {len(raw_results)} candidates accepted:\n"]
        for r in accepted:
            lines.append(
                f"- {r['fragment_a'].text[:80]}... x {r['fragment_b'].text[:80]}...\n"
                f"  -> {r['raw_insight']} (score: {r['score']:.2f})\n"
            )
        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "list_accepted_insights":
        store = Store()
        results = store.accepted_recombinations()
        if not results:
            return [TextContent(type="text", text="No accepted insights stored yet.")]
        lines = []
        for r in results:
            lines.append(f"- {r['insight']} (score: {r['critic_score']:.2f})")
        return [TextContent(type="text", text="\n".join(lines))]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
