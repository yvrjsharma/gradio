# Gradio apps as MCP Servers / Tools
Tags: MCP, TOOL, LLM, SERVER
In this guide, we will describe how to launch your Gradio app so that it can be used as an MCP Server. Punchline: it's as simple as setting `mcp_server=True` in `.launch()`.

## What is an MCP Server

## Automatically Creating an MCP Server with Gradio

## Launching an MCP Server Manually with the Gradio Clients

For a more fine-grained control, you might want to manually create an MCP Server that interfaces with hosted Gradio apps. This approach is useful when you want to:

- Combine multiple Gradio apps into a single MCP server
- Customize how your tools are presented to LLMs
- Add specialized logic around tool execution

Here's an example of creating a custom MCP server that connects to various Gradio apps hosted on [HuggingFace Spaces](https://huggingface.co/spaces):

```python
from mcp.server.fastmcp import FastMCP
from gradio_client import Client
import sys
import io
import json 

# Initialize FastMCP server
mcp = FastMCP("gradio-spaces")

# Dictionary to store Gradio clients
clients = {}

def get_client(space_id: str) -> Client:
    """Get or create a Gradio client for the specified space."""
    if space_id not in clients:
        clients[space_id] = Client(space_id)
    return clients[space_id]


@mcp.tool()
async def generate_image(prompt: str, space_id: str = "ysharma/SanaSprint") -> str:
    """Generate an image using Flux.
    
    Args:
        prompt: Text prompt describing the image to generate
        space_id: HuggingFace Space ID to use 
    """
    client = get_client(space_id)
    result = client.predict(
            prompt=prompt,
            model_size="1.6B",
            seed=0,
            randomize_seed=True,
            width=1024,
            height=1024,
            guidance_scale=4.5,
            num_inference_steps=2,
            api_name="/infer"
    )
    return result


@mcp.tool()
async def run_dia_tts(prompt: str, space_id: str = "ysharma/Dia-1.6B") -> str:
    """Text-to-Speech Synthesis.
    
    Args:
        prompt: Text prompt describing the conversation between speakers S1, S2
        space_id: HuggingFace Space ID to use 
    """
    client = get_client(space_id)
    result = client.predict(
            text_input=f"""{prompt}""",
            audio_prompt_input=None, 
            max_new_tokens=3072,
            cfg_scale=3,
            temperature=1.3,
            top_p=0.95,
            cfg_filter_top_k=30,
            speed_factor=0.94,
            api_name="/generate_audio"
    )
    return result


if __name__ == "__main__":
    # Ensure stdout uses UTF-8 encoding
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Initialize and run the server
    mcp.run(transport='stdio')

```

This server exposes two tools:
1. `run_dia_tts` - Generates a conversation for the given transcript in the form of `[S1]first-sentence. [S2]second-sentence. [S1]...`
2. `generate_image` - Generates images using a fast text-to-image model

To use this MCP Server with Claude Desktop (as MCP Client):

1. Save the code to a file (e.g., `gradio_mcp_server.py`)
2. Install the required dependencies: `pip install mcp gradio-client`
3. Configure Claude Desktop to use your server by editing the configuration file at `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
    "mcpServers": {
        "gradio-spaces": {
            "command": "python",
            "args": [
                "/absolute/path/to/gradio_mcp_server.py"
            ]
        }
    }
}
```

4. Restart Claude Desktop

Now, when you ask Claude about generating an image or transcribing audio, it can use your Gradio-powered tools to accomplish these tasks.

