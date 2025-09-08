# UMLS MCP Server

A Model Context Protocol (MCP) server that provides access to the Unified Medical Language System (UMLS) database through Claude Desktop and other MCP-compatible clients.

This server uses the official `mcp` Python package and follows the MCP specification for seamless integration with Claude Desktop via stdio transport.

## Features

This MCP server provides the following tools for querying UMLS data:

- **search_terms**: Search for medical terms in UMLS database by ontology
- **search_cui**: Search for CUIs (Concept Unique Identifiers) by term
- **get_cui_info**: Get detailed information about a specific CUI
- **get_cui_ancestors**: Get all ancestor CUIs in the hierarchy
- **get_cui_depth**: Get the depth of a CUI in the hierarchical structure
- **find_lowest_common_ancestor**: Find the lowest common ancestor (LCA) of two CUIs
- **wu_palmer_similarity**: Compute Wu-Palmer similarity between two CUIs
- **get_hpo_term**: Get HPO (Human Phenotype Ontology) term and code from a CUI

## Prerequisites

1. **UMLS API Server**: The MCP server requires the UMLS API to be running on port 8000. Make sure you have the UMLS API component of this project running first.

2. **Python 3.8+**: Required for running the MCP server.

3. **Conda Environment**: Recommended for managing dependencies and ensuring compatibility.

4. **Claude Desktop**: For the full experience, you'll want Claude Desktop installed to interact with the MCP server.

## Installation

1. **Create and Activate Conda Environment**:
   ```bash
   conda create -n umls-server python=3.11
   conda activate umls-server
   ```

2. **Install Dependencies**:
   ```bash
   cd mcp-server
   pip install -r requirements.txt
   ```

   The key dependencies include:
   - `mcp`: Official Model Context Protocol Python package
   - `httpx`: For HTTP requests to the UMLS API
   - `python-dotenv`: For environment variable management

3. **Configure Environment Variables** (optional):
   Create a `.env` file in the `mcp-server` directory:
   ```env
   UMLS_API_URL=http://localhost:8000
   ```

## Running the MCP Server

### With Claude Desktop (Recommended)

1. **Update Claude Desktop Configuration**:
   
   Edit your Claude Desktop configuration file. The location depends on your OS:
   
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. **Add the UMLS MCP Server**:
   
   ```json
   {
     "mcpServers": {
       "umls-server": {
         "command": "/path/to/conda/envs/umls-server/bin/python",
         "args": [
           "/absolute/path/to/your/umls-server/mcp-server/umls_mcp_server.py"
         ],
         "env": {
           "UMLS_API_URL": "http://localhost:8000"
         }
       }
     }
   }
   ```

   **Important**: 
   - Replace `/path/to/conda/envs/umls-server/bin/python` with the actual path to your conda environment's Python interpreter
   - Replace `/absolute/path/to/your/umls-server/` with the actual path to your project directory
   - You can find your conda environment path with: `conda info --envs`

3. **Restart Claude Desktop**: Close and reopen Claude Desktop for the changes to take effect.

### Standalone Testing

To test the MCP server independently in stdio mode:

```bash
conda activate umls-server
cd mcp-server
python3 umls_mcp_server.py
```

The server will start in stdio mode and wait for MCP protocol messages.

## Usage Examples

Once connected to Claude Desktop, you can use natural language to interact with the UMLS database:

### Searching for Medical Terms
```
Search for diabetes in the HPO ontology
```

### Finding CUI Information
```
What is CUI C0001699?
```

### Similarity Analysis
```
What is the Wu-Palmer similarity between asthma (C0004096) and pneumonia (C0032285)?
```

### Hierarchical Analysis
```
Find the ancestors of CUI C0001699
```

### HPO Lookups
```
Get the HPO term for CUI C0001699
```

## Configuration Options

### Environment Variables

- `UMLS_API_URL`: URL of the UMLS API server (default: `http://localhost:8000`)

### Claude Desktop Configuration

The Claude Desktop configuration supports these options:

- `command`: Path to the Python interpreter (use conda environment path)
- `args`: Path to the MCP server script
- `env`: Environment variables to set for the server

Example configuration for different setups:

**Using Conda (Recommended)**:
```json
{
  "mcpServers": {
    "umls-server": {
      "command": "/Users/user/miniconda3/envs/umls-server/bin/python",
      "args": [
        "/Users/user/projects/umls-server/mcp-server/umls_mcp_server.py"
      ],
      "env": {
        "UMLS_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

**Using System Python**:
```json
{
  "mcpServers": {
    "umls-server": {
      "command": "python3",
      "args": [
        "/path/to/umls-server/mcp-server/umls_mcp_server.py"
      ],
      "env": {
        "UMLS_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

## Troubleshooting

### Server Won't Start
1. Ensure the UMLS API is running on port 8000
2. Check that all dependencies are installed in the correct conda environment: `pip list | grep mcp`
3. Verify the Python path in your Claude Desktop configuration: `which python` in your conda environment
4. Ensure you've activated the conda environment when testing: `conda activate umls-server`

### Claude Desktop Can't Connect
1. Check the absolute path to `umls_mcp_server.py` in your configuration
2. Verify the conda environment Python path: `conda info --envs`
3. Ensure the script is executable: `chmod +x umls_mcp_server.py`
4. Restart Claude Desktop after configuration changes
5. Check Claude Desktop logs for specific error messages

### API Connection Issues
1. Verify the UMLS API is accessible at the configured URL: `curl http://localhost:8000/terms?search=test`
2. Check firewall settings if using a remote UMLS API
3. Review server logs for specific error messages
4. Ensure the conda environment has all required packages: `pip install -r requirements.txt`

### Import/Dependency Issues
1. Ensure you've created and activated the conda environment: `conda activate umls-server`
2. Install all requirements: `pip install -r requirements.txt`
3. Verify the `mcp` package is installed: `python -c "import mcp; print('MCP package working')"`
4. Check that httpx is available: `python -c "import httpx; print('httpx working')"`

### Testing the MCP Server

You can test the server independently using the MCP inspector (if available):
```bash
conda activate umls-server
npx @modelcontextprotocol/inspector python3 umls_mcp_server.py
```

Or test basic functionality:
```bash
conda activate umls-server
python3 umls_mcp_server.py
# The server will wait for stdio input in MCP format
```

## Development

### Adding New Tools

To add a new tool to the MCP server:

1. **Add tool definition** in the `list_tools()` function:
   ```python
   Tool(
       name="your_tool_name",
       description="Description of what your tool does",
       inputSchema={
           "type": "object",
           "properties": {
               "param1": {
                   "type": "string",
                   "description": "Description of parameter"
               }
           },
           "required": ["param1"]
       }
   )
   ```

2. **Add tool handler** in the `call_tool()` function:
   ```python
   elif name == "your_tool_name":
       param1 = arguments["param1"]
       result = await call_umls_api(f"/your-endpoint/{param1}")
       return [
           TextContent(
               type="text",
               text=f"Your formatted result: {result}"
           ).model_dump()
       ]
   ```

3. **Test the new tool** with Claude Desktop after restarting

### Code Structure

- `umls_mcp_server.py`: Main MCP server implementation
- `list_tools()`: Defines all available tools and their schemas
- `call_tool()`: Handles tool execution and API calls
- `call_umls_api()`: Helper function for making HTTP requests to the UMLS API

## Supported MCP Protocol Features

- ✅ Tools (all UMLS functions available as tools)
- ✅ STDIO transport (for Claude Desktop)
- ✅ Error handling and logging
- ✅ Environment variable configuration
- ✅ Async/await support
- ⚠️ SSE transport (not implemented - Claude Desktop uses stdio)
- ⚠️ Resources (not needed for this use case)
- ⚠️ Prompts (not needed for this use case)

## Architecture

```
Claude Desktop ↔ (stdio/MCP) ↔ MCP Server ↔ (HTTP) ↔ UMLS API ↔ UMLS Database
```

The MCP server acts as a protocol bridge between Claude Desktop (which speaks MCP over stdio) and the UMLS REST API (which speaks HTTP/JSON).

## Contributing

When making changes to the MCP server:

1. Update tool definitions in the `list_tools()` function
2. Add corresponding handlers in the `call_tool()` function
3. Ensure all returned content is serialized using `.model_dump()`
4. Test with both standalone and Claude Desktop modes
5. Update this README with new features

## License

MIT License - see the main project README for details. 