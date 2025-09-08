#!/bin/bash

# Kill any existing uvicorn processes
pkill -f "uvicorn"

# Start UMLS API (port 8000)
cd /path/to/umls-server/umls_api
nohup python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload > umls_api.log 2>&1 &

# Start MCP Server (port 8001)
cd /path/to/umls-server/mcp-server
nohup python3 -m uvicorn mcp_app:app --host 0.0.0.0 --port 8001 --reload > mcp_server.log 2>&1 &

# Start LLM Integration (port 8002)
cd /path/to/umls-server/llm-integration
nohup python3 -m uvicorn app:app --host 0.0.0.0 --port 8002 --reload > llm_integration.log 2>&1 &

echo "All services started with auto-reload enabled. Check the respective .log files for output." 