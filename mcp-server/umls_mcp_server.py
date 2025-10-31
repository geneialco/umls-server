#!/usr/bin/env python3
"""
UMLS MCP Server - A Model Context Protocol server for accessing UMLS database.

This server provides tools for querying the Unified Medical Language System (UMLS)
through MCP protocol, supporting both stdio and SSE connections.
"""

import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional
import json

import httpx
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    EmbeddedResource,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UMLS_API_URL = os.getenv("UMLS_API_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = 30.0
EXTENDED_TIMEOUT = 600.0  # For complex operations like Wu-Palmer similarity

# Initialize the MCP server
server = Server("umls-mcp-server")

async def call_umls_api(endpoint: str, params: Dict[str, Any] = None, timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Call the UMLS API with the given endpoint and parameters."""
    url = f"{UMLS_API_URL}{endpoint}"
    
    # Use extended timeout for relationship endpoints
    if "relationships" in endpoint:
        timeout = EXTENDED_TIMEOUT
    
    logger.info(f"Calling UMLS API: {url} with params: {params} (timeout: {timeout}s)")
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(url, params=params or {})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e}, Response: {e.response.text}")
            raise Exception(f"API error: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise Exception(f"Error connecting to UMLS API: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected error: {str(e)}")

@server.list_tools()
async def list_tools():
    tools = [
        Tool(
            name="search_terms",
            description="Search for medical terms in UMLS database by ontology",
            inputSchema={
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "The search term to look for"
                    },
                    "ontology": {
                        "type": "string",
                        "description": "The ontology to search in (e.g., HPO, NCI, SNOMEDCT_US)",
                        "default": "HPO"
                    }
                },
                "required": ["search"]
            }
        ),
        Tool(
            name="search_cui",
            description="Search for CUIs (Concept Unique Identifiers) by term",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search term to find matching CUIs"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_cui_info",
            description="Get detailed information about a specific CUI",
            inputSchema={
                "type": "object",
                "properties": {
                    "cui": {
                        "type": "string",
                        "description": "The CUI identifier (e.g., C0001699)"
                    }
                },
                "required": ["cui"]
            }
        ),
        Tool(
            name="get_cui_ancestors",
            description="Get all ancestor CUIs in the hierarchy",
            inputSchema={
                "type": "object",
                "properties": {
                    "cui": {
                        "type": "string",
                        "description": "The CUI identifier to get ancestors for"
                    }
                },
                "required": ["cui"]
            }
        ),
        Tool(
            name="get_cui_depth",
            description="Get the depth of a CUI in the hierarchical structure",
            inputSchema={
                "type": "object",
                "properties": {
                    "cui": {
                        "type": "string",
                        "description": "The CUI identifier to get depth for"
                    }
                },
                "required": ["cui"]
            }
        ),
        Tool(
            name="find_lowest_common_ancestor",
            description="Find the lowest common ancestor (LCA) of two CUIs",
            inputSchema={
                "type": "object",
                "properties": {
                    "cui1": {
                        "type": "string",
                        "description": "First CUI identifier"
                    },
                    "cui2": {
                        "type": "string",
                        "description": "Second CUI identifier"
                    }
                },
                "required": ["cui1", "cui2"]
            }
        ),
        Tool(
            name="wu_palmer_similarity",
            description="Compute Wu-Palmer similarity between two CUIs based on hierarchical structure",
            inputSchema={
                "type": "object",
                "properties": {
                    "cui1": {
                        "type": "string",
                        "description": "First CUI identifier"
                    },
                    "cui2": {
                        "type": "string",
                        "description": "Second CUI identifier"
                    }
                },
                "required": ["cui1", "cui2"]
            }
        ),
        Tool(
            name="get_hpo_term",
            description="Get HPO (Human Phenotype Ontology) term and code from a CUI",
            inputSchema={
                "type": "object",
                "properties": {
                    "cui": {
                        "type": "string",
                        "description": "The CUI identifier to get HPO information for"
                    }
                },
                "required": ["cui"]
            }
        ),
        Tool(
            name="get_relationships",
            description="Get direct relationships between two CUIs from the MRREL table",
            inputSchema={
                "type": "object",
                "properties": {
                    "cui1": {
                        "type": "string",
                        "description": "First CUI identifier"
                    },
                    "cui2": {
                        "type": "string",
                        "description": "Second CUI identifier"
                    },
                    "sab": {
                        "type": "string",
                        "description": "Source vocabulary filter (e.g., 'SNOMEDCT_US', 'HPO'). Optional."
                    }
                },
                "required": ["cui1", "cui2"]
            }
        ),
        Tool(
            name="get_indirect_relationships",
            description="Get indirect relationships between two CUIs through intermediate concepts",
            inputSchema={
                "type": "object",
                "properties": {
                    "cui1": {
                        "type": "string",
                        "description": "First CUI identifier"
                    },
                    "cui2": {
                        "type": "string",
                        "description": "Second CUI identifier"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum path length to search (1-3 recommended)",
                        "default": 2
                    },
                    "sab": {
                        "type": "string",
                        "description": "Source vocabulary filter (e.g., 'SNOMEDCT_US', 'HPO'). Optional."
                    }
                },
                "required": ["cui1", "cui2"]
            }
        )
        ,
        Tool(
            name="get_rxnorm_indications",
            description="Get RxNorm medications indicated to treat a disease CUI",
            inputSchema={
                "type": "object",
                "properties": {
                    "cui": {"type": "string", "description": "Disease CUI"},
                    "limit": {"type": "integer", "description": "Max results", "default": 50}
                },
                "required": ["cui"]
            }
        ),
        Tool(
            name="get_rxnorm_related",
            description="Get broader RxNorm medications related to a disease CUI",
            inputSchema={
                "type": "object",
                "properties": {
                    "cui": {"type": "string", "description": "Disease CUI"},
                    "limit": {"type": "integer", "description": "Max results", "default": 50}
                },
                "required": ["cui"]
            }
        )
    ]
    return tools

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """Handle tool calls by routing to appropriate UMLS API endpoints."""
    
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    
    try:
        if name == "search_terms":
            search = arguments["search"]
            ontology = arguments.get("ontology", "HPO")
            
            result = await call_umls_api("/terms", {
                "search": search,
                "ontology": ontology
            })
            
            return [
                TextContent(
                    type="text",
                    text=f"Found {len(result.get('results', []))} medical terms for '{search}' in {ontology} ontology:\n\n" + 
                         "\n".join([
                             f"• {item['code']}: {item['term']}\n  Description: {item.get('description', 'N/A')}"
                             for item in result.get('results', [])
                         ])
                ).model_dump()
            ]
            
        elif name == "search_cui":
            query = arguments["query"]
            
            result = await call_umls_api("/cuis", {"query": query})
            
            return [
                TextContent(
                    type="text",
                    text=f"Found {len(result.get('cuis', []))} CUIs for '{query}':\n\n" + 
                         "\n".join([
                             f"• {item['cui']}: {item['name']}"
                             for item in result.get('cuis', [])
                         ])
                ).model_dump()
            ]
            
        elif name == "get_cui_info":
            cui = arguments["cui"]
            
            result = await call_umls_api(f"/cuis/{cui}")
            
            return [
                TextContent(
                    type="text",
                    text=f"CUI Information:\n• CUI: {result['cui']}\n• Name: {result['name']}"
                ).model_dump()
            ]
            
        elif name == "get_cui_ancestors":
            cui = arguments["cui"]
            
            result = await call_umls_api(f"/cuis/{cui}/ancestors")
            
            ancestors = result.get('ancestors', [])
            return [
                TextContent(
                    type="text",
                    text=f"Found {len(ancestors)} ancestors for CUI {cui}:\n\n" + 
                         "\n".join([f"• {ancestor}" for ancestor in ancestors])
                ).model_dump()
            ]
            
        elif name == "get_cui_depth":
            cui = arguments["cui"]
            
            result = await call_umls_api(f"/cuis/{cui}/depth")
            
            return [
                TextContent(
                    type="text",
                    text=f"CUI {cui} has depth {result['depth']} in the hierarchy"
                ).model_dump()
            ]
            
        elif name == "find_lowest_common_ancestor":
            cui1 = arguments["cui1"]
            cui2 = arguments["cui2"]
            
            result = await call_umls_api(f"/cuis/{cui1}/{cui2}/lca", timeout=EXTENDED_TIMEOUT)
            
            return [
                TextContent(
                    type="text",
                    text=f"Lowest Common Ancestor Analysis:\n" +
                         f"• CUI 1: {cui1}\n" +
                         f"• CUI 2: {cui2}\n" +
                         f"• LCA: {result['lca']}\n" +
                         f"• LCA Depth: {result['depth']}"
                ).model_dump()
            ]
            
        elif name == "wu_palmer_similarity":
            cui1 = arguments["cui1"]
            cui2 = arguments["cui2"]
            
            result = await call_umls_api(f"/cuis/{cui1}/{cui2}/similarity/wu-palmer", timeout=EXTENDED_TIMEOUT)
            
            return [
                TextContent(
                    type="text",
                    text=f"Wu-Palmer Similarity Analysis:\n" +
                         f"• CUI 1: {cui1} (depth: {result['depth_c1']})\n" +
                         f"• CUI 2: {cui2} (depth: {result['depth_c2']})\n" +
                         f"• Lowest Common Ancestor: {result['lca']} (depth: {result['depth_lca']})\n" +
                         f"• Similarity Score: {result['similarity']:.4f}"
                ).model_dump()
            ]
            
        elif name == "get_hpo_term":
            cui = arguments["cui"]
            
            result = await call_umls_api(f"/cuis/{cui}/hpo")
            
            return [
                TextContent(
                    type="text",
                    text=f"HPO Information for CUI {cui}:\n" +
                         f"• HPO Code: {result['hpo_code']}\n" +
                         f"• HPO Term: {result['hpo_term']}"
                ).model_dump()
            ]
            
        elif name == "get_relationships":
            cui1 = arguments["cui1"]
            cui2 = arguments["cui2"]
            sab = arguments.get("sab")
            
            params = {}
            if sab:
                params["sab"] = sab
            
            result = await call_umls_api(f"/cuis/{cui1}/{cui2}/relationships", params, timeout=EXTENDED_TIMEOUT)
            
            relationships = result.get('relationships', [])
            if not relationships:
                return [
                    TextContent(
                        type="text",
                        text=f"No direct relationships found between CUIs {cui1} and {cui2}" + 
                             (f" in {sab}" if sab else "")
                    ).model_dump()
                ]
            
            relationship_text = f"Found {len(relationships)} direct relationships between CUIs {cui1} and {cui2}:\n\n"
            for i, rel in enumerate(relationships, 1):
                relationship_text += f"{i}. {rel['cui1_name']} → {rel['cui2_name']}\n"
                relationship_text += f"   Relationship: {rel['rel']}"
                if rel.get('rela'):
                    relationship_text += f" ({rel['rela']})"
                relationship_text += f"\n   Source: {rel['sab']}\n\n"
            
            return [
                TextContent(
                    type="text",
                    text=relationship_text
                ).model_dump()
            ]
            
        elif name == "get_indirect_relationships":
            cui1 = arguments["cui1"]
            cui2 = arguments["cui2"]
            max_depth = arguments.get("max_depth", 2)
            sab = arguments.get("sab")
            
            params = {"max_depth": max_depth}
            if sab:
                params["sab"] = sab
            
            result = await call_umls_api(f"/cuis/{cui1}/{cui2}/relationships/indirect", params, timeout=EXTENDED_TIMEOUT)
            
            indirect_rels = result.get('indirect_relationships', [])
            if not indirect_rels:
                return [
                    TextContent(
                        type="text",
                        text=f"No indirect relationships found between CUIs {cui1} and {cui2} through intermediate concepts" + 
                             (f" in {sab}" if sab else "")
                    ).model_dump()
                ]
            
            indirect_text = f"Found {len(indirect_rels)} indirect relationship paths between CUIs {cui1} and {cui2}:\n\n"
            for i, path in enumerate(indirect_rels, 1):
                indirect_text += f"{i}. Path: {path['path']}\n"
                indirect_text += f"   Intermediate: {path['intermediate_name']} ({path['intermediate_cui']})\n"
                
                # Step 1
                step1 = path['step1']
                indirect_text += f"   Step 1: {step1['from_name']} → {step1['to_name']}\n"
                indirect_text += f"           Relationship: {step1['rel']}"
                if step1.get('rela'):
                    indirect_text += f" ({step1['rela']})"
                indirect_text += f" (Source: {step1['sab']})\n"
                
                # Step 2
                step2 = path['step2']
                indirect_text += f"   Step 2: {step2['from_name']} → {step2['to_name']}\n"
                indirect_text += f"           Relationship: {step2['rel']}"
                if step2.get('rela'):
                    indirect_text += f" ({step2['rela']})"
                indirect_text += f" (Source: {step2['sab']})\n\n"
            
            return [
                TextContent(
                    type="text",
                    text=indirect_text
                ).model_dump()
            ]
        
        elif name == "get_rxnorm_indications":
            cui = arguments["cui"]
            limit = arguments.get("limit", 50)
            result = await call_umls_api(f"/cuis/{cui}/medications/indications", {"limit": limit})
            meds = result.get("medications", [])
            if not meds:
                return [TextContent(type="text", text=f"No indication medications found for CUI {cui}.").model_dump()]
            lines = [f"• {m['code']} | {m['name']} | {m.get('source','RXNORM')} | {m.get('relationship','')}" for m in meds]
            return [TextContent(type="text", text=f"Indication medications for {cui} (n={len(meds)}):\n\n" + "\n".join(lines)).model_dump()]
        
        elif name == "get_rxnorm_related":
            cui = arguments["cui"]
            limit = arguments.get("limit", 50)
            result = await call_umls_api(f"/cuis/{cui}/medications/related", {"limit": limit})
            meds = result.get("medications", [])
            if not meds:
                return [TextContent(type="text", text=f"No related medications found for CUI {cui}.").model_dump()]
            lines = [f"• {m['code']} | {m['name']} | {m.get('source','RXNORM')} | {m.get('relationship','')}" for m in meds]
            return [TextContent(type="text", text=f"Related medications for {cui} (n={len(meds)}):\n\n" + "\n".join(lines)).model_dump()]
            
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        logger.error(f"Error in tool {name}: {str(e)}")
        return [
            TextContent(
                type="text",
                text=f"Error: {str(e)}"
            ).model_dump()
        ]

async def main():
    """Main entry point for the MCP server."""
    
    # Check if we're running with stdio or need to set up SSE
    if len(sys.argv) > 1 and sys.argv[1] == "--sse":
        # SSE mode would be implemented here for web-based connections
        logger.info("SSE mode not implemented yet")
        return
    
    # Default to stdio mode for Claude Desktop
    logger.info("Starting UMLS MCP Server in stdio mode...")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main()) 