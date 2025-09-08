from fastapi import FastAPI, HTTPException, Depends, Header, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import httpx
import os
from dotenv import load_dotenv
import logging
import json
import openai
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="UMLS LLM Integration", description="Natural language interface for UMLS API")

# Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001")
API_KEY = os.getenv("API_KEY", "default_key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Models
class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    conversation_id: str
    raw_data: Optional[Any] = None

# Authentication
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key

# Helper functions
async def get_available_intents() -> List[str]:
    """Get list of available intents from MCP server."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{MCP_SERVER_URL}/intents",
                headers={"X-API-Key": API_KEY},
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting intents: {e}")
            return []

async def get_intent_details(intent: str) -> Dict[str, Any]:
    """Get details about a specific intent from MCP server."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{MCP_SERVER_URL}/intent_details/{intent}",
                headers={"X-API-Key": API_KEY},
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting intent details: {e}")
            return {}

async def process_intent(intent: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Process an intent by sending it to the MCP server."""
    logger.info(f"Processing intent: {intent} with parameters: {parameters}")
    
    # Set a longer timeout for complex operations like Wu-Palmer similarity
    timeout = 600.0 if intent in ["wu_palmer_similarity", "find_lca", "get_relationships", "get_indirect_relationships"] else 30.0
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            # Send the request in the format expected by the MCP Server
            response = await client.post(
                f"{MCP_SERVER_URL}/process_intent",
                json={"intent": intent, "parameters": parameters},
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": API_KEY
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                # Check if the result contains an error
                if "error" in result:
                    logger.error(f"Error from MCP Server: {result['error']}")
                    return {"error": result["error"], "status": result.get("status", "error")}
                return result
            else:
                error_detail = response.json().get("detail", "Unknown error")
                logger.error(f"Error from MCP Server: {error_detail}")
                return {"error": error_detail, "status": "error"}
        except httpx.TimeoutException as e:
            logger.error(f"Timeout error when calling MCP Server: {str(e)}")
            return {"error": "The operation is taking longer than expected. Please try again later.", "status": "timeout"}
        except Exception as e:
            logger.error(f"Error processing intent: {str(e)}")
            return {"error": str(e), "status": "error"}

def create_system_prompt() -> str:
    """Create the system prompt for the LLM."""
    return """You are a helpful assistant for querying the UMLS (Unified Medical Language System) database.
Your task is to understand natural language queries and map them to the appropriate UMLS API endpoints.

Available intents:
- search_terms: Search for medical terms in UMLS based on ontology
- get_cui_info: Get details about a specific CUI
- search_cui: Search for CUIs by term
- get_ancestors: Get all ancestors of a CUI
- get_depth: Get depth of a CUI in the hierarchy
- get_relations: Get hierarchical relations for a CUI
- get_cui_from_ontology: Map an ontology term to a CUI
- find_lca: Get the lowest common ancestor of two CUIs
- wu_palmer_similarity: Compute Wu-Palmer similarity between two CUIs
- get_hpo_term: Get the HPO term and code for a given CUI
- get_relationships: Get direct relationships between two CUIs
- get_indirect_relationships: Get indirect relationships between two CUIs through intermediate concepts

For each query, you should:
1. Identify the most appropriate intent
2. Extract the required parameters for that intent
3. Return a JSON object with the intent and parameters

Example:
User: "What is the CUI for diabetes?"
Response: {"intent": "search_cui", "parameters": {"query": "diabetes"}}

User: "What are the ancestors of CUI C0011849?"
Response: {"intent": "get_ancestors", "parameters": {"cui": "C0011849"}}

User: "What is the similarity between CUI C0011849 and C0011860?"
Response: {"intent": "wu_palmer_similarity", "parameters": {"cui1": "C0011849", "cui2": "C0011860"}}

User: "What is the HPO code for CUI C0011849?"
Response: {"intent": "get_hpo_term", "parameters": {"cui": "C0011849"}}

User: "Search for asthma in HPO"
Response: {"intent": "search_terms", "parameters": {"search": "asthma", "ontology": "HPO"}}

User: "What are the relationships between CUI C0011849 and C0011860?"
Response: {"intent": "get_relationships", "parameters": {"cui1": "C0011849", "cui2": "C0011860"}}

User: "What are the indirect relationships between CUI C0011849 and C0011860?"
Response: {"intent": "get_indirect_relationships", "parameters": {"cui1": "C0011849", "cui2": "C0011860"}}

User: "What are the relationships between CUI C0011849 and C0011860 in SNOMEDCT_US?"
Response: {"intent": "get_relationships", "parameters": {"cui1": "C0011849", "cui2": "C0011860", "sab": "SNOMEDCT_US"}}

Always respond with a valid JSON object containing the intent and parameters."""

def extract_intent_and_parameters(llm_response: str) -> Dict[str, Any]:
    """Extract intent and parameters from LLM response."""
    try:
        # Try to parse the response as JSON
        result = json.loads(llm_response)
        
        # Validate the response format
        if "intent" not in result or "parameters" not in result:
            raise ValueError("Response missing 'intent' or 'parameters'")
        
        return result
    except json.JSONDecodeError:
        # If the response is not valid JSON, try to extract it using string manipulation
        logger.warning(f"Failed to parse LLM response as JSON: {llm_response}")
        
        # Look for JSON-like structure in the response
        import re
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                if "intent" in result and "parameters" in result:
                    return result
            except:
                pass
        
        raise HTTPException(status_code=500, detail="Failed to extract intent and parameters from LLM response")

# Routes
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, api_key: str = Depends(verify_api_key)):
    """Process a natural language query and return a response."""
    query = request.query
    conversation_id = request.conversation_id or f"conv_{os.urandom(8).hex()}"
    
    logger.info(f"Processing query: {query}")
    
    # Get available intents for context
    available_intents = await get_available_intents()
    
    # Create messages for the LLM
    messages = [
        {"role": "system", "content": create_system_prompt()},
        {"role": "user", "content": query}
    ]
    
    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )
        
        # Extract the response text
        llm_response = response.choices[0].message.content.strip()
        logger.info(f"LLM response: {llm_response}")
        
        # Extract intent and parameters
        intent_data = extract_intent_and_parameters(llm_response)
        intent = intent_data["intent"]
        parameters = intent_data["parameters"]
        
        # Process the intent through the MCP server
        result = await process_intent(intent, parameters)
        
        # Format the response for the user
        formatted_response = format_response_for_user(intent, result)
        
        return QueryResponse(
            response=formatted_response,
            conversation_id=conversation_id,
            raw_data=result
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

def format_response_for_user(intent: str, result: Any) -> str:
    """Format the API result into a natural language response."""
    # Check if the result contains an error
    if isinstance(result, dict) and "error" in result:
        error_message = result["error"]
        
        # Handle case where error_message is a list (from Pydantic validation errors)
        if isinstance(error_message, list):
            error_details = []
            for error in error_message:
                if isinstance(error, dict) and "msg" in error:
                    error_details.append(error["msg"])
                else:
                    error_details.append(str(error))
            error_message = "; ".join(error_details)
        
        # Handle timeout errors
        if result.get("status") == "timeout":
            return f"The operation is taking longer than expected. This is normal for complex calculations like similarity measures. Please try again in a few moments."
        
        if "No common ancestor found" in error_message:
            return f"I couldn't find a common ancestor between the two medical terms. This means they are not related in the UMLS hierarchy, so I cannot calculate their similarity."
        elif "not found" in error_message.lower():
            return f"I couldn't find the information you requested: {error_message}"
        else:
            return f"I encountered an error while processing your request: {error_message}"
    
    if intent == "search_terms":
        if not result.get("results"):
            return "No medical terms found matching your query."
        
        terms = result["results"]
        response = f"I found {len(terms)} medical terms:\n\n"
        
        for i, term in enumerate(terms, 1):
            response += f"{i}. {term['term']} (Code: {term['code']})\n"
            if term.get("description"):
                response += f"   Description: {term['description'][:100]}...\n"
            response += "\n"
        
        return response
    
    elif intent == "get_cui_info":
        return f"The CUI {result['cui']} corresponds to: {result['name']}"
    
    elif intent == "search_cui":
        if not result.get("cuis"):
            return "No CUIs found matching your query."
        
        cuis = result["cuis"]
        response = f"I found {len(cuis)} CUIs for '{result['query']}':\n\n"
        
        for i, cui in enumerate(cuis, 1):
            response += f"{i}. {cui['name']} (CUI: {cui['cui']})\n"
        
        return response
    
    elif intent == "get_ancestors":
        if not result.get("ancestors"):
            return f"No ancestors found for CUI {result['cui']}."
        
        ancestors = result["ancestors"]
        return f"CUI {result['cui']} has {len(ancestors)} ancestors: {', '.join(ancestors)}"
    
    elif intent == "get_depth":
        return f"The depth of CUI {result['cui']} in the hierarchy is {result['depth']}."
    
    elif intent == "get_relations":
        parents = result.get("parents", [])
        children = result.get("children", [])
        ancestors = result.get("ancestors", [])
        
        response = f"Relations for CUI {result['cui']}:\n\n"
        
        if parents:
            response += f"Parents: {', '.join(parents)}\n"
        else:
            response += "No parents found.\n"
        
        if children:
            response += f"Children: {', '.join(children)}\n"
        else:
            response += "No children found.\n"
        
        if ancestors:
            response += f"Ancestors: {', '.join(ancestors)}\n"
        else:
            response += "No ancestors found.\n"
        
        return response
    
    elif intent == "get_cui_from_ontology":
        return f"The CUI for {result['ontology']} term {result['term']} is {result['cui']}."
    
    elif intent == "find_lca":
        return f"The lowest common ancestor of CUIs {result['cui1']} and {result['cui2']} is {result['lca']} (depth: {result['depth']})."
    
    elif intent == "wu_palmer_similarity":
        return f"The Wu-Palmer similarity between CUIs {result['cui1']} and {result['cui2']} is {result['similarity']:.4f}. Their lowest common ancestor is {result['lca']}."
    
    elif intent == "get_hpo_term":
        return f"The HPO term for CUI {result['cui']} is '{result['hpo_term']}' with code {result['hpo_code']}."
    
    elif intent == "get_relationships":
        if not result.get("relationships"):
            return f"No direct relationships found between CUIs {result['cui1']} and {result['cui2']}."
        
        relationships = result["relationships"]
        response = f"I found {len(relationships)} direct relationships between CUIs {result['cui1']} and {result['cui2']}:\n\n"
        
        for i, rel in enumerate(relationships, 1):
            response += f"{i}. {rel['cui1_name']} → {rel['cui2_name']}\n"
            response += f"   Relationship: {rel['rel']}"
            if rel.get('rela'):
                response += f" ({rel['rela']})"
            response += f"\n   Source: {rel['sab']}\n\n"
        
        return response
    
    elif intent == "get_indirect_relationships":
        if not result.get("indirect_relationships"):
            return f"No indirect relationships found between CUIs {result['cui1']} and {result['cui2']}."
        
        indirect_rels = result["indirect_relationships"]
        response = f"I found {len(indirect_rels)} indirect relationship paths between CUIs {result['cui1']} and {result['cui2']}:\n\n"
        
        for i, path in enumerate(indirect_rels, 1):
            response += f"{i}. Path: {path['path']}\n"
            response += f"   Intermediate: {path['intermediate_name']} ({path['intermediate_cui']})\n"
            
            # Step 1
            step1 = path['step1']
            response += f"   Step 1: {step1['from_name']} → {step1['to_name']}\n"
            response += f"           Relationship: {step1['rel']}"
            if step1.get('rela'):
                response += f" ({step1['rela']})"
            response += f" (Source: {step1['sab']})\n"
            
            # Step 2
            step2 = path['step2']
            response += f"   Step 2: {step2['from_name']} → {step2['to_name']}\n"
            response += f"           Relationship: {step2['rel']}"
            if step2.get('rela'):
                response += f" ({step2['rela']})"
            response += f" (Source: {step2['sab']})\n\n"
        
        return response
    
    else:
        return f"Received response for intent '{intent}': {json.dumps(result, indent=2)}"

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 