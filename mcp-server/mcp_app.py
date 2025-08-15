from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import httpx
import os
from dotenv import load_dotenv
import logging
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="UMLS MCP Server", description="Middleware for UMLS API")

# Configuration
UMLS_API_URL = os.getenv("UMLS_API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "default_key")

# Intent definitions
INTENT_MAPPINGS = {
    "search_terms": {
        "endpoint": "/terms",
        "method": "GET",
        "params": ["search", "ontology"]
    },
    "get_cui_info": {
        "endpoint": "/cuis/{cui}",
        "method": "GET",
        "params": ["cui"]
    },
    "search_cui": {
        "endpoint": "/cuis",
        "method": "GET",
        "params": ["query"]
    },
    "get_ancestors": {
        "endpoint": "/cuis/{cui}/ancestors",
        "method": "GET",
        "params": ["cui"]
    },
    "get_depth": {
        "endpoint": "/cuis/{cui}/depth",
        "method": "GET",
        "params": ["cui"]
    },
    "get_cui_from_ontology": {
        "endpoint": "/ontology/{ontology}/code/{code}",
        "method": "GET",
        "params": ["ontology", "code"]
    },
    "find_lca": {
        "endpoint": "/cuis/{cui1}/{cui2}/lca",
        "method": "GET",
        "params": ["cui1", "cui2"]
    },
    "wu_palmer_similarity": {
        "endpoint": "/cuis/{cui1}/{cui2}/similarity/wu-palmer",
        "method": "GET",
        "params": ["cui1", "cui2"]
    },
    "get_hpo_term": {
        "endpoint": "/cuis/{cui}/hpo",
        "method": "GET",
        "params": ["cui"]
    },
    "get_relationships": {
        "endpoint": "/cuis/{cui1}/{cui2}/relationships",
        "method": "GET",
        "params": ["cui1", "cui2", "sab"]
    },
    "get_indirect_relationships": {
        "endpoint": "/cuis/{cui1}/{cui2}/relationships/indirect",
        "method": "GET",
        "params": ["cui1", "cui2", "max_depth", "sab"]
    }
}

# Models
class IntentRequest(BaseModel):
    intent: str
    parameters: Dict[str, Any]

class IntentResponse(BaseModel):
    result: Any
    status: str = "success"
    message: Optional[str] = None

# Authentication
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key

# Helper functions
def format_url(endpoint: str, params: Dict[str, Any]) -> str:
    """Format URL with parameters."""
    try:
        return endpoint.format(**params)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required parameter: {e}")

async def call_umls_api(endpoint: str, method: str, params: Dict[str, Any]) -> Any:
    """Call the UMLS API with the given parameters."""
    url = f"{UMLS_API_URL}{endpoint}"
    
    logger.info(f"Calling UMLS API: {url} with params: {params}")
    
    async with httpx.AsyncClient() as client:
        try:
            if method == "GET":
                response = await client.get(url, params=params, timeout=10.0)
            else:
                raise HTTPException(status_code=405, detail=f"Method {method} not supported")
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e}, Response: {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise HTTPException(status_code=500, detail=f"Error connecting to UMLS API: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Routes
@app.post("/process_intent")
async def process_intent(request: IntentRequest):
    """Process an intent by mapping it to the appropriate UMLS API endpoint."""
    intent = request.intent
    parameters = request.parameters
    
    logger.info(f"Received request to process intent: {intent} with parameters: {parameters}")
    
    if intent not in INTENT_MAPPINGS:
        logger.error(f"Unknown intent: {intent}")
        raise HTTPException(status_code=400, detail=f"Unknown intent: {intent}")
    
    mapping = INTENT_MAPPINGS[intent]
    endpoint = mapping["endpoint"]
    method = mapping["method"]
    required_params = mapping["params"]
    
    logger.info(f"Intent '{intent}' mapped to endpoint '{endpoint}' with method '{method}'")
    
    # Handle parameter aliases (e.g., 'term' for 'search' in search_terms intent)
    if intent == "search_terms" and "term" in parameters:
        parameters["search"] = parameters.pop("term")
    
    # Check if all required parameters are present
    missing_params = [param for param in required_params if param not in parameters]
    if missing_params:
        logger.error(f"Missing required parameters: {missing_params}")
        raise HTTPException(status_code=400, detail=f"Missing required parameters: {missing_params}")
    
    # Format the endpoint URL with path parameters
    formatted_endpoint = endpoint
    path_params = {}
    query_params = {}
    
    for param in required_params:
        if f"{{{param}}}" in endpoint:
            formatted_endpoint = formatted_endpoint.replace(f"{{{param}}}", parameters[param])
            path_params[param] = parameters[param]
        else:
            query_params[param] = parameters[param]
    
    logger.info(f"Formatted endpoint URL: {formatted_endpoint}")
    logger.info(f"Path parameters: {path_params}")
    logger.info(f"Query parameters: {query_params}")
    
    # Call the UMLS API
    umls_api_url = f"{UMLS_API_URL}{formatted_endpoint}"
    logger.info(f"Calling UMLS API: {umls_api_url} with params: {query_params}")
    
    try:
        # Increase timeout for complex operations like Wu-Palmer similarity
        timeout = 600.0 if intent in ["wu_palmer_similarity", "find_lca", "get_relationships", "get_indirect_relationships"] else 30.0
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method == "GET":
                response = await client.get(umls_api_url, params=query_params)
            elif method == "POST":
                response = await client.post(umls_api_url, json=query_params)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported method: {method}")
            
            logger.info(f"Received response from UMLS API: {response.text}")
            
            if response.status_code == 404:
                error_detail = response.json().get("detail", "Resource not found")
                logger.error(f"Resource not found: {error_detail}")
                return {"error": error_detail, "status": "not_found"}
            
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException as e:
        logger.error(f"Timeout error: {str(e)}")
        return {"error": f"The operation is taking longer than expected. Please try again later.", "status": "timeout"}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.text}")
        error_detail = e.response.json().get("detail", str(e))
        logger.error(f"Request error: {error_detail}")
        return {"error": error_detail, "status": "error"}
    except Exception as e:
        logger.error(f"Error processing intent: {str(e)}")
        return {"error": str(e), "status": "error"}

@app.get("/intents", response_model=List[str])
async def list_intents(api_key: str = Depends(verify_api_key)):
    """List all available intents."""
    return list(INTENT_MAPPINGS.keys())

@app.get("/intent_details/{intent}")
async def get_intent_details(intent: str, api_key: str = Depends(verify_api_key)):
    """Get details about a specific intent."""
    if intent not in INTENT_MAPPINGS:
        raise HTTPException(status_code=404, detail=f"Intent not found: {intent}")
    
    return {
        "intent": intent,
        "endpoint": INTENT_MAPPINGS[intent]["endpoint"],
        "method": INTENT_MAPPINGS[intent]["method"],
        "required_parameters": INTENT_MAPPINGS[intent]["params"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 