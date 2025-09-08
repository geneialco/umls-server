from fastapi import FastAPI, HTTPException, Query
import aiomysql
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
import asyncio
import logging

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Timeout in seconds for external calls
TIMEOUT = 500

async def connect_db():
    """Establish database connection."""
    try:
        logging.info("Attempting to connect to database...")
        logging.info(f"Database: {os.getenv('DB_NAME')}")
        logging.info(f"User: {os.getenv('DB_USER')}")
        logging.info(f"Host: {os.getenv('DB_HOST', 'localhost')}")
        
        # Use Docker-compatible connection or fallback to unix socket for local dev
        db_host = os.getenv("DB_HOST")
        if db_host and db_host != "localhost":
            # Docker connection
            conn = await aiomysql.connect(
                host=db_host,
                port=int(os.getenv("DB_PORT", 3306)),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                db=os.getenv("DB_NAME"),
                cursorclass=aiomysql.DictCursor,
                autocommit=True,
                connect_timeout=10,
                charset='utf8mb4',
                use_unicode=True
            )
        else:
            # Local development with unix socket
            conn = await aiomysql.connect(
                unix_socket='/var/lib/mysql/mysql.sock',
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                db=os.getenv("DB_NAME"),
                cursorclass=aiomysql.DictCursor,
                autocommit=True,
                connect_timeout=10,
                charset='utf8mb4',
                use_unicode=True
            )
        logging.info("Successfully connected to database")
        return conn
    except Exception as e:
        logging.error(f"Database connection error: {str(e)}")
        logging.error(f"Error type: {type(e)}")
        logging.error(f"Error args: {e.args}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

def clean_html(html_text):
    """Remove HTML tags from text."""
    return BeautifulSoup(html_text, "html.parser").get_text() if html_text else None


@app.get("/hpo_to_cui/{hpo_code}")
async def get_cui_from_hpo(hpo_code: str):
    try:
        conn = await connect_db()
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute("""
                SELECT CUI 
                FROM MRCONSO 
                WHERE CODE = %s 
                AND SAB = 'HPO' 
                LIMIT 1
            """, (hpo_code,))
            result = await cursor.fetchone()

            if not result:
                raise HTTPException(status_code=404, detail="CUI not found for the given HPO code")

            return {
                "hpo_code": hpo_code,
                "cui": result["CUI"]
            }

    except Exception as e:
        logging.error(f"Error getting CUI from HPO: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if 'conn' in locals():
            conn.close()


@app.get("/terms")
async def search_terms(search: str, ontology: str = "HPO"):
    """Search for medical terms in UMLS based on ontology."""
    try:
        conn = await connect_db()
        async with conn.cursor() as cursor:
            await cursor.execute("""
                SELECT DISTINCT MRCONSO.CODE, MRCONSO.STR, MRDEF.DEF
                FROM MRCONSO
                LEFT JOIN MRDEF ON MRCONSO.CUI = MRDEF.CUI
                WHERE MRCONSO.SAB = %s
                AND MRCONSO.STR LIKE %s
                LIMIT 10;
            """, (ontology, f"%{search}%"))
            results = await cursor.fetchall()

            formatted_results = []
            seen_codes = set()

            for row in results:
                code = row["CODE"]
                term = row["STR"]
                description = clean_html(row["DEF"])

                if code not in seen_codes:
                    formatted_results.append({"code": code, "term": term, "description": description})
                    seen_codes.add(code)

    except Exception as e:
        logging.error(f"Error searching terms: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'conn' in locals():
            conn.close()

    if not formatted_results:
        raise HTTPException(status_code=404, detail="No results found")

    return {"results": formatted_results}


@app.get("/cuis/{cui}/ancestors", summary="Get all ancestors of a CUI")
async def get_ancestors(cui: str):
    """ Retrieve all ancestors of a CUI by extracting AUIs from MRHIER.PTR and mapping them to CUIs via MRCONSO. """
    try:
        conn = await connect_db()
        async with conn.cursor() as cursor:
            # Step 1: Retrieve the AUI paths (PTR) from MRHIER using SNOMEDCT_US
            logging.info(f"Fetching PTR paths for CUI {cui} from SNOMEDCT_US")
            await cursor.execute("SELECT PTR FROM MRHIER WHERE CUI = %s AND SAB = 'SNOMEDCT_US'", (cui,))
            results = await cursor.fetchall()
            logging.info(f"Found {len(results)} PTR paths for CUI {cui}")

            if not results:
                logging.info(f"No ancestors found for CUI {cui}")
                return {"cui": cui, "ancestors": []}  # No ancestors found

            # Step 2: Extract AUIs from PTR and map them to CUIs
            auis = set()
            for row in results:
                ptr_path = row["PTR"]
                if ptr_path:
                    auis.update(ptr_path.split("."))  # Extract AUIs from dot-separated paths
            logging.info(f"Extracted {len(auis)} unique AUIs from PTR paths")

            if not auis:
                logging.info(f"No AUIs found in PTR paths for CUI {cui}")
                return {"cui": cui, "ancestors": []}  # No ancestors found

            # Step 3: Map AUIs to CUIs using MRCONSO (filtered to SNOMEDCT_US)
            logging.info(f"Mapping {len(auis)} AUIs to CUIs from SNOMEDCT_US")
            await cursor.execute("""
                SELECT DISTINCT AUI, CUI FROM MRCONSO WHERE AUI IN %s AND SAB = 'SNOMEDCT_US'
            """, (tuple(auis),))
            mappings = await cursor.fetchall()
            logging.info(f"Found {len(mappings)} AUI to CUI mappings")

            # Convert AUIs to CUIs
            aui_to_cui = {m["AUI"]: m["CUI"] for m in mappings}
            ancestors_cuis = {aui_to_cui[aui] for aui in auis if aui in aui_to_cui}
            logging.info(f"Found {len(ancestors_cuis)} unique ancestor CUIs")

            return {"cui": cui, "ancestors": list(ancestors_cuis)}

    except Exception as e:
        logging.error(f"Error getting ancestors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'conn' in locals():
            conn.close()

@app.get("/cuis/{cui}", summary="Get details about a specific CUI")
async def get_cui_info(cui: str):
    """Get details about a given CUI."""
    try:
        conn = await connect_db()
        async with conn.cursor() as cursor:
            await cursor.execute("""
                SELECT CUI, STR 
                FROM MRCONSO 
                WHERE CUI = %s AND SAB = 'SNOMEDCT_US'
                LIMIT 1
            """, (cui,))
            result = await cursor.fetchone()

    except Exception as e:
        logging.error(f"Error getting CUI info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'conn' in locals():
            conn.close()

    if not result:
        raise HTTPException(status_code=404, detail="CUI not found")
    
    return {"cui": result["CUI"], "name": result["STR"]}


## UMLS CUI toolkit
@app.get("/cuis", summary="Search for CUIs by term")
async def search_cui(query: str = Query(..., description="Search term for CUI lookup")):
    """Search for CUIs matching a given term."""
    try:
        conn = await connect_db()
        async with conn.cursor() as cursor:
            await cursor.execute("""
                SELECT CUI, STR 
                FROM MRCONSO 
                WHERE STR LIKE %s AND SAB = 'SNOMEDCT_US'
                LIMIT 50
            """, (f"%{query}%",))
            results = await cursor.fetchall()

    except Exception as e:
        logging.error(f"Error searching CUIs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'conn' in locals():
            conn.close()

    if not results:
        raise HTTPException(status_code=404, detail="No CUIs found for the given term")
    
    return {"query": query, "cuis": [{"cui": r["CUI"], "name": r["STR"]} for r in results]}


@app.get("/cuis/{cui}/depth")
async def get_depth(cui: str):
    """Get the depth of a CUI in the hierarchy."""
    try:
        conn = await connect_db()
        async with conn.cursor() as cursor:
            # Get the minimum depth from MRHIER table using SNOMEDCT_US vocabulary
            await cursor.execute("""
                SELECT MIN(LENGTH(PTR) - LENGTH(REPLACE(PTR, '.', '')) + 1) as min_depth
                FROM MRHIER
                WHERE CUI = %s AND SAB = 'SNOMEDCT_US'
            """, (cui,))
            result = await cursor.fetchone()
            
            if not result or result["min_depth"] is None:
                raise HTTPException(status_code=404, detail=f"Depth not found for CUI {cui} in SNOMEDCT_US hierarchy. This concept may not have hierarchical relationships in the formal medical taxonomy.")
                
            return {"cui": cui, "depth": result["min_depth"]}
    except Exception as e:
        logging.error(f"Error getting depth: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'conn' in locals():
            conn.close()

@app.get("/cuis/{cui1}/{cui2}/similarity/wu-palmer", summary="Compute Wu-Palmer similarity")
async def wu_palmer_similarity(cui1: str, cui2: str):
    """Compute Wu-Palmer similarity between two CUIs using MRHIER and the fetch_depth helper."""
    logging.info("Computing Wu-Palmer similarity for %s and %s", cui1, cui2)

    # Get the lowest common ancestor asynchronously.
    lca_result = await find_lowest_common_ancestor(cui1, cui2)
    lca = lca_result.get("lca")
    logging.info("Lowest common ancestor for %s and %s is %s", cui1, cui2, lca)

    # Concurrently fetch depths for cui1, cui2, and the LCA.
    try:
        depth_c1, depth_c2, depth_lca = await asyncio.gather(
            get_depth(cui1),
            get_depth(cui2),
            get_depth(lca)
        )
    except HTTPException as e:
        logging.error("Error fetching depths: %s", e.detail)
        raise

    logging.info("Depths: %s -> %s, %s -> %s, LCA %s -> %s", cui1, depth_c1, cui2, depth_c2, lca, depth_lca)

    if depth_c1["depth"] == 0 or depth_c2["depth"] == 0:
        logging.error("One or both CUIs have no valid depth")
        raise HTTPException(status_code=400, detail="One or both CUIs have no valid depth in SNOMEDCT_US hierarchy. These concepts may not be part of the formal medical hierarchy.")

    # Validate that LCA depth is less than or equal to both constituent depths
    if depth_lca["depth"] > depth_c1["depth"] or depth_lca["depth"] > depth_c2["depth"]:
        logging.error("LCA depth (%s) is greater than constituent depths (%s, %s)", 
                     depth_lca["depth"], depth_c1["depth"], depth_c2["depth"])
        raise HTTPException(status_code=500, detail="Invalid hierarchy: LCA depth exceeds constituent depths")

    similarity = (2 * depth_lca["depth"]) / (depth_c1["depth"] + depth_c2["depth"] + depth_lca["depth"])
    logging.info("Computed Wu-Palmer similarity: %s", similarity)

    return {
        "cui1": cui1,
        "cui2": cui2,
        "lca": lca,
        "depth_c1": depth_c1["depth"],
        "depth_c2": depth_c2["depth"],
        "depth_lca": depth_lca["depth"],
        "similarity": similarity,
    }


@app.get("/cuis/{cui1}/{cui2}/lca", summary="Get the lowest common ancestor of two CUIs")
async def find_lowest_common_ancestor(cui1: str, cui2: str):
    """Find the lowest common ancestor (LCA) of two CUIs using the new depth functions."""
    logging.info("Fetching ancestors for %s and %s", cui1, cui2)
    try:
        # Get ancestors for each CUI
        ancestors1_response = await get_ancestors(cui1)
        ancestors2_response = await get_ancestors(cui2)
        
        ancestors1 = set(ancestors1_response.get("ancestors", []))
        ancestors2 = set(ancestors2_response.get("ancestors", []))
        
        # Check if one concept is an ancestor of the other
        if cui1 in ancestors2:
            # cui1 is an ancestor of cui2, so cui1 is the LCA
            logging.info(f"{cui1} is an ancestor of {cui2}, making it the LCA")
            try:
                depth_response = await get_depth(cui1)
                return {"cui1": cui1, "cui2": cui2, "lca": cui1, "depth": depth_response["depth"]}
            except HTTPException:
                raise HTTPException(status_code=404, detail=f"{cui1} not found in SNOMEDCT_US hierarchy")
        
        if cui2 in ancestors1:
            # cui2 is an ancestor of cui1, so cui2 is the LCA
            logging.info(f"{cui2} is an ancestor of {cui1}, making it the LCA")
            try:
                depth_response = await get_depth(cui2)
                return {"cui1": cui1, "cui2": cui2, "lca": cui2, "depth": depth_response["depth"]}
            except HTTPException:
                raise HTTPException(status_code=404, detail=f"{cui2} not found in SNOMEDCT_US hierarchy")
        
        # Exclude the original CUIs from being considered as their own ancestors
        common_ancestors = (ancestors1 & ancestors2) - {cui1, cui2}
        logging.info("Common ancestors: %s", common_ancestors)

        if not common_ancestors:
            # Check if both concepts exist in SNOMEDCT_US hierarchy
            try:
                await get_depth(cui1)
                cui1_in_hierarchy = True
            except HTTPException:
                cui1_in_hierarchy = False
                
            try:
                await get_depth(cui2)
                cui2_in_hierarchy = True
            except HTTPException:
                cui2_in_hierarchy = False
            
            if not cui1_in_hierarchy and not cui2_in_hierarchy:
                raise HTTPException(status_code=404, detail="Both concepts lack hierarchical relationships in SNOMEDCT_US. These concepts exist in the terminology but are not part of the formal medical hierarchy.")
            elif not cui1_in_hierarchy:
                raise HTTPException(status_code=404, detail=f"Concept {cui1} lacks hierarchical relationships in SNOMEDCT_US. This concept exists in the terminology but is not part of the formal medical hierarchy.")
            elif not cui2_in_hierarchy:
                raise HTTPException(status_code=404, detail=f"Concept {cui2} lacks hierarchical relationships in SNOMEDCT_US. This concept exists in the terminology but is not part of the formal medical hierarchy.")
            else:
                raise HTTPException(status_code=404, detail="No common ancestor found in SNOMEDCT_US hierarchy. These concepts may be too distantly related or belong to different medical domains.")

        # Fetch depths for each common ancestor (all are now from SNOMEDCT_US)
        depth_dict = {}
        for ancestor in common_ancestors:
            try:
                depth_response = await get_depth(ancestor)
                depth_dict[ancestor] = depth_response["depth"]
            except Exception as e:
                logging.error("Error fetching depth for %s: %s", ancestor, e)
                depth_dict[ancestor] = 0  # Fallback to 0 on error

        if not depth_dict:
            raise HTTPException(status_code=404, detail="Unable to compute depths for common ancestors")

        # Determine the LCA as the ancestor with the maximum depth (most specific common ancestor)
        lca = max(depth_dict.items(), key=lambda x: x[1])[0]
        logging.info("Lowest common ancestor for %s and %s is %s", cui1, cui2, lca)
        return {"cui1": cui1, "cui2": cui2, "lca": lca, "depth": depth_dict[lca]}
    except Exception as e:
        logging.error("Error finding LCA: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cuis/{cui1}/{cui2}/relationships", summary="Get relationships between two CUIs")
async def get_relationships(cui1: str, cui2: str, sab: str = Query(None, description="Source vocabulary (e.g., 'SNOMEDCT_US', 'HPO'). If not specified, returns relationships from all sources.")):
    """Get all relationships between two given CUIs from the MRREL table."""
    try:
        conn = await connect_db()
        async with conn.cursor() as cursor:
            # Build the query based on whether SAB is specified
            if sab:
                query = """
                    SELECT DISTINCT r.CUI1, r.CUI2, r.REL, r.RELA, r.SAB, r.SL, r.DIR, r.SUPPRESS,
                           c1.STR as CUI1_NAME, c2.STR as CUI2_NAME
                    FROM MRREL r
                    JOIN MRCONSO c1 ON r.CUI1 = c1.CUI AND c1.LAT = 'ENG'
                    JOIN MRCONSO c2 ON r.CUI2 = c2.CUI AND c2.LAT = 'ENG'
                    WHERE ((r.CUI1 = %s AND r.CUI2 = %s) OR (r.CUI1 = %s AND r.CUI2 = %s))
                    AND r.SAB = %s
                    ORDER BY r.RELA, r.SAB
                    LIMIT 50
                """
                params = (cui1, cui2, cui2, cui1, sab)
            else:
                query = """
                    SELECT DISTINCT r.CUI1, r.CUI2, r.REL, r.RELA, r.SAB, r.SL, r.DIR, r.SUPPRESS,
                           c1.STR as CUI1_NAME, c2.STR as CUI2_NAME
                    FROM MRREL r
                    JOIN MRCONSO c1 ON r.CUI1 = c1.CUI AND c1.LAT = 'ENG'
                    JOIN MRCONSO c2 ON r.CUI2 = c2.CUI AND c2.LAT = 'ENG'
                    WHERE (r.CUI1 = %s AND r.CUI2 = %s) OR (r.CUI1 = %s AND r.CUI2 = %s)
                    ORDER BY r.RELA, r.SAB
                    LIMIT 50
                """
                params = (cui1, cui2, cui2, cui1)
            
            await cursor.execute(query, params)
            results = await cursor.fetchall()
            
            if not results:
                return {
                    "cui1": cui1,
                    "cui2": cui2,
                    "relationships": [],
                    "message": f"No relationships found between {cui1} and {cui2}" + (f" in {sab}" if sab else "")
                }
            
            # Convert results to list of dictionaries
            relationships = []
            for row in results:
                relationships.append({
                    "cui1": row["CUI1"],
                    "cui2": row["CUI2"],
                    "rel": row["REL"],
                    "rela": row["RELA"],
                    "sab": row["SAB"],
                    "sl": row["SL"],
                    "dir": row["DIR"],
                    "suppress": row["SUPPRESS"],
                    "cui1_name": row["CUI1_NAME"],
                    "cui2_name": row["CUI2_NAME"]
                })
            
            return {
                "cui1": cui1,
                "cui2": cui2,
                "relationships": relationships,
                "count": len(relationships)
            }
            
    except Exception as e:
        logging.error(f"Error getting relationships: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'conn' in locals():
            conn.close()

@app.get("/cuis/{cui1}/{cui2}/relationships/indirect", summary="Get indirect relationships between two CUIs")
async def get_indirect_relationships(cui1: str, cui2: str, max_depth: int = Query(2, description="Maximum path length to search (1-3 recommended)"), sab: str = Query(None, description="Source vocabulary filter")):
    """Get indirect relationships between two CUIs by finding paths through intermediate concepts."""
    try:
        conn = await connect_db()
        async with conn.cursor() as cursor:
            # Build the base query with optional SAB filter
            sab_filter = "AND r.SAB = %s" if sab else ""
            sab_param = (sab,) if sab else ()
            
            # Step 1: Get all concepts that CUI1 relates to (limited to 50 for performance)
            query1 = f"""
                SELECT DISTINCT r.CUI2 as related_cui, r.REL, r.RELA, r.SAB
                FROM MRREL r
                WHERE r.CUI1 = %s {sab_filter}
                ORDER BY r.RELA, r.SAB
                LIMIT 50
            """
            params1 = (cui1,) + sab_param
            await cursor.execute(query1, params1)
            cui1_related = await cursor.fetchall()
            
            # Step 2: Get all concepts that relate to CUI2 (limited to 50 for performance)
            query2 = f"""
                SELECT DISTINCT r.CUI1 as related_cui, r.REL, r.RELA, r.SAB
                FROM MRREL r
                WHERE r.CUI2 = %s {sab_filter}
                ORDER BY r.RELA, r.SAB
                LIMIT 50
            """
            params2 = (cui2,) + sab_param
            await cursor.execute(query2, params2)
            cui2_related = await cursor.fetchall()
            
            # Step 3: Find common intermediate concepts
            cui1_related_set = {row["related_cui"] for row in cui1_related}
            cui2_related_set = {row["related_cui"] for row in cui2_related}
            common_intermediates = cui1_related_set & cui2_related_set
            
            if not common_intermediates:
                return {
                    "cui1": cui1,
                    "cui2": cui2,
                    "indirect_relationships": [],
                    "message": f"No indirect relationships found between {cui1} and {cui2} through intermediate concepts" + (f" in {sab}" if sab else "")
                }
            
            # Step 4: Build paths through common intermediates (limit to first 10)
            indirect_paths = []
            for intermediate in list(common_intermediates)[:10]:  # Limit to first 10 to avoid overwhelming results
                # Find the relationship from CUI1 to intermediate
                cui1_to_intermediate = None
                for row in cui1_related:
                    if row["related_cui"] == intermediate:
                        cui1_to_intermediate = row
                        break
                
                # Find the relationship from intermediate to CUI2
                intermediate_to_cui2 = None
                for row in cui2_related:
                    if row["related_cui"] == intermediate:
                        intermediate_to_cui2 = row
                        break
                
                if cui1_to_intermediate and intermediate_to_cui2:
                    # Get concept names in a single query
                    await cursor.execute("""
                        SELECT CUI, STR FROM MRCONSO 
                        WHERE CUI IN (%s, %s, %s) AND LAT = 'ENG'
                        ORDER BY CUI
                    """, (cui1, intermediate, cui2))
                    names_result = await cursor.fetchall()
                    
                    # Create a name lookup
                    name_lookup = {row["CUI"]: row["STR"] for row in names_result}
                    
                    indirect_paths.append({
                        "path": f"{cui1} → {intermediate} → {cui2}",
                        "intermediate_cui": intermediate,
                        "intermediate_name": name_lookup.get(intermediate, intermediate),
                        "step1": {
                            "from": cui1,
                            "to": intermediate,
                            "rel": cui1_to_intermediate["REL"],
                            "rela": cui1_to_intermediate["RELA"],
                            "sab": cui1_to_intermediate["SAB"],
                            "from_name": name_lookup.get(cui1, cui1),
                            "to_name": name_lookup.get(intermediate, intermediate)
                        },
                        "step2": {
                            "from": intermediate,
                            "to": cui2,
                            "rel": intermediate_to_cui2["REL"],
                            "rela": intermediate_to_cui2["RELA"],
                            "sab": intermediate_to_cui2["SAB"],
                            "from_name": name_lookup.get(intermediate, intermediate),
                            "to_name": name_lookup.get(cui2, cui2)
                        }
                    })
            
            return {
                "cui1": cui1,
                "cui2": cui2,
                "indirect_relationships": indirect_paths,
                "count": len(indirect_paths),
                "max_depth": max_depth,
                "common_intermediates_found": len(common_intermediates)
            }
            
    except Exception as e:
        logging.error(f"Error getting indirect relationships: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'conn' in locals():
            conn.close()

@app.get("/cuis/{cui}/hpo", summary="Get HPO term and code from CUI")
async def get_hpo_term(cui: str):
    """Get the HPO term and code associated with a given CUI."""
    try:
        conn = await connect_db()
        async with conn.cursor() as cursor:
            await cursor.execute("""
                SELECT STR, CODE 
                FROM MRCONSO 
                WHERE CUI = %s 
                AND SAB = 'HPO' 
                LIMIT 1
            """, (cui,))
            result = await cursor.fetchone()

    except Exception as e:
        logging.error(f"Error getting HPO term: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'conn' in locals():
            conn.close()

    if not result:
        raise HTTPException(status_code=404, detail="HPO term not found for the given CUI")
    
    return {
        "cui": cui,
        "hpo_term": result["STR"],
        "hpo_code": result["CODE"]
    }
