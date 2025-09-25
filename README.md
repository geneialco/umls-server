# 🏥 UMLS Natural Language Interface

A two-tier architecture for providing natural language access to the Unified Medical Language System (UMLS) database through Claude Desktop using the Model Context Protocol (MCP).

## 🚀 Quick Setup Guide

**Want to get started fast? Choose your setup method:**

### 🐳 Option 1: Docker Setup (Recommended)

1. **Prerequisites**: 
   - Docker and Docker Compose installed
   - UMLS license and data files from NCBI
   - At least 12GB RAM available for Docker
   - At least 100GB disk space for UMLS data

2. **Setup Environment**:
   ```bash
   git clone <repository-url>
   cd umls-server
   cp docker.env .env
   ```

3. **Configure Docker Desktop** (Important!):
   - Open Docker Desktop → Settings → Resources
   - Memory: Set to 12-16GB (if you have 16GB+ system RAM)
   - CPUs: Set to 6+ cores if available
   - Disk: Ensure 100GB+ available to Docker
   - **Restart Docker Desktop** after making changes

4. **Start Services**:
   ```bash
   docker compose up -d mysql
   # Load UMLS data with optional API optimizations (see detailed instructions below)
   ./scripts/load_umls_2025aa.sh
   docker compose up -d
   ```

5. **Test**: `curl "http://localhost:8000/terms?search=diabetes&ontology=HPO"`

### 🐍 Option 2: Local Python Setup

1. **Prerequisites**: Install Python 3.8+, Conda, and Claude Desktop

2. **Clone and Setup Environment**:
   ```bash
   git clone <repository-url>
   cd umls-server
   conda create -n umls-server python=3.11
   conda activate umls-server
   pip install -r requirements.txt
   ```

3. **Configure Database** (in `umls_api/.env`):
   ```env
   DB_HOST=your-db-host
   DB_USER=your-db-user  
   DB_PASSWORD=your-db-password
   DB_NAME=your-db-name
   ```

4. **Start UMLS API**:
   ```bash
   cd umls_api
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

5. **Configure Claude Desktop**:
   - Copy `claude_desktop_config.json` to your Claude config directory
   - Update the Python path: `conda info --envs` to find your environment path
   - Update the project path to your actual directory
   - Restart Claude Desktop

6. **Test**: Ask Claude "Search for diabetes in the HPO ontology"

✅ **Done!** You can now query UMLS through natural language in Claude Desktop.

## 🌟 Quick Start: Using Claude Desktop

The UMLS server consists of two components:
1. **UMLS API** (Port 8000): Direct database access
2. **MCP Server**: Claude Desktop integration via stdio

### For Claude Desktop Integration

1. Set up the conda environment and dependencies:
   ```sh
   conda create -n umls-server python=3.11
   conda activate umls-server
   pip install -r requirements.txt
   ```

2. Start the UMLS API:
   ```sh
   cd umls_api
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

3. Configure Claude Desktop using the provided template (copy `claude_desktop_config.json` and update paths):
   ```json
   {
     "_comment": "Sample Claude Desktop configuration for UMLS MCP Server",
     "_instructions": [
       "1. Find your conda environment path: 'conda info --envs'",
       "2. Replace '/path/to/conda/envs/umls-server/bin/python' with your actual conda Python path",
       "3. Replace '/path/to/umls-server' with your actual project directory path",
       "4. Make sure the UMLS API is running on port 8000",
       "5. Restart Claude Desktop after making changes"
     ],
     "_examples": {
       "macOS": "/Users/username/miniconda3/envs/umls-server/bin/python",
       "Linux": "/home/username/miniconda3/envs/umls-server/bin/python",
       "Windows": "C:\\Users\\username\\miniconda3\\envs\\umls-server\\python.exe"
     },
     "mcpServers": {
       "umls-server": {
         "command": "/path/to/conda/envs/umls-server/bin/python",
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

4. Restart Claude Desktop and start using natural language to query UMLS!

## 🌟 Example API Usage

Once your UMLS server is running, you can test the API endpoints:

```sh
# Search for terms in HPO ontology
curl "http://localhost:8000/terms?search=cancer&ontology=HPO"

# Search for SNOMED CT terms
curl "http://localhost:8000/terms?search=diabetes&ontology=SNOMEDCT_US"

# Response format:
{
  "results": [
    {
      "code": "HP:0002896",
      "term": "Liver cancer",
      "description": "A primary or metastatic malignant neoplasm that affects the liver."
    }
  ]
}
```

**For cloud deployment:** Replace `localhost:8000` with your server's IP address or domain.

## 🎯 System Requirements & Resource Planning

### Minimum System Requirements
- **Host Memory**: 16GB+ recommended (8GB minimum)
- **Docker Memory**: 12GB+ recommended (8GB minimum)
- **Docker CPUs**: 6+ cores recommended (4 minimum)
- **Disk Space**: 100GB+ free space recommended
- **Raw UMLS Data**: ~25GB
- **Final Database Size**: 30-40GB
- **Peak Space During Loading**: 60-80GB

### Loading Timeline

Expected loading times on modern hardware:

| Phase | Duration | Disk Usage |
|-------|----------|------------|
| **Setup** | 5 min | +1GB |
| **Data Loading** | 1-3 hours | +25GB |
| **Official Indexes** | 30-60 min | +15GB |
| **Cleanup** | 10 min | -20GB |
| **API Optimizations** | 5-15 min (optional) | +1GB |
| **Total** | 2-5 hours | 30-40GB final |

## 📥 Loading UMLS Data (Detailed Instructions)

### Step 1: Obtain UMLS Data

1. Visit [UMLS Terminology Services](https://uts.nlm.nih.gov/uts/)
2. Create an account and accept the license agreement
3. Download the UMLS Metathesaurus files (2025AA release recommended)
4. Extract the `.RRF` files

### Step 2: Prepare Data Files

Unzip your full 2025AA release folder into the `umls-data` folder. This should result in the following files being located within `umls-data/2025AA/META` directory:
- `MRCONSO.RRF` - Concept names and sources (~2.1GB)
- `MRDEF.RRF` - Definitions (~124MB)
- `MRHIER.RRF` - Hierarchical relationships (~5.5GB)
- `MRREL.RRF` - Related concepts (~5.7GB)
- `MRSTY.RRF` - Semantic types (~205MB)
- `MRSAT.RRF` - Attributes (~8.8GB, optional)

### Step 3: Pre-Loading Checklist

Before starting the data load, verify:

**System Resources:**
- [ ] Docker Desktop has 12GB+ RAM allocated
- [ ] Docker has 100GB+ disk space available
- [ ] Host system has sufficient free space
- [ ] Docker Desktop restarted after resource changes

**Database Setup:**
- [ ] MySQL container is running (`docker compose ps`)
- [ ] Database connection tested
- [ ] No other heavy processes running

**Data Preparation:**
- [ ] UMLS 2025AA files in `umls-data/` directory
- [ ] All required .RRF files present
- [ ] Files are not corrupted

### Step 4: Load Data

Run the loading script:
```bash
./scripts/load_umls_2025aa.sh
```

This script will:
- Create the necessary UMLS table structures using official UMLS scripts
- Load data from .RRF files with progress tracking
- Create optimized indexes for API performance
- Verify the data load with statistics
- **Optionally prompt for API performance optimizations** (recommended)

The script will ask if you want to run additional API-focused optimizations at the end. These create extra indices that improve query performance for the API endpoints.

**Note**: Loading can take 3-7 hours depending on your hardware and data size. Optional optimizations add 5-15 minutes.

## 🏗 Architecture

The system consists of two main components:

1. **UMLS API** (Port 8000): A FastAPI application that directly queries the UMLS database
2. **MCP Server**: A Model Context Protocol server that integrates with Claude Desktop via stdio

### Workflow

1. User sends a natural language query through Claude Desktop
2. Claude Desktop communicates with the MCP server via the MCP protocol
3. MCP server processes the request and calls the appropriate UMLS API endpoint (port 8000)
4. MCP server returns structured data to Claude Desktop
5. Claude Desktop formats the result into natural language
6. User receives a human-readable response

## 🔧 MCP Server Tools

The MCP server provides these tools to Claude Desktop:

| Tool | Description |
|------|-------------|
| `search_terms` | Search for medical terms in UMLS database by ontology |
| `search_cui` | Search for CUIs (Concept Unique Identifiers) by term |
| `get_cui_info` | Get detailed information about a specific CUI |
| `get_cui_ancestors` | Get all ancestor CUIs in the hierarchy |
| `get_cui_depth` | Get the depth of a CUI in the hierarchical structure |
| `find_lowest_common_ancestor` | Find the lowest common ancestor (LCA) of two CUIs |
| `wu_palmer_similarity` | Compute Wu-Palmer similarity between two CUIs |
| `get_hpo_term` | Get HPO (Human Phenotype Ontology) term and code from a CUI |

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/terms?search={term}&ontology={ontology}` | Search for a term (Default: `HPO`) |
| `GET`  | `/cuis?query={term}` | Search for CUIs matching a given term |
| `GET`  | `/cuis/{cui}` | Get details about a specific CUI |
| `GET`  | `/cuis/{cui}/depth` | Get depth of a CUI in the hierarchy |
| `GET`  | `/cuis/{cui}/ancestors` | Retrieve all ancestors of a CUI |
| `GET`  | `/cuis/{cui1}/{cui2}/lca` | Get the lowest common ancestor (LCA) of two CUIs |
| `GET`  | `/cuis/{cui1}/{cui2}/similarity/wu-palmer` | Compute Wu-Palmer similarity between two CUIs |
| `GET`  | `/cuis/{cui}/hpo` | Retrieve HPO code from CUI |
| `GET`  | `/hpo_to_cui/{hpo_code}` | Retrieve CUI from HPO code |

### Endpoint Descriptions

#### Search for Terms
`GET /terms?search={term}&ontology={ontology}`
- Searches for a term in the specified ontology (default: `HPO`).

#### Search for CUI by HPO code
`GET /hpo_to_cui/{hpo_code}`
- Searches for CUI by HPO code.

#### Search for CUIs by Term
`GET /cuis?query={term}`
- Finds CUIs that match a given search term.

#### Get CUI Information
`GET /cuis/{cui}`
- Fetches details about a given CUI, including name and description.

#### Get Depth of a CUI
`GET /cuis/{cui}/depth`
- Determines the depth of a CUI within the hierarchy.

#### Retrieve Ancestors of a CUI
`GET /cuis/{cui}/ancestors`
- Retrieves all ancestor CUIs of a given CUI.

#### Find Lowest Common Ancestor (LCA)
`GET /cuis/{cui1}/{cui2}/lca`
- Finds the lowest common ancestor of two CUIs.

#### Compute Wu-Palmer Similarity
`GET /cuis/{cui1}/{cui2}/similarity/wu-palmer`
- Computes Wu-Palmer similarity between two CUIs based on hierarchical depth.

#### Find HPO from CUI term
`GET /cuis/{cui}/hpo`
- Retrieves the HPO term and its corresponding code associated with a given CUI.

##  Option: Add MONDO dataset and MONDO code searching API (For Docker setup)
- 1: Download the MONDO json edition form this link: https://mondo.monarchinitiative.org/pages/download/
- 2: Move the downloaded MONDO json file to our umls-server folder
- 3: Run the loading script:

  ```cd ~/umls-server```

  ```bash scripts/merge_mondo_umls.sh```
  
- 4: The script only generates SQL, now we got the insert_mondo.sql file, we need to import it into the DB that already contains the UMLS schema:
   enter our umls-server repo root file
  
   ```cd ~/umls-server```
  
   copy the insert_mondo.sql into docker
  
   ```docker cp insert_mondo.sql umls-mysql:/tmp/insert_mondo.sql```
  
   in order to put the SQL data into our dataset, first open docker SQL server
  
   ```
   cd ~/umls-server
   docker compose up -d
   ```

   Then run the SQL code, you need input the database password you set before
  
   ```
   docker exec -it umls-mysql bash -lc \ 'mysql -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DATABASE" < /tmp/insert_mondo.sql'
   ```
  
- 5: Enter database add MONDO into the MRSAB table:

   ```
   docker exec -it umls-mysql mysql -u root -p umls
   ```
  
copy this SQL code and run it:

```
INSERT INTO MRSAB (
  VSAB, RSAB, SON, SF, SVER,
  VSTART, VEND, IMETA, SRL,
  LAT, CENC, CURVER, SABIN,
  SSN, SCIT
) VALUES (
  'MONDO2025',
  'MONDO',
  'MONDO Disease Ontology',
  'MONDO',
  'v2025-01-01',
  '20250101',
  NULL,
  '0',
  0,
  'ENG',
  'NLM',
  'Y',
  'Y',
  'Imported MONDO ontology',
  'Source: MONDO release'
);
```


- 6: Verify the import, Check that MONDO rows were inserted into MRCONSO:

   ```docker exec -it umls-mysql bash -lc \ 'mysql -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DATABASE" -e "SELECT CUI, CODE, STR FROM MRCONSO WHERE SAB=\"MONDO\" LIMIT 10;"'```

- 7: Restart the API

  ```docker restart umls-api```
  
- 8: Now you can use our getting MONDO code API:

  `GET /cuis/{cui}/mondo` -summary:Get MONDO term and code from CUI
  
  `GET /terms/{term}/mondo` -summary:Get MONDO code by term

## 🏗️ UMLS Database Schema

The setup creates these main tables:

### MRCONSO (Concept Names and Sources)
- Primary table containing all concept information
- Key fields: CUI, STR, SAB, CODE, TTY
- Indexes on CUI, SAB, CODE, STR, AUI

### MRDEF (Definitions)
- Contains definitions for concepts
- Key fields: CUI, DEF, SAB
- Linked to MRCONSO via CUI

### MRHIER (Hierarchical Relationships)  
- Contains parent-child relationships
- Key fields: CUI, PTR, SAB
- Used for ancestor/descendant queries

### MRREL (Related Concepts)
- Contains relationships between concepts
- Key fields: CUI1, CUI2, REL, RELA
- Used for similarity calculations

## 🛠️ Management Commands

### Start Services
```bash
# Start MySQL only
docker compose up -d mysql

# Start all services
docker compose up -d

# Start with logs
docker compose up
```

### Monitor Services
```bash
# View logs
docker compose logs -f

# View MySQL logs specifically  
docker compose logs -f mysql

# View API logs specifically
docker compose logs -f umls-api
```

### Database Management
```bash
# Connect to MySQL
docker exec -it umls-mysql mysql -u umls_user -p umls

# Run API performance optimizations (if not done during initial load)
./scripts/run_optimization.sh

# Backup database
docker exec umls-mysql mysqldump -u umls_user -p umls > umls_backup.sql

# Restore database
docker exec -i umls-mysql mysql -u umls_user -p umls < umls_backup.sql
```

### Cleanup
```bash
# Stop services
docker compose down

# Remove all data (⚠️ This will delete your UMLS database!)
docker compose down -v

# Remove images
docker compose down --rmi all
```

## 🔧 Troubleshooting

### Container Issues

**MySQL container won't start:**
```bash
# Check logs
docker compose logs mysql

# Common issues:
# - Insufficient memory (need at least 8GB)
# - Port 3306 already in use
# - Permission issues with volumes
```

**API container can't connect to MySQL:**
```bash
# Verify MySQL is running
docker compose ps

# Check network connectivity
docker exec umls-api ping mysql

# Verify environment variables
docker exec umls-api env | grep DB_
```

### Performance Issues

**Slow queries:**
- Ensure indexes are created (run the load script with optimizations)
- Run additional API optimizations: `./scripts/run_optimization.sh`
- Increase MySQL buffer pool size in docker-compose.yml
- Monitor with: `docker stats`

**High memory usage:**
- Adjust `innodb_buffer_pool_size` in docker-compose.yml
- Monitor with: `docker exec umls-mysql mysql -e "SHOW ENGINE INNODB STATUS"`

### Data Loading Issues

**Load script fails:**
```bash
# Check if files exist
ls -la umls-data/

# Verify file format (should be pipe-delimited)
head -n 5 umls-data/MRCONSO.RRF

# Check MySQL permissions
docker exec umls-mysql mysql -u umls_user -p -e "SELECT USER(), DATABASE();"
```

### Resource Monitoring During Load

**Monitor system resources:**
```bash
# Check Docker resource usage
docker stats --no-stream

# Monitor disk space in real-time
watch -n 30 'docker system df && df -h'

# Monitor MySQL data directory
docker exec umls-mysql df -h /var/lib/mysql
```

**Monitor MySQL processes:**
```bash
# Check active queries
docker exec umls-mysql mysql -u umls_user -p -e "SHOW PROCESSLIST;"

# Monitor MySQL status
docker exec umls-mysql mysql -u umls_user -p -e "SHOW ENGINE INNODB STATUS\G"
```

### Space Management

**If you run out of space during loading:**
```bash
# Clean Docker cache
docker system prune -a -f
docker volume prune -f

# Remove unused images
docker image prune -a -f

# Check MySQL data usage
docker exec umls-mysql du -sh /var/lib/mysql
```

**Emergency space recovery:**
```bash
# Stop containers
docker compose down

# Clean everything except volumes
docker system prune -a -f

# Remove temporary files
docker exec umls-mysql find /tmp -name "*.RRF" -delete

# Restart with more space
docker compose up -d mysql
```

## 📊 Performance Tuning

### MySQL Optimization

For production use, consider adjusting these MySQL settings in `docker-compose.yml`:

```yaml
command: >
  --innodb_buffer_pool_size=4G
  --innodb_log_file_size=512M
  --max_connections=500
  --query_cache_size=128M
  --tmp_table_size=64M
  --max_heap_table_size=64M
```

### Query Performance

With proper indexing, expect:
- **Simple term searches**: < 100ms
- **Complex relationship queries**: 200ms - 2s
- **Hierarchical traversals**: 100ms - 1s
- **Cross-ontology mappings**: 500ms - 5s

## 🔒 Security Considerations

- Change default passwords in `.env` file
- Restrict MySQL port access if not needed externally
- Use firewall rules to limit API access
- Keep UMLS data secure and comply with license terms
- Regular security updates for Docker images

## 📦 Installation (Advanced/Local Setup)

### Prerequisites

- Python 3.8+
- Conda (recommended for environment management)
- MySQL/MariaDB with UMLS data
- Claude Desktop (for MCP integration)

### Local Setup

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd umls-server
   ```

2. Create and activate conda environment:
   ```sh
   conda create -n umls-server python=3.11
   conda activate umls-server
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Set up the UMLS API:
   ```sh
   cd umls_api
   cp .env.example .env
   # Edit .env with your database credentials:
   # DB_HOST=your-db-host
   # DB_USER=your-db-user
   # DB_PASSWORD=your-db-password
   # DB_NAME=your-db-name
   ```

5. Configure Claude Desktop:
   ```sh
   # Copy the sample configuration
   cp claude_desktop_config.json /path/to/claude/config/
   # Edit the configuration with your actual paths
   ```

### Running the Services

1. Start the UMLS API:
   ```sh
   conda activate umls-server
   cd umls_api
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

2. The MCP Server runs automatically when Claude Desktop starts (no manual startup required).

For long-running processes on EC2, use:
```sh
nohup uvicorn app:app --host 0.0.0.0 --port 8000 > umls_api.log 2>&1 &
```

### Setting up as Systemd Service (Linux)

Create systemd service files for the UMLS API:

**umls-api.service**:
```ini
[Unit]
Description=UMLS API Service
After=network.target

[Service]
User=developer
WorkingDirectory=/path/to/umls_api
ExecStart=/path/to/conda/envs/umls-server/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Install and enable the service:
```sh
sudo cp umls-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable umls-api
sudo systemctl start umls-api
```

## 🔒 Security Considerations

- The UMLS API should be secured appropriately
- Consider using HTTPS in production
- Restrict access to the services using a VPN or firewall
- Make sure port 8000 is open in your EC2 Security Group for API access
- The MCP server runs locally and communicates with Claude Desktop via stdio

## 🛠 Troubleshooting (Local Setup)

### UMLS API Issues
- **API not accessible?** Make sure the conda environment is activated and port 8000 is open
- **No results found?** Ensure your ontology (`HPO`, `NCI`, etc.) is correct
- **API stops when logging out?** Use the `nohup` command shown above

### Claude Desktop Integration Issues
- **Tools not showing up?** Check the Claude Desktop configuration file paths
- **MCP server errors?** Ensure the conda environment path is correct in the configuration
- **Permission errors?** Make sure the MCP server script is executable

### Environment Issues
- **Import errors?** Ensure you've activated the conda environment and installed all dependencies
- **API connection errors?** Check that the UMLS API is running on port 8000

## 📝 Next Steps

1. **Configure Claude Desktop**: Update your `claude_desktop_config.json` to use `http://localhost:8000`
2. **Performance Optimization**: If you skipped optimizations during loading, run `./scripts/run_optimization.sh`
3. **Load Additional Ontologies**: Customize the loading script for specific vocabularies
4. **Set Up Monitoring**: Add health checks and monitoring
5. **Backup Strategy**: Implement regular database backups
6. **Scale**: Consider read replicas for high-traffic scenarios

## 🤝 Contributing

1. Fork the repo & create a new branch (`feature-name`)
2. Commit changes (`git commit -m "Added feature"`)
3. Push & open a pull request

## 📜 License

MIT License

---

🎉 **Success!** You now have a powerful UMLS natural language interface running with Docker and MySQL! Your system can handle millions of medical records with optimized performance for real-time queries through Claude Desktop.

