#!/bin/bash

# UMLS 2025AA Data Loading Script using Official UMLS Scripts
# Uses official mysql_tables.sql and mysql_indexes.sql for standard compliance
# Automatically handles @LINE_TERMINATION@ placeholder replacement

set -e

echo "🏥 UMLS 2025AA Data Loading Script (Official)"
echo "=============================================="
echo "Loading UMLS Spring 2025 Base Release (2025AA)"
echo "Using official UMLS mysql_tables.sql and mysql_indexes.sql"
echo ""

# Configuration
# Only these two values need to be customized:
# - CONTAINER_NAME: Docker container name for MySQL
# - UMLS_DATA_DIR: Path to UMLS data directory
# All database credentials are read from .env file
CONTAINER_NAME="umls-mysql"
UMLS_DATA_DIR="umls-data/2025AA/META"

# Load database configuration from .env file
if [ -f ".env" ]; then
    echo "📋 Loading database configuration from .env file..."
    export $(grep -v '^#' .env | xargs)
    DB_NAME="${DB_NAME:-umls}"
    DB_USER="${DB_USER:-umls_user}"
    DB_PASSWORD="${DB_PASSWORD:-umls_password}"
    MYSQL_ROOT_PASSWORD="${MYSQL_ROOT_PASSWORD:-umls_root_password}"
    echo "   Database: $DB_NAME"
    echo "   User: $DB_USER"
    echo "   Host: ${DB_HOST:-localhost}"
else
    echo "⚠️  No .env file found, using defaults"
    DB_NAME="umls"
    DB_USER="umls_user"
    DB_PASSWORD="umls_password"
    MYSQL_ROOT_PASSWORD="umls_root_password"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if Docker Compose is running
echo -n "🐳 Checking MySQL container... "
if ! docker compose ps | grep -q "$CONTAINER_NAME"; then
    echo -e "${RED}❌ MySQL container is not running${NC}"
    echo "Please start with: docker compose up -d mysql"
    exit 1
fi
echo -e "${GREEN}✅ Running${NC}"

# Test database connection
echo -n "🔐 Testing database connection... "
if docker exec $CONTAINER_NAME mysql -u$DB_USER -p$DB_PASSWORD $DB_NAME -e "SELECT 1;" &>/dev/null; then
    echo -e "${GREEN}✅ Connected successfully${NC}"
else
    echo -e "${RED}❌ Cannot connect to database${NC}"
    echo "Please check your database credentials in .env file"
    exit 1
fi

# Ensure local_infile is enabled globally
echo -n "🔧 Configuring MySQL for data loading... "
docker exec $CONTAINER_NAME mysql -u root -p$MYSQL_ROOT_PASSWORD -e "SET GLOBAL local_infile = 1;" &>/dev/null
echo -e "${GREEN}✅${NC}"

# Clean up temp files
echo -n "🧹 Cleaning temporary files... "
docker exec $CONTAINER_NAME bash -c "rm -f /tmp/*.RRF /tmp/*.rrf" &>/dev/null
echo -e "${GREEN}✅${NC}"

# Note: Using official UMLS mysql_tables.sql and mysql_indexes.sql for data loading
# This ensures compatibility with the standard UMLS installation process

# Show data file summary
echo ""
echo -e "${BLUE}📋 Data Files Summary:${NC}"
for file in "MRCONSO.RRF" "MRDEF.RRF" "MRHIER.RRF" "MRREL.RRF" "MRSTY.RRF" "MRSAT.RRF"; do
    if [ -f "$UMLS_DATA_DIR/$file" ]; then
        size=$(ls -lh "$UMLS_DATA_DIR/$file" | awk '{print $5}')
        echo "   ✅ $file: $size"
    else
        echo "   ❌ $file: Not found"
    fi
done

echo ""
echo -e "${YELLOW}⚠️  This will use the official UMLS scripts to create tables and load ALL data files.${NC}"
echo -e "${YELLOW}⚠️  Process will take 1-3 hours and cannot be interrupted safely.${NC}"
echo ""
read -p "Continue with full UMLS data loading? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user"
    exit 0
fi

echo ""
echo "🚀 Starting UMLS data loading using official scripts..."

# Step 1: Copy RRF files to container data directory
echo ""
echo -e "${BLUE}📁 Preparing data files in container...${NC}"
# Copy RRF files and subdirectories to container
echo -n "   Copying RRF files and subdirectories to container... "
docker exec $CONTAINER_NAME mkdir -p /tmp/umls_data &>/dev/null
for file in "$UMLS_DATA_DIR"/*.RRF; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        docker cp "$file" "$CONTAINER_NAME:/tmp/umls_data/$filename" &>/dev/null
    fi
done
# Copy all subdirectories (e.g., CHANGE) recursively
for dir in "$UMLS_DATA_DIR"/*/; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        docker cp "$dir" "$CONTAINER_NAME:/tmp/umls_data/$dirname" &>/dev/null
    fi
done
echo -e "${GREEN}✅${NC}"

# Create empty MRCXT.RRF if it doesn't exist (required by mysql_tables.sql)
docker exec $CONTAINER_NAME touch /tmp/umls_data/MRCXT.RRF &>/dev/null

# Step 2: Execute official mysql_tables.sql (creates tables + loads data)
echo ""
echo -e "${BLUE}🔧 Creating tables and loading data using official mysql_tables.sql...${NC}"
echo -n "   This will take 1-3 hours depending on your system... "

# Copy mysql_tables.sql to container and preprocess it
docker cp "$UMLS_DATA_DIR/mysql_tables.sql" "$CONTAINER_NAME:/tmp/mysql_tables_original.sql" &>/dev/null

# Preprocess the SQL file to replace @LINE_TERMINATION@ with actual line termination
docker exec $CONTAINER_NAME bash -c "
    sed 's/@LINE_TERMINATION@/\"\\\\n\"/g' /tmp/mysql_tables_original.sql > /tmp/mysql_tables.sql
    rm /tmp/mysql_tables_original.sql
" &>/dev/null

# Copy the official SQL files into the container
docker cp "$UMLS_DATA_DIR/mysql_tables.sql" $CONTAINER_NAME:/tmp/mysql_tables.sql
if [ -f "$UMLS_DATA_DIR/mysql_indexes.sql" ]; then
    docker cp "$UMLS_DATA_DIR/mysql_indexes.sql" $CONTAINER_NAME:/tmp/mysql_indexes.sql
fi

# Execute the official script from the data directory
if docker exec $CONTAINER_NAME bash -c "
    cd /tmp/umls_data
    mysql --local-infile=1 -u$DB_USER -p$DB_PASSWORD $DB_NAME < /tmp/mysql_tables.sql
" &>/dev/null; then
    echo -e "${GREEN}✅ Tables created and data loaded successfully${NC}"
else
    echo -e "${RED}❌ Failed to create tables and load data${NC}"
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Check that all RRF files are present and accessible"
    echo "2. Verify MySQL has sufficient disk space"
    echo "3. Check the last few lines of the error:"
    echo ""
    # Show last few lines of error for debugging
    docker exec $CONTAINER_NAME bash -c "
        cd /tmp/umls_data
        mysql --local-infile=1 -u$DB_USER -p$DB_PASSWORD $DB_NAME < /tmp/mysql_tables.sql
    " 2>&1 | tail -10
    echo ""
    echo "If you see '@LINE_TERMINATION@' errors, the script has been updated to handle this."
    exit 1
fi

# Step 3: Create indexes using official mysql_indexes.sql
echo ""
echo -e "${BLUE}🔍 Creating indexes using official mysql_indexes.sql...${NC}"
echo -n "   This may take 30-60 minutes... "

if docker cp "$UMLS_DATA_DIR/mysql_indexes.sql" "$CONTAINER_NAME:/tmp/mysql_indexes.sql" &>/dev/null; then
    if docker exec $CONTAINER_NAME mysql -u$DB_USER -p$DB_PASSWORD $DB_NAME < /tmp/mysql_indexes.sql &>/dev/null; then
        echo -e "${GREEN}✅ Indexes created successfully${NC}"
    else
        echo -e "${RED}❌ Failed to create indexes${NC}"
        echo "Database may still be functional but queries will be slower"
    fi
else
    echo -e "${RED}❌ Failed to copy mysql_indexes.sql${NC}"
fi

# Clean up temporary files
docker exec $CONTAINER_NAME rm -rf /tmp/umls_data /tmp/mysql_tables.sql /tmp/mysql_indexes.sql &>/dev/null

echo ""
echo -e "${GREEN}🎉 UMLS 2025AA loading completed successfully using official scripts!${NC}"
echo ""
echo "📊 Final database statistics:"
docker exec $CONTAINER_NAME mysql -u$DB_USER -p$DB_PASSWORD $DB_NAME -e "
SELECT 
    TABLE_NAME as 'Table',
    TABLE_ROWS as 'Records',
    ROUND((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024, 2) AS 'Size (MB)'
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = '$DB_NAME' 
AND TABLE_NAME LIKE 'MR%'
ORDER BY TABLE_ROWS DESC;
"

echo ""
echo -e "${GREEN}✅ Complete UMLS database ready with official schema and indexes!${NC}"
echo ""

# Optional optimization step
echo -e "${BLUE}🔍 Optional: API Performance Optimization${NC}"
echo "Would you like to run additional API-focused optimizations?"
echo "This will create extra indices to improve query performance for the API."
echo "• Adds non-duplicate indices optimized for API queries"
echo "• Takes 5-15 minutes additional time"
echo "• Recommended for production use"
echo ""
read -p "Run API optimizations now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${BLUE}🚀 Running API optimizations...${NC}"
    if ./scripts/run_optimization.sh --from-load-script; then
        echo -e "${GREEN}✅ API optimizations completed successfully!${NC}"
    else
        echo -e "${YELLOW}⚠️  API optimizations failed, but database is still functional${NC}"
        echo "You can run optimizations later with: ./scripts/run_optimization.sh"
    fi
else
    echo "Skipping optimizations. You can run them later with: ./scripts/run_optimization.sh"
fi

echo ""
echo -e "${GREEN}🚀 Ready to start API!${NC}"
echo "Next: docker compose up -d umls-api" 