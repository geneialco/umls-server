#!/bin/bash
set -e

# Ensure mondo.json exists
if [ ! -f "mondo.json" ]; then
  echo "❌ mondo.json not found. Please put it in the current directory."
  exit 1
fi

# Run inline Python script
python3 <<'EOF'
import json

with open("mondo.json", "r") as f:
    data = json.load(f)

print(len(data.get("graphs", [])[0].get("nodes", [])), "terms loaded")

sql_lines = []

def normalize_mondo_id(mondo_id: str):
    if mondo_id.startswith("http://purl.obolibrary.org/obo/"):
        mondo_core = mondo_id.split("/")[-1]   # MONDO_0004995
        mondo_std = mondo_core.replace("_", ":")  # MONDO:0004995
        mondo_num = mondo_core.split("_")[1]      # 0004995
        return mondo_std, mondo_num
    return mondo_id, mondo_id

def clean_label(label: str):
    if not label:
        return ""
    return label.replace('"', '\\"').replace("'", "\\'")

for node in data.get("graphs", [])[0].get("nodes", []):
    mondo_raw = node.get("id", "")
    label = clean_label(node.get("lbl", ""))
    xrefs = [x["val"] if isinstance(x, dict) else x for x in node.get("meta", {}).get("xrefs", [])]

    umls_ids = [x.split(":")[1] for x in xrefs if isinstance(x, str) and x.startswith("UMLS:")]
    if not umls_ids:
        continue

    mondo_std, mondo_num = normalize_mondo_id(mondo_raw)

    for umls_cui in umls_ids:
        sql = (
            f"INSERT INTO MRCONSO "
            f"(CUI, LAT, TS, LUI, STT, SUI, ISPREF, AUI, SCUI, SAB, TTY, CODE, STR, SRL, SUPPRESS) "
            f"VALUES ('{umls_cui}', 'ENG', 'P', 'LM{mondo_num}', 'PF', 'SM{mondo_num}', 'Y', "
            f"'AM{mondo_num}', '{mondo_std}', 'MONDO', 'PT', '{mondo_std}', \"{label}\", 0, 'N');"
        )
        sql_lines.append(sql)

print("Generated SQL statements:", len(sql_lines))

with open("insert_mondo.sql", "w") as f:
    f.write("\n".join(sql_lines))

print("✅ Saved as insert_mondo.sql")
EOF
