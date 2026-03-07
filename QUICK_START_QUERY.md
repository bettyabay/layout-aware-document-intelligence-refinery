# Quick Start: Query Agent in Web UI

## 🚀 Fast Track (5 Steps)

### Step 1: Start the Web Server
```bash
python -m src.web.app
```
Open: **http://localhost:8000**

### Step 2: Find Your Document
- Go to the home page
- Find your processed document ID (e.g., `2018_Audited_Financial_Statement_Report`)
- Or navigate directly to: `http://localhost:8000/query/{doc_id}`

### Step 3: Set Up Query Agent
- Click **"Set Up Query Agent"** button
- Wait for success message (chunks loaded, facts extracted)

### Step 4: Ask Questions
Try these queries:
- **Semantic**: "What is the total revenue?"
- **Navigation**: "Find sections about financial statements"  
- **SQL**: `SELECT * FROM facts WHERE fact_key LIKE '%revenue%'`

### Step 5: View Provenance
- See the answer
- Check provenance citations (page numbers, bounding boxes)
- Verify by opening the original PDF

## 📋 Full Workflow

1. **Upload** → `http://localhost:8000` (if needed)
2. **Triage** → `http://localhost:8000/triage/{doc_id}`
3. **Extraction** → `http://localhost:8000/extraction/{doc_id}`
4. **Chunks** → `http://localhost:8000/chunks/{doc_id}`
5. **Query** → `http://localhost:8000/query/{doc_id}` ⭐

## 🔧 Troubleshooting

**"No chunks found"** → Process document through extraction pipeline first

**"Setup failed"** → Check dependencies: `pip install chromadb sentence-transformers`

**"Query failed"** → Ensure setup completed successfully, check server logs

## 💡 Tips

- Use **semantic queries** for natural language questions
- Use **navigation queries** to find document sections
- Use **SQL queries** for precise fact extraction
- Always check **provenance** to verify answers
