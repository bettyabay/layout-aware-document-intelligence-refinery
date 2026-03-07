# Step-by-Step Guide: Testing the Query Agent in Web UI

This guide will walk you through testing the complete Query Agent (Stage 5) using the web UI.

## Prerequisites

1. **Python environment** with all dependencies installed:
   ```bash
   pip install -e ".[dev]"
   ```

2. **Processed document** - You need at least one document that has been:
   - Triaged (has a DocumentProfile)
   - Extracted (has extraction results)
   - Chunked (has LDUs saved)

## Step 1: Ensure You Have a Processed Document

If you don't have a processed document yet, you can use the existing one or process a new one:

### Option A: Use Existing Processed Document
Check if you have chunks saved:
```bash
# Check for existing chunks
ls .refinery/chunks/
# Or
ls outputs/*_chunks.json
```

### Option B: Process a New Document
1. Start the web UI (see Step 2)
2. Go to http://localhost:8000
3. Upload a PDF document
4. Wait for processing to complete

## Step 2: Start the Web UI

Open a terminal and run:

```bash
python -m src.web.app
```

Or if you're in the project root:

```bash
cd src/web
python app.py
```

The server will start on **http://localhost:8000**

## Step 3: Navigate to Query Interface

1. Open your browser and go to: **http://localhost:8000**
2. Find your processed document in the list
3. Click on the document ID or navigate directly to:
   ```
   http://localhost:8000/query/{doc_id}
   ```
   
   Replace `{doc_id}` with your actual document ID (e.g., `2018_Audited_Financial_Statement_Report`)

## Step 4: Set Up Query Agent

1. On the query page, you'll see a "Set Up Query Agent" button
2. Click it to:
   - Load chunks into the vector store (ChromaDB)
   - Extract facts into the SQLite fact table
3. Wait for the setup to complete (you'll see a success message)

**What happens behind the scenes:**
- LDUs are embedded and stored in ChromaDB
- Key-value facts are extracted and stored in SQLite
- The Query Agent is ready to answer questions

## Step 5: Test Different Query Types

The Query Agent supports three types of queries:

### A. Natural Language / Semantic Search
Try queries like:
- "What is the total revenue?"
- "What are the key financial metrics?"
- "Tell me about the balance sheet"

**How it works:** Uses vector similarity search to find relevant chunks.

### B. PageIndex Navigation
Try queries like:
- "Find sections about financial statements"
- "Where is the income statement?"
- "Show me sections related to revenue"

**How it works:** Traverses the PageIndex tree to find relevant sections.

### C. SQL Queries (Structured Facts)
Try queries like:
- `SELECT * FROM facts WHERE fact_key LIKE '%revenue%'`
- `SELECT fact_key, fact_value FROM facts WHERE fact_type = 'currency'`

**How it works:** Executes SQL queries on the extracted fact table.

## Step 6: View Results with Provenance

After submitting a query, you'll see:

1. **Answer**: The response to your question
2. **Provenance**: For each citation:
   - Document name
   - Page number
   - Bounding box coordinates
   - Content hash

## Step 7: Verify Provenance

To verify the provenance:

1. Note the page number from the provenance
2. Open the original PDF document
3. Navigate to that page
4. Check the bounding box coordinates to locate the exact region

## Troubleshooting

### "No chunks found"
- **Solution**: Make sure the document has been chunked. Run semantic chunking first.

### "Setup failed"
- **Check**: Ensure ChromaDB dependencies are installed: `pip install chromadb sentence-transformers`
- **Check**: Ensure SQLite is available (usually built-in with Python)

### "Query failed"
- **Check**: Make sure setup completed successfully
- **Check**: Verify the document ID is correct
- **Check**: Look at server logs for error details

### Vector store not working
- **Solution**: Delete `.refinery/vector_store` and try setup again
- **Solution**: Check disk space (ChromaDB needs space for embeddings)

## Example Workflow

Here's a complete example workflow:

1. **Upload document**: `http://localhost:8000` → Upload PDF
2. **View triage**: `http://localhost:8000/triage/{doc_id}` → See classification
3. **View extraction**: `http://localhost:8000/extraction/{doc_id}` → See extracted content
4. **View chunks**: `http://localhost:8000/chunks/{doc_id}` → See semantic chunks
5. **Query document**: `http://localhost:8000/query/{doc_id}` → Ask questions!

## Advanced: Using the Query Agent Programmatically

You can also use the Query Agent in Python:

```python
from src.utils.query_helpers import create_query_agent

# Initialize
query_agent = create_query_agent()

# Query
result = query_agent.query(
    query="What is the total revenue?",
    doc_id="2018_Audited_Financial_Statement_Report",
    doc_name="2018_Audited_Financial_Statement_Report.pdf"
)

print(result["answer"])
for prov in result["provenance_chain"]:
    print(f"Page {prov['page_number']}: {prov['document_name']}")
```

## Next Steps

- Try different query types
- Test with multiple documents
- Verify provenance accuracy
- Experiment with SQL queries on fact tables
