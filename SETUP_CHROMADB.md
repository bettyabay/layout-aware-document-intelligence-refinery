# ChromaDB Setup Guide

## ✅ Good News: ChromaDB is Already Installed!

ChromaDB was installed when we ran:
```bash
pip install chromadb sentence-transformers
```

## 🎯 What You Need to Do

ChromaDB doesn't need manual setup - it's **automatically initialized** when you use the Query Agent. Here's the flow:

### Step 1: Make Sure You Have Chunks

Before you can query, you need to have processed a document and created chunks. Check if you have chunks:

**Option A: Check in outputs folder**
```bash
ls outputs/*_chunks.json
```

**Option B: Check in .refinery/chunks folder**
```bash
ls .refinery/chunks/*_chunks.json
```

If you have `2018_Audited_Financial_Statement_Report_chunks.json`, you're good!

### Step 2: Access Query Interface

1. Open browser: `http://127.0.0.1:8000`
2. Find your document
3. Click "Query" button
4. Or go directly to: `http://127.0.0.1:8000/query/2018_Audited_Financial_Statement_Report`

### Step 3: Set Up Query Agent (This Initializes ChromaDB)

1. On the query page, click **"Set Up Query Agent"**
2. This will:
   - ✅ Create ChromaDB vector store (`.refinery/vector_store/`)
   - ✅ Load all chunks into ChromaDB with embeddings
   - ✅ Extract facts into SQLite (`.refinery/facts.db`)
   - ✅ Make the Query Agent ready to use

**That's it!** ChromaDB is set up automatically - no manual configuration needed.

## 🔍 Verify ChromaDB is Working

After clicking "Set Up Query Agent", you should see:
- Success message: "Success! Added X chunks and extracted Y facts"
- The query interface becomes available

## 📁 Where ChromaDB Stores Data

ChromaDB data is stored in:
```
.refinery/vector_store/
```

This folder is created automatically when you first use the Query Agent.

## ❌ If Setup Fails

If you see an error when clicking "Set Up Query Agent", check:

1. **Do you have chunks?**
   - Make sure you've processed a document through the full pipeline
   - Chunks should be in `outputs/` or `.refinery/chunks/`

2. **Is ChromaDB installed?**
   ```bash
   python -c "import chromadb; print('ChromaDB installed!')"
   ```

3. **Check server logs**
   - Look at the terminal where the server is running
   - Any errors will be shown there

## 🚀 Quick Start

1. **Server is running** ✅ (confirmed)
2. **Open browser**: `http://127.0.0.1:8000`
3. **Go to Query page**: Click "Query" on your document
4. **Click "Set Up Query Agent"** - This initializes ChromaDB automatically
5. **Start querying!**

No manual ChromaDB setup needed - it's all automatic! 🎉
