# Using Google Colab for GPU-Accelerated Extraction

Google Colab provides free GPU access (Tesla T4) which can significantly speed up MinerU and Docling processing, especially for OCR-heavy documents.

## Quick Start

### 1. Access Google Colab
- Go to: https://colab.research.google.com/
- Sign in with your Google account
- Create a new notebook or upload your existing `.ipynb` files

### 2. Enable GPU Runtime
1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU** (T4)
3. Click **Save**

### 3. Upload Your Notebooks

**Option A: Direct Upload**
- File → Upload notebook → Select your `.ipynb` files from:
  - `notebooks/test_layout_extractor_docling.ipynb`
  - `notebooks/test_layout_extractor_mineru.ipynb`

**Option B: Mount Google Drive** (for persistent storage)
```python
from google.colab import drive
drive.mount('/content/drive')

# Your files will be at: /content/drive/MyDrive/...
```

### 4. Install Dependencies

Add this cell at the top of your Colab notebook:

```python
# Install project dependencies
!pip install -q docling mineru[all] pdfplumber pydantic

# Install your project (if you upload the full repo)
# !pip install -q -e /content/drive/MyDrive/layout-aware-document-intelligence-refinery
```

### 5. Upload Your Data Files

**Option A: Direct Upload (small files)**
```python
from google.colab import files
uploaded = files.upload()  # Select your PDF files
```

**Option B: Google Drive (recommended for large files)**
```python
# After mounting drive, copy files:
!cp /content/drive/MyDrive/your-data/*.pdf /content/data/
```

**Option C: Download from URL**
```python
!wget -O /content/data/document.pdf "https://your-url/document.pdf"
```

### 6. Update File Paths in Notebooks

In your Colab notebook, update paths:

```python
# Instead of Windows paths like:
# pdf_path = Path(r"C:\Users\...\data\class_b\file.pdf")

# Use Colab paths:
pdf_path = Path("/content/data/class_b/file.pdf")

# Or if using Drive:
pdf_path = Path("/content/drive/MyDrive/data/class_b/file.pdf")
```

## Complete Setup Example

Here's a complete setup cell for Colab:

```python
# ============================================
# Colab Setup Cell
# ============================================

# 1. Install dependencies
!pip install -q docling mineru[all] pdfplumber pydantic fastapi uvicorn jinja2

# 2. Mount Google Drive (optional, for persistent storage)
from google.colab import drive
drive.mount('/content/drive')

# 3. Clone or upload your project
# Option A: Clone from GitHub (if your repo is public)
# !git clone https://github.com/yourusername/layout-aware-document-intelligence-refinery.git
# %cd layout-aware-document-intelligence-refinery

# Option B: Upload via Drive
# Upload your project folder to Google Drive, then:
# %cd /content/drive/MyDrive/layout-aware-document-intelligence-refinery

# 4. Set up paths
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

# 5. Create data directory
!mkdir -p /content/data
!mkdir -p /content/data/class_a
!mkdir -p /content/data/class_b
!mkdir -p /content/data/class_c
!mkdir -p /content/data/class_d

# 6. Upload your PDF files (run this cell, then select files)
from google.colab import files
print("Upload your PDF files:")
uploaded = files.upload()

# Move uploaded files to data directory
import shutil
for filename in uploaded.keys():
    if filename.endswith('.pdf'):
        class_type = 'class_a'  # Adjust based on your file
        shutil.move(filename, f'/content/data/{class_type}/{filename}')
        print(f"Moved {filename} to /content/data/{class_type}/")

print("\n✅ Setup complete! GPU is enabled.")
print(f"GPU: {!nvidia-smi}")  # Check GPU status
```

## GPU Verification

Check if GPU is available:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# For Docling
try:
    from docling.document_converter import DocumentConverter
    print("✅ Docling installed")
except ImportError:
    print("❌ Docling not installed")
```

## Speed Improvements

With GPU acceleration:
- **Docling OCR**: 5-10x faster on GPU vs CPU
- **MinerU**: 3-5x faster with GPU acceleration
- **Large documents**: Especially beneficial for 50+ page documents

## Important Notes

1. **Session Limits**: Free Colab has usage limits (~12 hours/day, may disconnect after inactivity)
2. **File Persistence**: Files in `/content` are deleted when session ends. Use Google Drive for persistence
3. **Memory**: Free tier has ~15GB RAM, which should be enough for most documents
4. **Upload Limits**: Large PDFs (>100MB) may take time to upload

## Alternative: Local GPU Setup

If you have an NVIDIA GPU locally:

1. Install CUDA toolkit
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. Docling and MinerU will automatically use GPU if available

## Troubleshooting

**GPU not available:**
- Check Runtime → Change runtime type → GPU is selected
- Free tier may have limited availability during peak hours

**Out of memory:**
- Process documents in smaller batches
- Use `do_ocr=False` for digital PDFs (faster, less memory)

**Slow uploads:**
- Use Google Drive instead of direct upload
- Compress PDFs if possible

## Quick Reference

```python
# Check GPU
!nvidia-smi

# Check disk space
!df -h

# Check memory
!free -h

# List files
!ls -lh /content/data/
```
