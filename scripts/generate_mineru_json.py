#!/usr/bin/env python3
"""Helper script to generate MinerU JSON for a PDF document.

This script can be run independently to generate MinerU JSON files
that the MinerUExtractor expects.

Usage:
    python scripts/generate_mineru_json.py <path_to_pdf>
    python scripts/generate_mineru_json.py <path_to_pdf> -o <output_dir>
"""

import sys
import subprocess
from pathlib import Path


def generate_mineru_json(pdf_path: Path, output_dir: Path = None) -> Path:
    """Generate MinerU JSON for a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output (defaults to PDF's directory)
        
    Returns:
        Path to the generated JSON file
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    if output_dir is None:
        output_dir = pdf_path.parent
    else:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Expected output JSON path
    expected_json = pdf_path.with_suffix(pdf_path.suffix + ".mineru.json")
    
    print(f"📄 Processing: {pdf_path.name}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Expected JSON: {expected_json}\n")
    
    # Try CLI with correct syntax
    cmd = ["mineru", "-p", str(pdf_path), "-o", str(output_dir)]
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes
        )
        
        if result.returncode != 0:
            print(f"❌ MinerU CLI failed (exit code {result.returncode})")
            print(f"Error: {result.stderr}")
            print(f"\nTry running manually:")
            print(f"  mineru -p \"{pdf_path}\" -o \"{output_dir}\"")
            sys.exit(1)
        
        # Look for generated JSON
        possible_locations = [
            expected_json,
            output_dir / f"{pdf_path.stem}.json",
            output_dir / f"{pdf_path.stem}.mineru.json",
        ]
        
        # Also search recursively
        json_files = list(output_dir.rglob("*.json"))
        
        for location in possible_locations + json_files:
            if location.exists() and location.is_file():
                if location != expected_json:
                    import shutil
                    shutil.copy2(str(location), str(expected_json))
                    print(f"✓ Found JSON at: {location}")
                    print(f"✓ Copied to: {expected_json}")
                else:
                    print(f"✓ JSON generated: {expected_json}")
                return expected_json
        
        print(f"⚠️  MinerU completed but JSON not found in expected locations")
        print(f"   Check: {output_dir}")
        sys.exit(1)
        
    except subprocess.TimeoutExpired:
        print("❌ MinerU timed out after 30 minutes")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ MinerU command not found")
        print("   Install with: pip install mineru[all]")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "-o" else None
    
    try:
        json_path = generate_mineru_json(pdf_path, output_dir)
        print(f"\n✅ Success! JSON ready at: {json_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
