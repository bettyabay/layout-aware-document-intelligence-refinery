"""Document viewer component with bounding box overlay."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from PIL import Image
import fitz  # PyMuPDF


def display_pdf_with_bboxes(
    pdf_path: Path,
    page_number: int,
    bboxes: list[dict] = None,
    highlight_color: tuple = (255, 255, 0, 128)
):
    """
    Display a PDF page with optional bounding box overlays.
    
    Args:
        pdf_path: Path to PDF file
        page_number: Page number to display (1-indexed)
        bboxes: List of bounding boxes to highlight
        highlight_color: RGBA color for highlights
    """
    try:
        # Open PDF
        doc = fitz.open(pdf_path)
        
        if page_number < 1 or page_number > len(doc):
            st.error(f"Invalid page number: {page_number}")
            return
        
        # Get page (0-indexed)
        page = doc[page_number - 1]
        
        # Render page to image
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Draw bounding boxes if provided
        if bboxes:
            from PIL import ImageDraw
            
            draw = ImageDraw.Draw(img, "RGBA")
            
            for bbox in bboxes:
                x0 = bbox.get("x0", 0) * 2  # Scale by zoom factor
                y0 = bbox.get("y0", 0) * 2
                x1 = bbox.get("x1", 0) * 2
                y1 = bbox.get("y1", 0) * 2
                
                # Draw rectangle
                draw.rectangle(
                    [(x0, y0), (x1, y1)],
                    fill=highlight_color,
                    outline=(255, 0, 0, 255),
                    width=2
                )
        
        # Display image
        st.image(img, use_container_width=True)
        
        doc.close()
        
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")


def display_page_navigation(
    total_pages: int,
    current_page: int,
    key_prefix: str = "page_nav"
):
    """
    Display page navigation controls.
    
    Args:
        total_pages: Total number of pages
        current_page: Current page number (1-indexed)
        key_prefix: Prefix for Streamlit widget keys
    """
    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
    
    with col1:
        if st.button("⏮️ First", key=f"{key_prefix}_first"):
            st.session_state[f"{key_prefix}_page"] = 1
            st.rerun()
    
    with col2:
        if st.button("◀️ Previous", key=f"{key_prefix}_prev"):
            if current_page > 1:
                st.session_state[f"{key_prefix}_page"] = current_page - 1
                st.rerun()
    
    with col3:
        page_input = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=current_page,
            key=f"{key_prefix}_input"
        )
        if page_input != current_page:
            st.session_state[f"{key_prefix}_page"] = page_input
            st.rerun()
    
    with col4:
        if st.button("Next ▶️", key=f"{key_prefix}_next"):
            if current_page < total_pages:
                st.session_state[f"{key_prefix}_page"] = current_page + 1
                st.rerun()
        if st.button("Last ⏭️", key=f"{key_prefix}_last"):
            st.session_state[f"{key_prefix}_page"] = total_pages
            st.rerun()
    
    st.caption(f"Page {current_page} of {total_pages}")
