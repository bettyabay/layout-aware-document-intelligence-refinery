"""Provenance display component."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from src.models import ProvenanceChain


def display_provenance_chain(provenance: list[ProvenanceChain | dict]):
    """
    Display a provenance chain with citations.
    
    Args:
        provenance: List of ProvenanceChain objects or dicts
    """
    if not provenance:
        st.info("No provenance information available.")
        return
    
    st.subheader("Provenance Chain")
    st.markdown(f"**Total Citations:** {len(provenance)}")
    
    for i, prov in enumerate(provenance, 1):
        # Convert to dict if needed
        if isinstance(prov, ProvenanceChain):
            prov_dict = prov.model_dump()
        else:
            prov_dict = prov
        
        with st.expander(f"Citation {i}: {prov_dict.get('document_name', 'Unknown')} - Page {prov_dict.get('page_number', 'N/A')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Document Information**")
                st.markdown(f"- **Name:** {prov_dict.get('document_name', 'N/A')}")
                st.markdown(f"- **Page:** {prov_dict.get('page_number', 'N/A')}")
                
                bbox = prov_dict.get('bbox', {})
                if bbox:
                    if isinstance(bbox, dict):
                        st.markdown(f"- **Bounding Box:** ({bbox.get('x0', 0):.1f}, {bbox.get('y0', 0):.1f}, "
                                  f"{bbox.get('x1', 0):.1f}, {bbox.get('y1', 0):.1f})")
                    else:
                        st.markdown(f"- **Bounding Box:** {bbox}")
            
            with col2:
                st.markdown("**Metadata**")
                content_hash = prov_dict.get('content_hash', 'N/A')
                st.markdown(f"- **Content Hash:** `{content_hash[:16]}...`" if len(content_hash) > 16 else f"- **Content Hash:** `{content_hash}`")
                
                confidence = prov_dict.get('confidence')
                if confidence is not None:
                    st.markdown(f"- **Confidence:** {confidence:.2%}")
            
            # Display text excerpt if available
            text_excerpt = prov_dict.get('text_excerpt')
            if text_excerpt:
                st.markdown("**Text Excerpt:**")
                st.code(text_excerpt, language=None)
            
            # Visual bbox representation
            if bbox and isinstance(bbox, dict):
                st.markdown("**Bounding Box Visualization:**")
                visualize_bbox(bbox)


def visualize_bbox(bbox: dict, width: int = 400, height: int = 100):
    """
    Visualize a bounding box as a simple diagram.
    
    Args:
        bbox: Bounding box dict with x0, y0, x1, y1
        width: Display width
        height: Display height
    """
    try:
        from PIL import Image, ImageDraw
        
        # Create a simple visualization
        img = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(img)
        
        # Scale bbox to fit visualization
        x0 = bbox.get("x0", 0)
        y0 = bbox.get("y0", 0)
        x1 = bbox.get("x1", 0)
        y1 = bbox.get("y1", 0)
        
        # Normalize to visualization size
        scale_x = width / max(x1 - x0, 1)
        scale_y = height / max(y1 - y0, 1)
        
        vis_x0 = x0 * scale_x
        vis_y0 = y0 * scale_y
        vis_x1 = x1 * scale_x
        vis_y1 = y1 * scale_y
        
        # Draw rectangle
        draw.rectangle(
            [(vis_x0, vis_y0), (vis_x1, vis_y1)],
            fill=(255, 255, 0, 128),
            outline=(255, 0, 0),
            width=2
        )
        
        st.image(img, use_container_width=False)
        
    except Exception as e:
        st.caption(f"BBox: ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})")


def display_provenance_summary(provenance: list[ProvenanceChain | dict]):
    """
    Display a compact provenance summary.
    
    Args:
        provenance: List of ProvenanceChain objects or dicts
    """
    if not provenance:
        return
    
    st.caption(f"📎 {len(provenance)} citation(s)")
    
    # Group by document
    by_doc = {}
    for prov in provenance:
        if isinstance(prov, ProvenanceChain):
            doc_name = prov.document_name
            page_num = prov.page_number
        else:
            doc_name = prov.get("document_name", "Unknown")
            page_num = prov.get("page_number", 0)
        
        if doc_name not in by_doc:
            by_doc[doc_name] = []
        by_doc[doc_name].append(page_num)
    
    # Display grouped citations
    citations = []
    for doc_name, pages in by_doc.items():
        unique_pages = sorted(set(pages))
        if len(unique_pages) == 1:
            citations.append(f"{doc_name}, p.{unique_pages[0]}")
        else:
            citations.append(f"{doc_name}, pp.{unique_pages[0]}-{unique_pages[-1]}")
    
    st.markdown(" | ".join(citations))
