#!/usr/bin/env python3
"""
Error Image Viewer - Web-based version using Streamlit

Features:
- Display marked screenshots from error_imgs folders
- Show judgment_response.txt and aggregation_response.txt content
- Navigate between entries with prev/next buttons
- Jump to specific entry by number
- Zoom functionality using browser's built-in zoom
- Works in any browser - perfect for remote servers
"""

import streamlit as st
import os
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO


def load_error_folders():
    """Load all error folders from the error_imgs directory"""
    error_imgs_path = Path("error_imgs")
    if not error_imgs_path.exists():
        st.error(f"error_imgs directory not found at {error_imgs_path.absolute()}")
        return []
    
    # Get all subdirectories and sort them by name (which includes timestamp)
    error_folders = sorted([
        d for d in error_imgs_path.iterdir() 
        if d.is_dir() and (d / "marked_screenshot.png").exists()
    ])
    
    return error_folders


def load_text_file(file_path):
    """Load text file content"""
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return f"No {file_path.name} found"
    except Exception as e:
        return f"Error reading {file_path.name}:\n{str(e)}"


def get_image_base64(image_path):
    """Convert image to base64 for HTML display"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


def main():
    st.set_page_config(
        page_title="Error Image Viewer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 Error Image Viewer")
    st.markdown("Interactive viewer for RegionFocus error images and responses")
    
    # Load error folders
    error_folders = load_error_folders()
    
    if not error_folders:
        st.error("No error folders found!")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.write(f"Total entries: **{len(error_folders)}**")
    
    # Initialize session state
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    
    # Navigation controls
    col1, col2, col3 = st.sidebar.columns([1, 1, 1])
    
    with col1:
        if st.button("◀ Prev", disabled=st.session_state.current_index == 0):
            st.session_state.current_index -= 1
            st.rerun()
    
    with col2:
        current_display = st.session_state.current_index + 1
        st.write(f"**{current_display} / {len(error_folders)}**")
    
    with col3:
        if st.button("Next ▶", disabled=st.session_state.current_index >= len(error_folders) - 1):
            st.session_state.current_index += 1
            st.rerun()
    
    # Jump to entry
    st.sidebar.markdown("---")
    jump_to = st.sidebar.number_input(
        "Jump to entry:", 
        min_value=1, 
        max_value=len(error_folders), 
        value=current_display,
        step=1
    )
    
    if st.sidebar.button("Go to Entry"):
        st.session_state.current_index = jump_to - 1
        st.rerun()
    
    # Get current folder
    current_folder = error_folders[st.session_state.current_index]
    
    # Display folder name
    st.header(f"📁 {current_folder.name}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🖼️ Screenshot")
        
        # Load and display image
        image_path = current_folder / "marked_screenshot.png"
        if image_path.exists():
            try:
                # Load image
                image = Image.open(image_path)
                
                # Display image with zoom controls
                st.image(
                    image, 
                    caption=f"Marked screenshot from {current_folder.name}",
                    use_column_width=True
                )
                
                # Add zoom instructions
                st.info("💡 **Tip**: Use your browser's zoom (Ctrl/Cmd + scroll) to zoom into the image!")
                
                # Image info
                st.caption(f"Image size: {image.width} x {image.height} pixels")
                
            except Exception as e:
                st.error(f"Error loading image: {e}")
        else:
            st.error("marked_screenshot.png not found!")
    
    with col2:
        # Judgment response
        st.subheader("📋 Judgment Response")
        judgment_path = current_folder / "judgment_response.txt"
        judgment_content = load_text_file(judgment_path)
        
        st.text_area(
            "judgment_response.txt",
            value=judgment_content,
            height=300,
            label_visibility="collapsed"
        )
        
        # Aggregation response
        st.subheader("📊 Aggregation Response")
        aggregation_path = current_folder / "aggregation_response.txt"
        aggregation_content = load_text_file(aggregation_path)
        
        st.text_area(
            "aggregation_response.txt",
            value=aggregation_content,
            height=300,
            label_visibility="collapsed"
        )
    
    # Keyboard shortcuts info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### ⌨️ Keyboard Shortcuts
    - **R**: Refresh page
    - **Ctrl/Cmd + Plus**: Zoom in
    - **Ctrl/Cmd + Minus**: Zoom out
    - **Ctrl/Cmd + 0**: Reset zoom
    """)
    
    # Additional info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    ### ℹ️ Current Entry Info
    - **Folder**: `{current_folder.name}`
    - **Has Judgment**: ✅ Yes
    - **Has Aggregation**: {'✅ Yes' if (current_folder / 'aggregation_response.txt').exists() else '❌ No'}
    """)


if __name__ == "__main__":
    main()
