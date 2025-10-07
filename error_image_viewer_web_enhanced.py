#!/usr/bin/env python3
"""
Error Image Viewer - Enhanced Web-based version with Hover Magnification

Features:
- Display marked screenshots from error_imgs folders
- Show judgment_response.txt and aggregation_response.txt content
- Navigate between entries with prev/next buttons
- Jump to specific entry by number
- ENHANCED: Hover magnification with adjustable zoom (like original tkinter version)
- Works in any browser - perfect for remote servers
"""

import streamlit as st
import os
import sys
import json
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO


def load_error_folders(folder_name="error_imgs"):
    """Load all error folders from the specified directory"""
    error_imgs_path = Path(folder_name)
    if not error_imgs_path.exists():
        st.error(f"{folder_name} directory not found at {error_imgs_path.absolute()}")
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


def load_annotation_metadata(folder_path):
    """Load annotation metadata if available"""
    metadata_path = folder_path / "annotation_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading annotation metadata: {e}")
    return None


def create_annotation_svg(annotation_metadata, display_width, display_height, original_width, original_height):
    """Create SVG overlay for annotations that can be flashed independently"""
    if not annotation_metadata:
        return ""
    
    # Calculate scaling factors
    scale_x = display_width / original_width
    scale_y = display_height / original_height
    
    svg_elements = []
    
    # Add bounding box
    if 'bbox' in annotation_metadata:
        bbox = annotation_metadata['bbox']
        x1 = bbox['x1'] * scale_x
        y1 = bbox['y1'] * scale_y
        x2 = bbox['x2'] * scale_x
        y2 = bbox['y2'] * scale_y
        width = x2 - x1
        height = y2 - y1
        
        svg_elements.append(f'''
            <rect class="annotation-element bbox" 
                  x="{x1}" y="{y1}" width="{width}" height="{height}"
                  fill="none" stroke="{bbox['color']}" stroke-width="{bbox['width']}" 
                  data-original-color="{bbox['color']}"/>
        ''')
    
    # Add predicted point
    if 'pred_point' in annotation_metadata:
        pred = annotation_metadata['pred_point']
        cx = pred['x'] * scale_x
        cy = pred['y'] * scale_y
        r = pred['radius'] * min(scale_x, scale_y)
        
        svg_elements.append(f'''
            <circle class="annotation-element pred-point" 
                    cx="{cx}" cy="{cy}" r="{r}"
                    fill="{pred['color']}" stroke="{pred['color']}" stroke-width="{pred['width']}"
                    data-original-color="{pred['color']}"/>
        ''')
    
    # Add judgment point if present
    if 'judge_point' in annotation_metadata:
        judge = annotation_metadata['judge_point']
        cx = judge['x'] * scale_x
        cy = judge['y'] * scale_y
        r = judge['radius'] * min(scale_x, scale_y)
        
        svg_elements.append(f'''
            <circle class="annotation-element judge-point" 
                    cx="{cx}" cy="{cy}" r="{r}"
                    fill="{judge['color']}" stroke="{judge['color']}" stroke-width="{judge['width']}"
                    data-original-color="{judge['color']}"/>
        ''')
    
    # Add text labels
    if 'labels' in annotation_metadata:
        for label in annotation_metadata['labels']:
            x = label['x'] * scale_x
            y = label['y'] * scale_y
            
            svg_elements.append(f'''
                <text class="annotation-element label" 
                      x="{x}" y="{y}" fill="{label['color']}" 
                      font-family="Arial, sans-serif" font-size="16" font-weight="bold"
                      data-original-color="{label['color']}">{label['text']}</text>
            ''')
    
    return ''.join(svg_elements)


def create_magnifying_glass_html(image_base64, image_width, image_height, annotation_metadata=None):
    """Create HTML with magnifying glass effect"""
    
    # Calculate display size to fit within container while maintaining aspect ratio
    max_width = 900  # Maximum width for the viewer
    max_height = 600  # Maximum height for the viewer
    
    # Calculate scaling to fit within both width and height constraints
    width_scale = max_width / image_width
    height_scale = max_height / image_height
    scale = min(width_scale, height_scale, 1.0)  # Don't upscale, only downscale
    
    display_width = int(image_width * scale)
    display_height = int(image_height * scale)
    
    # Generate SVG overlay for annotations
    annotation_svg = create_annotation_svg(annotation_metadata, display_width, display_height, image_width, image_height)
    
    html_code = f"""
    <div id="scrollable-container" style="
        max-width: 100%; 
        max-height: 650px; 
        overflow: auto; 
        border: 2px solid #ddd; 
        border-radius: 8px; 
        background: #f8f9fa;
        padding: 10px;
    ">
        <div id="image-container" style="position: relative; display: inline-block; overflow: visible;">
            <img id="main-image" 
                 src="data:image/png;base64,{image_base64}" 
                 style="width: {display_width}px; height: {display_height}px; display: block;"
                 onmousemove="showMagnifier(event)"
                 onmouseenter="showMagnifier(event)"
                 onmouseleave="hideMagnifier()">
        
        <!-- SVG overlay for annotations that can flash independently -->
        <svg id="annotation-overlay" style="
            position: absolute; top: 0; left: 0; 
            width: {display_width}px; height: {display_height}px;
            pointer-events: none; z-index: 10;">
            {annotation_svg}
        </svg>
        
        <!-- Magnifying glass window -->
        <div id="magnifier" style="
            position: fixed;
            width: 200px;
            height: 200px;
            border: 3px solid #00ff00;
            border-radius: 50%;
            background: white;
            background-repeat: no-repeat;
            background-size: {display_width * 4}px {display_height * 4}px;
            background-image: url('data:image/png;base64,{image_base64}');
            display: none;
            pointer-events: none;
            z-index: 1000;
            box-shadow: 0 0 30px rgba(0,255,0,0.7);
        "></div>
        </div>
    </div>
    
    <!-- Controls -->
    <div style="margin-top: 15px; text-align: center;">
        <!-- Zoom controls -->
        <div style="margin-bottom: 10px;">
            <button onclick="changeZoom(-0.5)" style="
                background: #ff6666; color: white; border: none; padding: 8px 16px; 
                margin: 5px; border-radius: 4px; cursor: pointer; font-weight: bold;">Zoom Out</button>
            <span id="zoom-level" style="margin: 0 20px; font-weight: bold; color: #0066cc;">Zoom: 400%</span>
            <button onclick="changeZoom(0.5)" style="
                background: #00dd99; color: white; border: none; padding: 8px 16px; 
                margin: 5px; border-radius: 4px; cursor: pointer; font-weight: bold;">Zoom In</button>
        </div>
        
        <!-- Flash highlights button -->
        <div>
            <button id="flash-btn" onclick="toggleFlashing()" style="
                background: #ffaa00; color: white; border: none; padding: 10px 20px; 
                margin: 5px; border-radius: 4px; cursor: pointer; font-weight: bold;
                box-shadow: 0 2px 5px rgba(255,170,0,0.3);">🔆 Flash Highlights</button>
        </div>
    </div>
    
    <script>
        let currentZoom = 4.0;  // Default 400% zoom
        const minZoom = 0.5;    // 50% minimum
        const maxZoom = 5.0;    // 500% maximum
        let isFlashing = false;
        let flashInterval = null;
        
        function showMagnifier(e) {{
            const img = document.getElementById('main-image');
            const magnifier = document.getElementById('magnifier');
            
            // Get image dimensions and position
            const rect = img.getBoundingClientRect();
            
            // Calculate mouse position relative to image
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Calculate background position for magnifier (ensure full zoom window is shown)
            const bgX = -x * currentZoom + 100;  // 100 is half of magnifier width
            const bgY = -y * currentZoom + 100;  // 100 is half of magnifier height
            
            // Position magnifier using fixed positioning (allows it to go outside bounds)
            const magnifierX = e.clientX + 20;  // Offset from cursor
            const magnifierY = e.clientY + 20;
            
            // Ensure magnifier stays within viewport
            const viewportWidth = window.innerWidth;
            const viewportHeight = window.innerHeight;
            const magnifierSize = 200;
            
            let finalX = magnifierX;
            let finalY = magnifierY;
            
            // Adjust position if magnifier would go off-screen
            if (magnifierX + magnifierSize > viewportWidth) {{
                finalX = e.clientX - magnifierSize - 20;  // Show to left of cursor
            }}
            if (magnifierY + magnifierSize > viewportHeight) {{
                finalY = e.clientY - magnifierSize - 20;  // Show above cursor
            }}
            
            // Update magnifier
            magnifier.style.left = finalX + 'px';
            magnifier.style.top = finalY + 'px';
            magnifier.style.backgroundPosition = bgX + 'px ' + bgY + 'px';
            magnifier.style.backgroundSize = ({display_width} * currentZoom) + 'px ' + ({display_height} * currentZoom) + 'px';
            magnifier.style.display = 'block';
        }}
        
        function hideMagnifier() {{
            const magnifier = document.getElementById('magnifier');
            magnifier.style.display = 'none';
        }}
        
        function changeZoom(delta) {{
            currentZoom = Math.max(minZoom, Math.min(maxZoom, currentZoom + delta));
            document.getElementById('zoom-level').textContent = 'Zoom: ' + Math.round(currentZoom * 100) + '%';
            
            // Update magnifier background size for all future hover events
            const magnifier = document.getElementById('magnifier');
            magnifier.style.backgroundSize = ({display_width} * currentZoom) + 'px ' + ({display_height} * currentZoom) + 'px';
        }}
        
        function toggleFlashing() {{
            const flashBtn = document.getElementById('flash-btn');
            const annotationElements = document.querySelectorAll('.annotation-element');
            
            if (!isFlashing) {{
                // Start flashing
                isFlashing = true;
                flashBtn.textContent = '⏹️ Stop Flashing';
                flashBtn.style.background = '#ff4444';
                
                // Create flashing effect for annotation elements only
                flashInterval = setInterval(function() {{
                    const time = Date.now();
                    const brightness = Math.sin(time / 200) * 0.4 + 1.2;  // 0.8 to 1.6
                    const hue = (time / 100) % 360;  // Cycle through colors
                    const saturation = Math.sin(time / 300) * 0.5 + 1.5;  // 1.0 to 2.0
                    
                    // Apply flashing effects only to annotation elements
                    annotationElements.forEach(element => {{
                        const originalColor = element.getAttribute('data-original-color');
                        
                        // Create bright, flashing colors based on original color
                        if (originalColor === 'red') {{
                            const flashColor = `hsl(${{0 + Math.sin(time / 150) * 30}}, 100%, ${{50 + Math.sin(time / 200) * 20}}%)`;
                            element.setAttribute('stroke', flashColor);
                            element.setAttribute('fill', flashColor);
                        }} else if (originalColor === 'green') {{
                            const flashColor = `hsl(${{120 + Math.sin(time / 150) * 30}}, 100%, ${{50 + Math.sin(time / 200) * 20}}%)`;
                            element.setAttribute('stroke', flashColor);
                            element.setAttribute('fill', flashColor);
                        }} else if (originalColor === 'blue') {{
                            const flashColor = `hsl(${{240 + Math.sin(time / 150) * 30}}, 100%, ${{50 + Math.sin(time / 200) * 20}}%)`;
                            element.setAttribute('stroke', flashColor);
                            element.setAttribute('fill', flashColor);
                        }} else if (originalColor === 'orange') {{
                            const flashColor = `hsl(${{30 + Math.sin(time / 150) * 30}}, 100%, ${{50 + Math.sin(time / 200) * 20}}%)`;
                            element.setAttribute('stroke', flashColor);
                            element.setAttribute('fill', flashColor);
                        }} else {{
                            // For any other colors, use rainbow cycling
                            const flashColor = `hsl(${{hue}}, 100%, ${{50 + Math.sin(time / 200) * 20}}%)`;
                            element.setAttribute('stroke', flashColor);
                            element.setAttribute('fill', flashColor);
                        }}
                    }});
                }}, 50);  // Update every 50ms for smooth animation
                
            }} else {{
                // Stop flashing
                isFlashing = false;
                flashBtn.textContent = '🔆 Flash Highlights';
                flashBtn.style.background = '#ffaa00';
                
                if (flashInterval) {{
                    clearInterval(flashInterval);
                    flashInterval = null;
                }}
                
                // Reset annotation elements to original colors
                annotationElements.forEach(element => {{
                    const originalColor = element.getAttribute('data-original-color');
                    element.setAttribute('stroke', originalColor);
                    element.setAttribute('fill', originalColor);
                }});
            }}
        }}
        
        // Keyboard shortcuts for zoom
        document.addEventListener('keydown', function(e) {{
            if (e.target.tagName.toLowerCase() !== 'input' && e.target.tagName.toLowerCase() !== 'textarea') {{
                if (e.key === '+' || e.key === '=') {{
                    e.preventDefault();
                    changeZoom(0.5);
                }} else if (e.key === '-') {{
                    e.preventDefault();
                    changeZoom(-0.5);
                }} else if (e.key === '0') {{
                    e.preventDefault();
                    currentZoom = 4.0;
                    document.getElementById('zoom-level').textContent = 'Zoom: 400%';
                    const magnifier = document.getElementById('magnifier');
                    magnifier.style.backgroundSize = ({display_width} * currentZoom) + 'px ' + ({display_height} * currentZoom) + 'px';
                }} else if (e.key === 'f' || e.key === 'F') {{
                    e.preventDefault();
                    toggleFlashing();
                }}
            }}
        }});
        
        // Clean up flashing when page unloads
        window.addEventListener('beforeunload', function() {{
            if (flashInterval) {{
                clearInterval(flashInterval);
            }}
        }});
    </script>
    """
    
    return html_code


def main():
    # Get folder name from command line arguments
    folder_name = "error_imgs"  # default
    if len(sys.argv) > 1:
        folder_name = sys.argv[1]
    
    st.set_page_config(
        page_title="Error Image Viewer - Enhanced",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 Error Image Viewer - Enhanced")
    st.markdown("Interactive viewer with **hover magnification** for RegionFocus error images")
    st.info(f"📁 Loading error images from: **{folder_name}**")
    
    # Load error folders
    error_folders = load_error_folders(folder_name)
    
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
        st.subheader("🔍 Screenshot with Hover Magnification")
        
        # Load and display image with magnifying glass
        image_path = current_folder / "marked_screenshot.png"
        if image_path.exists():
            try:
                # Load image
                image = Image.open(image_path)
                image_base64 = get_image_base64(image_path)
                
                # Load annotation metadata
                annotation_metadata = load_annotation_metadata(current_folder)
                
                if image_base64:
                    # Calculate display scaling info
                    max_width = 900
                    max_height = 600
                    width_scale = max_width / image.width
                    height_scale = max_height / image.height
                    scale = min(width_scale, height_scale, 1.0)
                    display_width = int(image.width * scale)
                    display_height = int(image.height * scale)
                    
                    # Create magnifying glass HTML with annotation metadata
                    magnifier_html = create_magnifying_glass_html(
                        image_base64, image.width, image.height, annotation_metadata
                    )
                    
                    # Display the interactive image with increased height for scrollable container
                    st.components.v1.html(magnifier_html, height=800)
                    
                    # Instructions
                    st.info("""
                    🔍 **Enhanced Magnification Instructions:**
                    - **Hover** over the image to see magnified view (circular magnifier)
                    - **Scrolling**: If image is large, scroll horizontally/vertically within the frame
                    - **Auto-fit**: Images are automatically scaled to fit the viewer while maintaining aspect ratio
                    - **Zoom controls**: Use +/- buttons or keyboard shortcuts
                    - **Flash highlights**: Click 🔆 button to make ONLY prediction points and annotation boxes flash
                    - **Selective flashing**: Only bright annotation colors flash, background image stays normal
                    - **Keyboard shortcuts**: 
                      - `+` (zoom in), `-` (zoom out), `0` (reset to 400%)
                      - `F` (toggle flashing highlights)
                    - **Default zoom**: 400% (adjustable from 50% to 500%)
                    - **Smart positioning**: Magnifier stays visible even at image edges
                    """)
                    
                    # Image info
                    scale_percent = int(scale * 100)
                    st.caption(f"Original image size: {image.width} x {image.height} pixels | Display size: {display_width} x {display_height} pixels ({scale_percent}% scale)")
                
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
            height=500,
            label_visibility="collapsed"
        )
        
        # Aggregation response
        st.subheader("📊 Aggregation Response")
        aggregation_path = current_folder / "aggregation_response.txt"
        aggregation_content = load_text_file(aggregation_path)
        
        st.text_area(
            "aggregation_response.txt",
            value=aggregation_content,
            height=250,
            label_visibility="collapsed"
        )
    
    # Enhanced features info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 🔍 Enhanced Features
    - **Auto-fit**: Images scaled to fit viewer frame
    - **Scrollable**: Horizontal/vertical scrolling for overflow
    - **Hover Zoom**: Move mouse over image
    - **Adjustable**: 50% to 500% zoom
    - **Default**: 400% magnification
    - **Smart Bounds**: Magnifier always visible
    - **Selective Flash**: Only annotation elements flash
    - **Background Preserved**: Image background stays normal
    - **Keyboard**: `+`, `-`, `0`, `F` keys
    - **Visual**: Bright circular magnifying glass
    - **Target**: Highlights prediction points & boxes only
    """)
    
    # Additional info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    ### ℹ️ Current Entry Info
    - **Entry**: {current_display} of {len(error_folders)}
    - **Folder**: `{current_folder.name}`
    - **Has Judgment**: ✅ Yes
    - **Has Aggregation**: {'✅ Yes' if (current_folder / 'aggregation_response.txt').exists() else '❌ No'}
    """)


if __name__ == "__main__":
    main()
