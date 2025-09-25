#!/bin/bash
# Enhanced Web-based Error Image Viewer launcher with hover magnification

# Get folder name from command line argument, default to "error_imgs"
FOLDER_NAME=${1:-"error_imgs_qwen25vl_72b"}

echo "🔍 Starting Enhanced Error Image Viewer (Web-based with Hover Magnification)..."
echo "📁 Using folder: $FOLDER_NAME"
echo ""
echo "✨ ENHANCED FEATURES:"
echo "  - Hover magnification with 400% default zoom"
echo "  - Smart bounds: magnifier always stays visible"
echo "  - Selective flash: ONLY annotation elements flash"
echo "  - Background preserved: Image stays normal during flash"
echo "  - Keyboard shortcuts for all controls"
echo ""
echo "🌐 After starting, you can access the viewer at:"
echo "  Local: http://localhost:8501"
echo "  Network: http://$(hostname -I | awk '{print $1}'):8501"
echo ""
echo "📋 Usage:"
echo "  - Hover over image to see magnified view"
echo "  - Use +/- keys or buttons to adjust zoom"
echo "  - Press F to flash ONLY prediction points & boxes"
echo "  - Press 0 to reset zoom to 400%"
echo ""
echo "💡 Script Usage:"
echo "  $0 [folder_name]"
echo "  Default folder: error_imgs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run error_image_viewer_web_enhanced.py --server.port=8501 --server.address=0.0.0.0 -- "$FOLDER_NAME"
