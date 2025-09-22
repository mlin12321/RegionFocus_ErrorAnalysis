# Error Image Viewer

A comprehensive GUI application for viewing error images and their associated text files from the RegionFocus project.

## Features

### Image Display
- **Main Image View**: Displays the `marked_screenshot.png` from each error folder
- **Scrollable Canvas**: Large images can be scrolled both horizontally and vertically
- **High-Quality Display**: Images are displayed at their original resolution

### Hover Zoom (Enhanced Web Version)
- **Dynamic Magnification**: Hover over any part of the image to see a circular magnifying glass
- **Smart Bounds**: Magnifier automatically repositions to stay visible even at image edges
- **Adjustable Zoom Level**: Use +/- buttons or keyboard to adjust zoom from 50% to 500%
- **Default Zoom**: Starts at 400% zoom level for detailed inspection
- **Selective Flash**: Button to make ONLY annotation elements flash (prediction points, boxes, etc.)
- **Background Preserved**: Image background remains normal during flashing
- **Keyboard Shortcuts**: 
  - `+` (zoom in), `-` (zoom out), `0` (reset to 400%)
  - `F` (toggle flashing highlights)
- **Enhanced Colors**: Brighter, more vibrant colors for better visibility
- **Visual Design**: Bright green circular magnifying glass with enhanced shadow effects

### Text File Display
- **Judgment Response**: Shows content from `judgment_response.txt` (always present)
- **Aggregation Response**: Shows content from `aggregation_response.txt` (when available)
- **Scrollable Text Areas**: Both text areas are independently scrollable
- **Monospace Font**: Uses Consolas font for better readability of structured text

### Navigation
- **Previous/Next Buttons**: Navigate sequentially through all entries
- **Entry Counter**: Shows current position (e.g., "5 / 1069")
- **Jump to Entry**: Direct navigation by entering an entry number
- **Keyboard Support**: Press Enter in the jump field to navigate
- **Ordered Display**: Entries are sorted by folder name (timestamp-based)

### User Interface
- **Responsive Layout**: Adjusts to different window sizes
- **Professional Look**: Uses ttk widgets for modern appearance
- **Split Layout**: Image on left, text content on right
- **Status Display**: Shows current folder name and zoom level

## Usage

### Starting the Application

#### Web-based Version (Recommended for remote servers)
```bash
# Method 1: Enhanced web version with hover magnification
./run_error_viewer_web.sh

# Method 2: Alternative enhanced launcher  
./run_error_viewer_web_enhanced.sh

# Method 3: Run directly with Python
conda activate regionfocus  # If using conda
streamlit run error_image_viewer_web_enhanced.py --server.port=8501 --server.address=0.0.0.0
```

#### Desktop GUI Version (For local use with display)
```bash
# Method 1: Use the launcher script (handles display issues automatically)
./run_error_viewer.sh

# Method 2: Run directly with Python (requires display)
python3 error_image_viewer.py
```

### Navigation

#### Web Version
- **Previous/Next**: Use the arrow buttons to move between entries
- **Jump to Entry**: Type a number (1-1069) and press Enter or click "Go"
- **Hover Magnification**: Move mouse over image to see magnified circular view (400% default)
- **Smart Positioning**: Magnifier automatically stays visible at image edges
- **Selective Flash**: Click 🔆 button or press F to flash ONLY prediction points & annotation boxes
- **Zoom Control**: Use +/- buttons or keyboard shortcuts to adjust magnification
- **Access**: Open http://localhost:8501 in your web browser

#### Desktop Version  
- **Previous/Next**: Use the arrow buttons to move between entries
- **Jump to Entry**: Type a number (1-1069) and press Enter or click "Go"
- **Zoom Control**: Hover over image and scroll to adjust zoom level

### File Structure Expected
```
error_imgs/
├── wrong_images_android_studio_mac_screenshot_2024-11-05_16-01-19/
│   ├── marked_screenshot.png          (required)
│   ├── judgment_response.txt          (required)
│   └── aggregation_response.txt       (optional)
├── wrong_images_android_studio_mac_screenshot_2024-11-05_16-05-07/
│   └── ...
└── ...
```

## Requirements

- **Python 3.x** with tkinter (usually included)
- **Pillow (PIL)** for image handling
- **Linux/macOS/Windows** compatible

## Installation

The application uses only standard Python libraries plus Pillow:

```bash
pip install Pillow
```

## Technical Details

- **Framework**: tkinter for cross-platform GUI
- **Image Handling**: PIL/Pillow for image processing and display
- **Zoom Implementation**: Dynamic image cropping and scaling
- **Memory Efficient**: Images are loaded on-demand
- **Error Handling**: Graceful handling of missing files and corrupted images

## Troubleshooting

### Common Issues
1. **"error_imgs directory not found"**: Ensure you're running from the RegionFocus directory
2. **Images not loading**: Check that `marked_screenshot.png` files exist in the folders
3. **Zoom not working**: Ensure mouse is over the image area when scrolling

### Performance Notes
- Large images may take a moment to load
- Zoom functionality works best with images under 10MB
- The application loads ~1069 folder entries efficiently
