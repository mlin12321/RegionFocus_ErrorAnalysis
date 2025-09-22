#!/usr/bin/env python3
"""
Script to visualize evaluation results by overlaying ground truth bounding boxes
and predicted points on images from qwen25vl_RegionFocus_7b_filtered.json
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial


def load_evaluation_data(json_path):
    """Load the evaluation results from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['details']


def create_output_directory(output_dir):
    """Create the output directory if it doesn't exist"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def is_substantially_different(point1, point2, threshold=20):
    """
    Check if two points are substantially different (more than threshold pixels in either direction)
    
    Args:
        point1: List of [x, y] coordinates
        point2: List of [x, y] coordinates  
        threshold: Minimum pixel difference to consider substantial
    
    Returns:
        Boolean indicating if points are substantially different
    """
    if not point1 or not point2:
        return False
    
    x_diff = abs(point1[0] - point2[0])
    y_diff = abs(point1[1] - point2[1])
    
    return x_diff > threshold or y_diff > threshold


def draw_bbox_and_point(image, bbox, pred_point, correctness, judgment_point=None):
    """
    Draw ground truth bounding box, predicted point, and optionally judgment point on the image
    Only shows judgment point if it's substantially different from predicted point
    
    Args:
        image: PIL Image object
        bbox: List of [x1, y1, x2, y2] coordinates for ground truth bounding box
        pred_point: List of [x, y] coordinates for predicted point
        correctness: String indicating if prediction was correct or wrong
        judgment_point: Optional list of [x, y] coordinates for judgment point
    
    Returns:
        Tuple of (Modified PIL Image object, annotation_metadata dict)
    """
    draw = ImageDraw.Draw(image)
    
    # Draw ground truth bounding box in red
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
    
    # Draw predicted point as a circle
    pred_x, pred_y = pred_point
    point_radius = 8
    
    # Use different colors based on correctness
    point_color = 'green' if correctness == 'correct' else 'blue'
    
    # Draw circle for predicted point
    draw.ellipse([
        pred_x - point_radius, pred_y - point_radius,
        pred_x + point_radius, pred_y + point_radius
    ], outline=point_color, width=3, fill=point_color)
    
    # Only draw judgment point if it's substantially different from predicted point
    show_judgment = judgment_point and is_substantially_different(pred_point, judgment_point)
    if show_judgment:
        judge_x, judge_y = judgment_point
        judge_radius = 8
        
        # Draw orange circle for judgment point
        draw.ellipse([
            judge_x - judge_radius, judge_y - judge_radius,
            judge_x + judge_radius, judge_y + judge_radius
        ], outline='orange', width=3, fill='orange')
    
    # Add labels
    try:
        # Try to use a default font, fall back to default if not available
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Add "GT" label near the bounding box
    draw.text((x1, y1 - 20), "GT", fill='red', font=font)
    
    # Add "PRED" label near the predicted point
    draw.text((pred_x + point_radius + 5, pred_y - 10), "PRED", fill=point_color, font=font)
    
    # Add "JUDGE" label near the judgment point if it's shown
    if show_judgment:
        judge_x, judge_y = judgment_point
        draw.text((judge_x + judge_radius + 5, judge_y - 10), "JUDGE", fill='orange', font=font)
    
    # Create annotation metadata for web viewer flashing
    annotation_metadata = {
        'bbox': {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'color': 'red', 'width': 3, 'type': 'rectangle'
        },
        'pred_point': {
            'x': pred_x, 'y': pred_y, 'radius': point_radius,
            'color': point_color, 'width': 3, 'type': 'circle'
        },
        'labels': [
            {'x': x1, 'y': y1 - 20, 'text': 'GT', 'color': 'red'},
            {'x': pred_x + point_radius + 5, 'y': pred_y - 10, 'text': 'PRED', 'color': point_color}
        ]
    }
    
    # Add judgment point metadata if shown
    if show_judgment:
        judge_x, judge_y = judgment_point
        annotation_metadata['judge_point'] = {
            'x': judge_x, 'y': judge_y, 'radius': judge_radius,
            'color': 'orange', 'width': 3, 'type': 'circle'
        }
        annotation_metadata['labels'].append({
            'x': judge_x + judge_radius + 5, 'y': judge_y - 10, 'text': 'JUDGE', 'color': 'orange'
        })
    
    return image, annotation_metadata


def parse_judgment_point(judgment_file_path):
    """
    Parse judgment_response.txt file to extract point coordinates
    
    Args:
        judgment_file_path: Path to judgment_response.txt file
    
    Returns:
        List of [x, y] coordinates if found, None otherwise
    """
    if not os.path.exists(judgment_file_path):
        return None
    
    try:
        with open(judgment_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('Point: [') and line.endswith(']'):
                    # Extract coordinates from "Point: [x, y]" format
                    coords_str = line[8:-1]  # Remove "Point: [" and "]"
                    coords = coords_str.split(', ')
                    if len(coords) == 2:
                        x = float(coords[0])
                        y = float(coords[1])
                        return [x, y]
        return None
    except Exception as e:
        print(f"Error parsing judgment file {judgment_file_path}: {str(e)}")
        return None


def get_debug_folder_name(img_path):
    """
    Generate debug folder name from image path
    
    Args:
        img_path: Image path like "./images/android_studio_mac/screenshot_2024-11-07_15-38-30.png"
    
    Returns:
        Debug folder name like "android_studio_mac_screenshot_2024-11-07_15-38-30"
    """
    # Extract filename without extension
    filename = os.path.basename(img_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Extract application/platform from path
    path_parts = img_path.replace('./', '').split('/')
    if len(path_parts) >= 2:
        app_platform = path_parts[1]  # e.g., "android_studio_mac"
        return f"{app_platform}_{name_without_ext}"
    else:
        return name_without_ext


def copy_debug_files(debug_base_path, debug_folder_name, output_folder):
    """
    Copy judgment_response.txt and aggregation_response.txt from debug folder if they exist
    
    Args:
        debug_base_path: Base path to debug folder
        debug_folder_name: Name of the specific debug folder
        output_folder: Destination folder to copy files to
    
    Returns:
        tuple: (copied_files_count, message_string)
    """
    debug_folder_path = os.path.join(debug_base_path, debug_folder_name)
    copied_files = []
    
    if not os.path.exists(debug_folder_path):
        return 0, f"Debug folder not found: {debug_folder_path}"
    
    # Copy judgment_response.txt if it exists
    judgment_file = os.path.join(debug_folder_path, "judgment_response.txt")
    if os.path.exists(judgment_file):
        shutil.copy2(judgment_file, output_folder)
        copied_files.append("judgment_response.txt")
    
    # Copy aggregation_response.txt if it exists
    aggregation_file = os.path.join(debug_folder_path, "aggregation_response.txt")
    if os.path.exists(aggregation_file):
        shutil.copy2(aggregation_file, output_folder)
        copied_files.append("aggregation_response.txt")
    
    if copied_files:
        return len(copied_files), f"Copied: {', '.join(copied_files)}"
    else:
        return 0, "No debug files found to copy"


def process_single_datapoint(datapoint, base_path, output_dir, debug_base_path="debug"):
    """
    Process a single evaluation datapoint
    
    Args:
        datapoint: Dictionary containing evaluation data
        base_path: Base path for resolving relative image paths
        output_dir: Output directory for modified images
        debug_base_path: Base path to debug folder
    
    Returns:
        tuple: (success_bool, message_string)
    """
    img_path = datapoint['img_path']
    bbox = datapoint['bbox']
    pred_point = datapoint['pred']
    correctness = datapoint['correctness']
    
    # Resolve the full image path
    if img_path.startswith('./'):
        full_img_path = os.path.join(base_path, img_path[2:])
    else:
        full_img_path = os.path.join(base_path, img_path)
    
    # Check if image exists
    if not os.path.exists(full_img_path):
        return False, f"Warning: Image not found: {full_img_path}"
    
    try:
        # Create unique folder name for this datapoint
        relative_path = img_path.replace('./', '').replace('/', '_')
        folder_name = f"{correctness}_{relative_path}".replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        datapoint_folder = os.path.join(output_dir, folder_name)
        
        # Create the datapoint folder
        os.makedirs(datapoint_folder, exist_ok=True)
        
        # Load the image
        image = Image.open(full_img_path)
        
        # Convert to RGB if necessary (in case of RGBA or other formats)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Copy debug files first so we can parse judgment point
        debug_folder_name = get_debug_folder_name(img_path)
        debug_files_count, debug_message = copy_debug_files(
            os.path.join(base_path, debug_base_path), 
            debug_folder_name, 
            datapoint_folder
        )
        
        # Parse judgment point from copied judgment_response.txt if it exists
        judgment_file_path = os.path.join(datapoint_folder, "judgment_response.txt")
        judgment_point = parse_judgment_point(judgment_file_path)
        
        # Check if judgment point will be shown (substantially different from prediction)
        show_judgment = judgment_point and is_substantially_different(pred_point, judgment_point)
        
        # Draw bbox, prediction point, and judgment point (if substantially different)
        modified_image, annotation_metadata = draw_bbox_and_point(image, bbox, pred_point, correctness, judgment_point)
        
        # Save the modified image in the datapoint folder
        output_filename = "marked_screenshot.png"
        output_path = os.path.join(datapoint_folder, output_filename)
        modified_image.save(output_path)
        
        # Save annotation metadata for web viewer flashing
        metadata_path = os.path.join(datapoint_folder, "annotation_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(annotation_metadata, f, indent=2)
        
        if judgment_point:
            if show_judgment:
                judgment_info = " + judgment point (shown)"
            else:
                judgment_info = " + judgment point (filtered - too similar)"
        else:
            judgment_info = ""
        
        success_message = f"Processed: {img_path} -> {folder_name}/ (screenshot + {debug_files_count} debug files{judgment_info})"
        return True, success_message
        
    except Exception as e:
        return False, f"Error processing {img_path}: {str(e)}"


def process_datapoint_wrapper(args):
    """Wrapper function for multiprocessing"""
    datapoint, base_path, output_dir, debug_base_path = args
    return process_single_datapoint(datapoint, base_path, output_dir, debug_base_path)


def main():
    parser = argparse.ArgumentParser(description='Visualize evaluation results')
    parser.add_argument('--json_path', 
                       default='results/qwen25vl_RegionFocus_7b_filtered.json',
                       help='Path to the evaluation JSON file')
    parser.add_argument('--output_dir', 
                       default='error_imgs',
                       help='Output directory for modified images')
    parser.add_argument('--base_path', 
                       default='.',
                       help='Base path for resolving relative image paths')
    parser.add_argument('--limit', 
                       type=int, 
                       default=None,
                       help='Limit number of images to process (for testing)')
    parser.add_argument('--num_processes', 
                       type=int, 
                       default=None,
                       help='Number of parallel processes (default: number of CPU cores)')
    parser.add_argument('--debug_path', 
                       default='debug',
                       help='Path to debug folder containing judgment and aggregation responses')
    
    args = parser.parse_args()
    
    # Load evaluation data
    print(f"Loading evaluation data from {args.json_path}...")
    datapoints = load_evaluation_data(args.json_path)
    print(f"Found {len(datapoints)} datapoints")
    
    # Create output directory
    create_output_directory(args.output_dir)
    print(f"Created output directory: {args.output_dir}")
    
    # Determine number of processes
    num_processes = args.num_processes if args.num_processes else cpu_count()
    print(f"Using {num_processes} parallel processes")
    
    # Process each datapoint using multiprocessing
    limit = args.limit if args.limit else len(datapoints)
    datapoints_to_process = datapoints[:limit]
    
    # Prepare arguments for multiprocessing
    process_args = [(datapoint, args.base_path, args.output_dir, args.debug_path) for datapoint in datapoints_to_process]
    
    processed_count = 0
    error_count = 0
    
    print(f"Processing {len(datapoints_to_process)} images...")
    
    # Use multiprocessing Pool to process images in parallel
    with Pool(processes=num_processes) as pool:
        # Process in chunks to show progress
        chunk_size = max(1, len(process_args) // 20)  # 20 progress updates
        
        for i in range(0, len(process_args), chunk_size):
            chunk = process_args[i:i + chunk_size]
            results = pool.map(process_datapoint_wrapper, chunk)
            
            # Count results and print messages
            for success, message in results:
                if success:
                    processed_count += 1
                    print(message)
                else:
                    error_count += 1
                    print(message)
            
            # Progress indicator
            progress = min(i + chunk_size, len(process_args))
            print(f"Progress: {progress}/{len(process_args)} datapoints processed")
    
    print(f"\nCompleted!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Errors encountered: {error_count} images")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
