#!/usr/bin/env python3
"""
Script to analyze debug folders and categorize them based on presence of judgment_response.txt or aggregation_response.txt files.
Cross-references entries from the results JSON file with corresponding debug folders.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import re

def extract_folder_name_from_img_path(img_path):
    """
    Extract the expected debug folder name from the image path.
    ./images/ppt_windows/screenshot_2024-10-27_21-39-08.png -> ppt_windows_screenshot_2024-10-27_21-39-08
    """
    # Remove ./images/ prefix and .png suffix
    path_part = img_path.replace('./images/', '').replace('.png', '')
    
    # Split by '/' to get application_platform and screenshot_timestamp
    parts = path_part.split('/')
    if len(parts) == 2:
        app_platform = parts[0]
        screenshot_part = parts[1]  # screenshot_2024-10-27_21-39-08
        return f"{app_platform}_{screenshot_part}"
    
    return None

def analyze_debug_folders():
    # Paths
    results_file = '/local/scratch/lin.3976/RegionFocus/results/qwen25vl_RegionFocus_7b_filtered.json'
    debug_dirs = [
        '/local/scratch/lin.3976/RegionFocus/debug',
        '/local/scratch/lin.3976/RegionFocus/debug_copy_qwen25vl_7b'
    ]
    output_file = '/local/scratch/lin.3976/RegionFocus/debug_folder_analysis.txt'
    
    # Load JSON results
    print("Loading results JSON file...")
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Extract expected folder names from JSON entries
    expected_folders = set()
    img_path_to_folder = {}
    
    for entry in data['details']:
        img_path = entry['img_path']
        folder_name = extract_folder_name_from_img_path(img_path)
        if folder_name:
            expected_folders.add(folder_name)
            img_path_to_folder[img_path] = folder_name
    
    print(f"Found {len(expected_folders)} expected folders from JSON entries")
    
    # Scan debug directories
    all_debug_folders = set()
    folder_to_debug_dir = {}
    
    for debug_dir in debug_dirs:
        if os.path.exists(debug_dir):
            print(f"Scanning {debug_dir}...")
            for item in os.listdir(debug_dir):
                item_path = os.path.join(debug_dir, item)
                if os.path.isdir(item_path):
                    all_debug_folders.add(item)
                    folder_to_debug_dir[item] = debug_dir
    
    print(f"Found {len(all_debug_folders)} total debug folders")
    
    # Categorize folders
    categories = {
        'has_judgment': [],
        'has_aggregation': [],
        'has_both': [],
        'has_neither': [],
        'not_in_json': []
    }
    
    # Check each debug folder
    for folder in all_debug_folders:
        debug_dir = folder_to_debug_dir[folder]
        folder_path = os.path.join(debug_dir, folder)
        
        judgment_file = os.path.join(folder_path, 'judgment_response.txt')
        aggregation_file = os.path.join(folder_path, 'aggregation_response.txt')
        
        has_judgment = os.path.exists(judgment_file)
        has_aggregation = os.path.exists(aggregation_file)
        
        # Determine category
        if folder not in expected_folders:
            categories['not_in_json'].append(folder)
        elif has_judgment and has_aggregation:
            categories['has_both'].append(folder)
        elif has_judgment:
            categories['has_judgment'].append(folder)
        elif has_aggregation:
            categories['has_aggregation'].append(folder)
        else:
            categories['has_neither'].append(folder)
    
    # Check for missing folders (in JSON but not in debug)
    missing_folders = expected_folders - all_debug_folders
    
    # Sort all categories
    for category in categories:
        categories[category].sort()
    
    # Write results
    print(f"Writing results to {output_file}...")
    with open(output_file, 'w') as f:
        f.write("DEBUG FOLDER ANALYSIS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total JSON entries: {len(data['details'])}\n")
        f.write(f"Expected folders from JSON: {len(expected_folders)}\n")
        f.write(f"Total debug folders found: {len(all_debug_folders)}\n")
        f.write(f"Missing folders (in JSON but not in debug): {len(missing_folders)}\n\n")
        
        # Write categories
        f.write("FOLDERS WITH judgment_response.txt ONLY:\n")
        f.write("-" * 40 + "\n")
        for folder in categories['has_judgment']:
            f.write(f"{folder}\n")
        f.write(f"\nTotal: {len(categories['has_judgment'])}\n\n")
        
        f.write("FOLDERS WITH aggregation_response.txt ONLY:\n")
        f.write("-" * 40 + "\n")
        for folder in categories['has_aggregation']:
            f.write(f"{folder}\n")
        f.write(f"\nTotal: {len(categories['has_aggregation'])}\n\n")
        
        f.write("FOLDERS WITH BOTH FILES:\n")
        f.write("-" * 40 + "\n")
        for folder in categories['has_both']:
            f.write(f"{folder}\n")
        f.write(f"\nTotal: {len(categories['has_both'])}\n\n")
        
        f.write("FOLDERS WITH NEITHER FILE:\n")
        f.write("-" * 40 + "\n")
        for folder in categories['has_neither']:
            f.write(f"{folder}\n")
        f.write(f"\nTotal: {len(categories['has_neither'])}\n\n")
        
        f.write("FOLDERS NOT IN JSON (extra debug folders):\n")
        f.write("-" * 40 + "\n")
        for folder in categories['not_in_json']:
            f.write(f"{folder}\n")
        f.write(f"\nTotal: {len(categories['not_in_json'])}\n\n")
        
        if missing_folders:
            f.write("MISSING FOLDERS (in JSON but not in debug directories):\n")
            f.write("-" * 40 + "\n")
            for folder in sorted(missing_folders):
                f.write(f"{folder}\n")
            f.write(f"\nTotal: {len(missing_folders)}\n\n")
        
        # Summary
        f.write("SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Folders with judgment_response.txt only: {len(categories['has_judgment'])}\n")
        f.write(f"Folders with aggregation_response.txt only: {len(categories['has_aggregation'])}\n")
        f.write(f"Folders with both files: {len(categories['has_both'])}\n")
        f.write(f"Folders with neither file: {len(categories['has_neither'])}\n")
        f.write(f"Extra debug folders (not in JSON): {len(categories['not_in_json'])}\n")
        f.write(f"Missing folders: {len(missing_folders)}\n")
    
    print("Analysis complete!")
    print("\nSummary:")
    print(f"- Folders with judgment_response.txt only: {len(categories['has_judgment'])}")
    print(f"- Folders with aggregation_response.txt only: {len(categories['has_aggregation'])}")
    print(f"- Folders with both files: {len(categories['has_both'])}")
    print(f"- Folders with neither file: {len(categories['has_neither'])}")
    print(f"- Extra debug folders (not in JSON): {len(categories['not_in_json'])}")
    print(f"- Missing folders: {len(missing_folders)}")

if __name__ == "__main__":
    analyze_debug_folders()
