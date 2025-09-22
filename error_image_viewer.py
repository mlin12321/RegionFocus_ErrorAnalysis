#!/usr/bin/env python3
"""
Error Image Viewer - A GUI application to browse error images and their associated text files.

Features:
- Display marked screenshots from error_imgs folders
- Show judgment_response.txt and aggregation_response.txt content
- Navigate between entries with prev/next buttons
- Jump to specific entry by number
- Hover zoom functionality with adjustable zoom level (scroll wheel)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import os
import glob
from pathlib import Path


class ErrorImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Error Image Viewer")
        self.root.geometry("1400x900")
        
        # Data
        self.error_folders = []
        self.current_index = 0
        self.zoom_level = 2.0  # Default 200% zoom
        self.original_image = None
        self.display_image = None
        
        # Zoom window
        self.zoom_window = None
        self.zoom_label = None
        
        # Initialize UI
        self.setup_ui()
        self.load_error_folders()
        
        if self.error_folders:
            self.display_current_entry()
    
    def setup_ui(self):
        """Setup the main UI components"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top navigation frame
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Navigation controls
        ttk.Button(nav_frame, text="◀ Previous", command=self.previous_entry).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Next ▶", command=self.next_entry).pack(side=tk.LEFT, padx=(0, 20))
        
        # Entry counter and jump controls
        ttk.Label(nav_frame, text="Entry:").pack(side=tk.LEFT)
        self.current_label = ttk.Label(nav_frame, text="0 / 0")
        self.current_label.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(nav_frame, text="Jump to:").pack(side=tk.LEFT)
        self.jump_entry = ttk.Entry(nav_frame, width=8)
        self.jump_entry.pack(side=tk.LEFT, padx=(5, 5))
        self.jump_entry.bind('<Return>', self.jump_to_entry)
        ttk.Button(nav_frame, text="Go", command=self.jump_to_entry).pack(side=tk.LEFT)
        
        # Zoom level display
        ttk.Label(nav_frame, text="Zoom:").pack(side=tk.RIGHT, padx=(20, 5))
        self.zoom_label_widget = ttk.Label(nav_frame, text="200%")
        self.zoom_label_widget.pack(side=tk.RIGHT)
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Image
        image_frame = ttk.LabelFrame(content_frame, text="Screenshot", padding=10)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Image canvas with scrollbars
        self.image_canvas = tk.Canvas(image_frame, bg='white')
        v_scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        h_scrollbar = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        self.image_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mouse events for zoom functionality
        self.image_canvas.bind('<Motion>', self.on_mouse_move)
        self.image_canvas.bind('<Leave>', self.hide_zoom)
        self.image_canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        self.image_canvas.bind('<Button-4>', self.on_mouse_wheel)  # Linux scroll up
        self.image_canvas.bind('<Button-5>', self.on_mouse_wheel)  # Linux scroll down
        
        # Right side - Text content
        text_frame = ttk.Frame(content_frame)
        text_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        text_frame.configure(width=400)
        text_frame.pack_propagate(False)
        
        # Folder name
        self.folder_label = ttk.Label(text_frame, text="", font=('Arial', 10, 'bold'))
        self.folder_label.pack(fill=tk.X, pady=(0, 10))
        
        # Judgment response
        judgment_frame = ttk.LabelFrame(text_frame, text="Judgment Response", padding=5)
        judgment_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.judgment_text = tk.Text(judgment_frame, wrap=tk.WORD, height=15, font=('Consolas', 9))
        judgment_scroll = ttk.Scrollbar(judgment_frame, orient=tk.VERTICAL, command=self.judgment_text.yview)
        self.judgment_text.configure(yscrollcommand=judgment_scroll.set)
        judgment_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.judgment_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Aggregation response
        aggregation_frame = ttk.LabelFrame(text_frame, text="Aggregation Response", padding=5)
        aggregation_frame.pack(fill=tk.BOTH, expand=True)
        
        self.aggregation_text = tk.Text(aggregation_frame, wrap=tk.WORD, height=15, font=('Consolas', 9))
        aggregation_scroll = ttk.Scrollbar(aggregation_frame, orient=tk.VERTICAL, command=self.aggregation_text.yview)
        self.aggregation_text.configure(yscrollcommand=aggregation_scroll.set)
        aggregation_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.aggregation_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def load_error_folders(self):
        """Load all error folders from the error_imgs directory"""
        error_imgs_path = Path(__file__).parent / "error_imgs"
        if not error_imgs_path.exists():
            messagebox.showerror("Error", f"error_imgs directory not found at {error_imgs_path}")
            return
        
        # Get all subdirectories and sort them by name (which includes timestamp)
        self.error_folders = sorted([
            d for d in error_imgs_path.iterdir() 
            if d.is_dir() and (d / "marked_screenshot.png").exists()
        ])
        
        print(f"Found {len(self.error_folders)} error folders")
    
    def display_current_entry(self):
        """Display the current entry's image and text files"""
        if not self.error_folders or self.current_index >= len(self.error_folders):
            return
        
        current_folder = self.error_folders[self.current_index]
        
        # Update navigation info
        self.current_label.config(text=f"{self.current_index + 1} / {len(self.error_folders)}")
        self.folder_label.config(text=current_folder.name)
        
        # Load and display image
        self.load_image(current_folder / "marked_screenshot.png")
        
        # Load text files
        self.load_judgment_response(current_folder / "judgment_response.txt")
        self.load_aggregation_response(current_folder / "aggregation_response.txt")
    
    def load_image(self, image_path):
        """Load and display the screenshot image"""
        try:
            self.original_image = Image.open(image_path)
            self.display_image = ImageTk.PhotoImage(self.original_image)
            
            # Clear canvas and add image
            self.image_canvas.delete("all")
            self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
            
            # Update scroll region
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            self.image_canvas.delete("all")
            self.image_canvas.create_text(100, 100, text=f"Error loading image:\n{e}", fill="red")
    
    def load_judgment_response(self, text_path):
        """Load judgment response text"""
        self.judgment_text.delete(1.0, tk.END)
        try:
            if text_path.exists():
                with open(text_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.judgment_text.insert(1.0, content)
            else:
                self.judgment_text.insert(1.0, "No judgment_response.txt found")
        except Exception as e:
            self.judgment_text.insert(1.0, f"Error reading judgment_response.txt:\n{e}")
    
    def load_aggregation_response(self, text_path):
        """Load aggregation response text"""
        self.aggregation_text.delete(1.0, tk.END)
        try:
            if text_path.exists():
                with open(text_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.aggregation_text.insert(1.0, content)
            else:
                self.aggregation_text.insert(1.0, "No aggregation_response.txt found")
        except Exception as e:
            self.aggregation_text.insert(1.0, f"Error reading aggregation_response.txt:\n{e}")
    
    def previous_entry(self):
        """Navigate to previous entry"""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_entry()
    
    def next_entry(self):
        """Navigate to next entry"""
        if self.current_index < len(self.error_folders) - 1:
            self.current_index += 1
            self.display_current_entry()
    
    def jump_to_entry(self, event=None):
        """Jump to specific entry number"""
        try:
            target = int(self.jump_entry.get()) - 1  # Convert to 0-based index
            if 0 <= target < len(self.error_folders):
                self.current_index = target
                self.display_current_entry()
            else:
                messagebox.showerror("Error", f"Entry number must be between 1 and {len(self.error_folders)}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
    
    def on_mouse_move(self, event):
        """Handle mouse movement over image for zoom functionality"""
        if not self.original_image:
            return
        
        # Get mouse position relative to canvas
        canvas_x = self.image_canvas.canvasx(event.x)
        canvas_y = self.image_canvas.canvasy(event.y)
        
        # Show zoom window
        self.show_zoom(event.x_root, event.y_root, canvas_x, canvas_y)
    
    def show_zoom(self, screen_x, screen_y, canvas_x, canvas_y):
        """Show zoom window at mouse position"""
        if not self.original_image:
            return
        
        # Create zoom window if it doesn't exist
        if not self.zoom_window:
            self.zoom_window = tk.Toplevel(self.root)
            self.zoom_window.wm_overrideredirect(True)
            self.zoom_window.configure(bg='black', bd=2, relief='solid')
            self.zoom_label = ttk.Label(self.zoom_window)
            self.zoom_label.pack()
        
        # Calculate zoom region
        zoom_size = 100  # Size of the zoom region in original image pixels
        half_zoom = zoom_size // 2
        
        # Get image coordinates
        img_x = int(canvas_x)
        img_y = int(canvas_y)
        
        # Define crop region
        left = max(0, img_x - half_zoom)
        top = max(0, img_y - half_zoom)
        right = min(self.original_image.width, img_x + half_zoom)
        bottom = min(self.original_image.height, img_y + half_zoom)
        
        # Crop and zoom the region
        try:
            cropped = self.original_image.crop((left, top, right, bottom))
            zoomed_width = int(cropped.width * self.zoom_level)
            zoomed_height = int(cropped.height * self.zoom_level)
            zoomed = cropped.resize((zoomed_width, zoomed_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            zoom_photo = ImageTk.PhotoImage(zoomed)
            self.zoom_label.configure(image=zoom_photo)
            self.zoom_label.image = zoom_photo  # Keep a reference
            
            # Position zoom window
            offset = 20
            self.zoom_window.geometry(f"+{screen_x + offset}+{screen_y + offset}")
            self.zoom_window.deiconify()
            
        except Exception as e:
            print(f"Error creating zoom: {e}")
    
    def hide_zoom(self, event=None):
        """Hide zoom window"""
        if self.zoom_window:
            self.zoom_window.withdraw()
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zoom level adjustment"""
        if event.num == 4 or event.delta > 0:  # Scroll up
            self.zoom_level = min(5.0, self.zoom_level + 0.2)
        elif event.num == 5 or event.delta < 0:  # Scroll down
            self.zoom_level = max(0.5, self.zoom_level - 0.2)
        
        # Update zoom level display
        self.zoom_label_widget.config(text=f"{int(self.zoom_level * 100)}%")


def main():
    root = tk.Tk()
    app = ErrorImageViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
