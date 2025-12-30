"""
Script to create the icon.ico file from a PNG image.
Place your icon.png file in the root directory first, then run this script.
"""
from PIL import Image
import os

def create_ico_from_png():
    """Convert PNG to ICO format for Windows executable"""
    png_path = "icon.png"
    ico_path = "icon.ico"
    
    if not os.path.exists(png_path):
        print(f"Error: {png_path} not found!")
        print("Please save the icon image as 'icon.png' in the root directory first.")
        return False
    
    try:
        # Open the PNG image
        img = Image.open(png_path)
        
        # Convert to RGBA if needed
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Create multiple sizes for the icon
        icon_sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
        
        # Save as ICO
        img.save(ico_path, format='ICO', sizes=icon_sizes)
        print(f"✓ Successfully created {ico_path}")
        return True
    except Exception as e:
        print(f"Error creating icon: {e}")
        return False

if __name__ == "__main__":
    create_ico_from_png()
