"""
Build script for creating the iFocus BCI Game executable
This script will:
1. Check/create the icon file
2. Build the executable using PyInstaller
3. Provide the output location
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main build process"""
    print("=" * 60)
    print("iFocus BCI Game - Executable Builder")
    print("=" * 60)
    
    root = Path(__file__).parent
    icon_png = root / "icon.png"
    icon_ico = root / "icon.ico"
    spec_file = root / "iFocus_BCI_Game.spec"
    
    # Step 1: Check for icon files
    print("\n[1/3] Checking icon files...")
    if not icon_png.exists():
        print("⚠ Warning: icon.png not found!")
        print("  Please save your icon image as 'icon.png' in this directory.")
        print("  The executable will be built without a custom icon.")
    elif not icon_ico.exists():
        print("  Converting icon.png to icon.ico...")
        try:
            from PIL import Image
            img = Image.open(icon_png)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            icon_sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
            img.save(icon_ico, format='ICO', sizes=icon_sizes)
            print("  ✓ Icon converted successfully!")
        except Exception as e:
            print(f"  ⚠ Error converting icon: {e}")
            print("  The executable will be built without a custom icon.")
    else:
        print("  ✓ Icon file found!")
    
    # Step 2: Build the executable
    print("\n[2/3] Building executable with PyInstaller...")
    print("  This may take a few minutes...\n")
    
    try:
        # Run PyInstaller with the spec file
        cmd = [sys.executable, "-m", "PyInstaller", str(spec_file), "--clean", "--noconfirm"]
        result = subprocess.run(cmd, cwd=str(root), check=True)
        
        if result.returncode == 0:
            print("\n  ✓ Build completed successfully!")
        else:
            print("\n  ✗ Build failed!")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ Build error: {e}")
        return False
    except Exception as e:
        print(f"\n  ✗ Unexpected error: {e}")
        return False
    
    # Step 3: Show results
    print("\n[3/3] Build Results:")
    print("=" * 60)
    
    # Check for single-file executable
    exe_file_single = root / "dist" / "iFocus_BCI_Game.exe"
    dist_folder = root / "dist" / "iFocus_BCI_Game"
    exe_file_folder = dist_folder / "iFocus_BCI_Game.exe"
    
    if exe_file_single.exists():
        print(f"✓ Single-file executable created successfully!")
        print(f"\n📁 Location: {exe_file_single.parent}")
        print(f"🎮 Executable: {exe_file_single}")
        print(f"📦 Size: {exe_file_single.stat().st_size / (1024*1024):.1f} MB")
        print(f"\n💡 You can now:")
        print(f"   1. Copy iFocus_BCI_Game.exe to any location")
        print(f"   2. Double-click to run - it's completely self-contained!")
        print(f"   3. Create a desktop shortcut")
        print(f"   4. Share the single .exe file - no dependencies needed!")
        print("\n✨ This is a standalone executable - no installation required!")
        return True
    elif exe_file_folder.exists():
        print(f"✓ Executable created successfully!")
        print(f"\n📁 Location: {dist_folder}")
        print(f"🎮 Executable: {exe_file_folder}")
        print(f"\n💡 You can now:")
        print(f"   1. Copy the entire '{dist_folder.name}' folder to any location")
        print(f"   2. Run iFocus_BCI_Game.exe to start the game")
        print(f"   3. Create a desktop shortcut to the .exe file")
        print("\n⚠ Note: Keep all files in the folder together - don't move just the .exe!")
        return True
    else:
        print("✗ Executable not found. Please check the build output for errors.")
        return False

if __name__ == "__main__":
    success = main()
    print("\n" + "=" * 60)
    sys.exit(0 if success else 1)
