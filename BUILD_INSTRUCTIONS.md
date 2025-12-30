# Building the iFocus BCI Game Executable

This guide explains how to create a standalone executable for the iFocus BCI Game.

## Prerequisites

- Python 3.7 or higher
- All dependencies from requirements.txt installed
- The icon image saved as `icon.png` in the root directory

## Quick Start

### Step 1: Save the Icon

**Important:** Save your BCI game icon image as `icon.png` in the root directory (`e:\ifocus_outsource\icon.png`).

The icon should be:
- PNG format
- Square dimensions (recommended: 512x512 or 1024x1024)
- Transparent background (optional)

### Step 2: Run the Build Script

Simply run:

```bash
python build_executable.py
```

This script will automatically:
1. Convert your icon.png to icon.ico format
2. Build the executable using PyInstaller
3. Create a distributable folder

### Alternative: Manual Build

If you prefer to build manually:

```bash
# Convert icon (if you have icon.png)
python create_icon.py

# Build with PyInstaller
pyinstaller iFocus_BCI_Game.spec --clean --noconfirm
```

## Output

After building, you'll find:
- **Single Executable:** `dist/iFocus_BCI_Game.exe` (approx. 350-400 MB)
- This is a completely self-contained file - no other files needed!

## Distribution

To share the game:
1. Simply copy the `iFocus_BCI_Game.exe` file
2. No installation required - just double-click to run!
3. Works on any Windows machine - no Python required!

## Troubleshooting

### Icon not showing
- Make sure `icon.png` exists before building
- Run `python create_icon.py` to verify icon conversion
- Rebuild with `python build_executable.py`

### Missing dependencies
- Check that all packages in requirements.txt are installed
- Install missing packages: `pip install -r requirements.txt`

### Build errors
- Run with `--clean` flag: `pyinstaller iFocus_BCI_Game.spec --clean`
- Check Python version compatibility
- Ensure all source files are accessible

### Runtime errors
- Make sure the calibration data folder is included
- Keep all files in the dist folder together
- Check Windows Defender/Antivirus isn't blocking the exe

## File Structure

```
e:\ifocus_outsource\
├── icon.png                    # Your icon image (you need to add this)
├── icon.ico                    # Converted icon (auto-generated)
├── build_executable.py         # Build script (run this)
├── create_icon.py             # Icon converter helper
├── iFocus_BCI_Game.spec       # PyInstaller configuration
├── ifocus_2_player_game/      # Game source code
├── ifocus_sdk/                # SDK source code
├── data/                      # Calibration data
└── dist/                      # Build output (created after build)
    └── iFocus_BCI_Game/       # Distributable folder
        └── iFocus_BCI_Game.exe # Your game executable
```

## Customization

Edit `iFocus_BCI_Game.spec` to:
- Change the executable name
- Add/remove hidden imports
- Include additional data files
- Toggle console window visibility
