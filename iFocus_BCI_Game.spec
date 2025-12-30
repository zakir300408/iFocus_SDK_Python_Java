# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for iFocus 2-Player BCI Game
This file configures how to build the executable with all dependencies
"""

import sys
from pathlib import Path

# Get the root directory
ROOT = Path('e:/ifocus_outsource').resolve()

block_cipher = None

# Collect all data files
datas = [
    (str(ROOT / 'ifocus_2_player_game' / 'assets'), 'ifocus_2_player_game/assets'),
    (str(ROOT / 'data' / 'calibration'), 'data/calibration'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'qasync',
    'bleak',
    'numpy',
    'scipy',
    'pyedflib',
    'sklearn',
    'joblib',
    'asyncio',
    'ifocus_sdk.APIs.calibrationControl',
    'ifocus_sdk.APIs.FocusComs',
    'ifocus_sdk.APIs.FocusInference',
    'ifocus_sdk.APIs.iFocusParser',
    'ifocus_sdk.APIs.trainFocusModel',
    'ifocus_sdk.APIs._preprocess_extract_features',
    'ifocus_2_player_game.ui.ifocus_ui',
    'ifocus_2_player_game.ui.play_window_ui',
    'ifocus_2_player_game.ui.play_window_config',
    'ifocus_2_player_game.ui.ifocus_hooks',
]

a = Analysis(
    [str(ROOT / 'ifocus_2_player_game' / 'main.py')],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'IPython', 'jupyter', 'PyQt5', 'PyQt6', 'tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='iFocus_BCI_Game',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to False to hide console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / 'icon.ico') if (ROOT / 'icon.ico').exists() else None,
)
