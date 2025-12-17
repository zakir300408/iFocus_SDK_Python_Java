from __future__ import annotations

import asyncio
import sys
from pathlib import Path
import logging

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import PySide6 before qasync to ensure correct Qt binding is detected
from PySide6.QtWidgets import QApplication
try:
    from qasync import QEventLoop
except ImportError:
    print("Please install qasync: pip install qasync")
    sys.exit(1)

# SDK Imports
from ifocus_sdk.APIs.calibrationControl import (
    calibrationControl, 
    _scan_ifocus_devices, 
    _multi_parsers,
    setWearingCheckEnabled
)
# UI Imports
from ifocus_2_player_game.ui.ifocus_ui import IFocusWindow, DeviceInfo
import ifocus_2_player_game.ui.play_window_ui as play_ui
import ifocus_2_player_game.ui.play_window_config as play_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

class GameController:
    def __init__(self, window: IFocusWindow):
        self.window = window
        self.window.searchRequested.connect(self.on_search_requested)
        self.window.deviceToggled.connect(self.on_device_toggled)
        self.window.playRequested.connect(self.on_play_requested)
        
        self._status_task = None
        self._running = True
        
        # Keep track of connected devices
        self.connected_devices = set()

    def start(self):
        self._status_task = asyncio.get_event_loop().create_task(self._update_status_loop())

    async def stop(self):
        self._running = False
        if self._status_task:
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass

    def on_search_requested(self):
        asyncio.create_task(self._scan())

    async def _scan(self):
        try:
            logger.info("Scanning for devices...")
            # _scan_ifocus_devices returns list of (addr, name, rssi)
            devices_raw = await _scan_ifocus_devices(timeout_s=3.0)
            
            devices = []
            for addr, name, rssi in devices_raw:
                devices.append(DeviceInfo(name=name, mac=addr, rssi=rssi))
            
            # Limit to 4
            devices = devices[:4]

            if not devices:
                self.window.set_status("No devices found.")
            else:
                self.window.set_status(f"Found {len(devices)} device(s). Tap to connect.")
                self.window.set_devices(devices)
        except Exception as exc:
            logger.error(f"Scan failed: {exc}")
            self.window.set_status(f"Scan failed: {exc}")
        finally:
            self.window.end_scan_ui()

    def on_device_toggled(self, mac: str, selected: bool):
        asyncio.create_task(self._toggle_device(mac, selected))

    async def _toggle_device(self, mac: str, connect: bool):
        try:
            if connect:
                self.window.set_status(f"Connecting to {mac}...")
                self.window.set_connection_state(mac, "connecting")
                
                success = await calibrationControl('connect', deviceId=mac)
                if success:
                    self.window.set_status(f"Connected to {mac}")
                    self.window.set_connection_state(mac, "connected")
                    self.connected_devices.add(mac)
                    
                    # Start streaming to get wearing status
                    # We use a dummy start to enable notifications
                    try:
                        await calibrationControl('start', subjectId=mac, stateLabel='MONITOR')
                    except Exception as e:
                        logger.warning(f"Could not start monitoring for {mac}: {e}")
                        
                else:
                    self.window.set_status(f"Failed to connect to {mac}")
                    self.window.set_connection_state(mac, "disconnected")
            else:
                self.window.set_status(f"Disconnecting from {mac}...")
                await calibrationControl('disconnect', deviceId=mac)
                self.window.set_status(f"Disconnected from {mac}")
                self.window.set_connection_state(mac, "disconnected")
                self.connected_devices.discard(mac)
        except Exception as exc:
            logger.error(f"Connection error for {mac}: {exc}")
            self.window.set_status(f"Connection error for {mac}: {exc}")
            self.window.set_connection_state(mac, "disconnected")

    async def _update_status_loop(self):
        while self._running:
            try:
                for mac in list(self.connected_devices):
                    # Wearing status
                    parser = _multi_parsers.get(mac)
                    if parser:
                        is_wearing = parser.wearing_status
                        self.window.set_wearing_status(mac, is_wearing)
                
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"Status loop error: {e}")
                await asyncio.sleep(1.0)

    def on_play_requested(self, players_info: List[dict]):
        self.window.hide()
        asyncio.create_task(self._run_game_sequence(players_info))

    async def _run_game_sequence(self, players_info: List[dict]):
        macs = [p["mac"] for p in players_info]
        logger.info(f"Starting play sequence for {macs}")
        
        # Setup players
        players = []
        for i, p_info in enumerate(players_info):
            players.append(play_config.PlayerConfig(
                number=i+1, 
                name=p_info["name"], 
                wearing=True
            ))
        
        # Shared state for the game loop
        game_state = {
            "strengths": [50] * len(macs),
            "wearing": [True] * len(macs),
            "running": True
        }

        # Helper to run a stage
        async def run_stage(task_type, duration):
            logger.info(f"Starting stage: {task_type}")
            
            # Configure session
            play_config.get_default_session_config = lambda: play_config.SessionConfig(
                task_type=task_type,
                players=players,
                duration_seconds=duration
            )
            
            # Start data collection
            for mac in macs:
                await calibrationControl('start', subjectId=mac, stateLabel=task_type)
            
            # Run UI with shared state
            # We need to run the game loop and a background task to update state simultaneously
            game_task = asyncio.create_task(play_ui.run_game_loop(game_state))
            
            # Monitor loop for this stage
            start_time = asyncio.get_event_loop().time()
            while not game_task.done():
                # Update wearing status
                current_wearing = []
                for mac in macs:
                    parser = _multi_parsers.get(mac)
                    if parser:
                        current_wearing.append(parser.wearing_status)
                    else:
                        current_wearing.append(False)
                game_state["wearing"] = current_wearing
                
                # If we are in LIVE mode, we might want to update strengths from inference
                # But inference is callback based. We'll handle that separately.
                
                await asyncio.sleep(0.1)

            await game_task
            
            # Stop data collection
            for mac in macs:
                await calibrationControl('stop', deviceId=mac)

        # 1. Relax
        await run_stage("RELAX", 10)
        
        # 2. Focus
        await run_stage("FOCUS", 10)
        
        # 3. Live Play
        logger.info("Training models...")
        from ifocus_sdk.APIs.trainFocusModel import trainFocusModel
        
        # We can show a "Training..." status on the UI if we had a loading screen,
        # but for now we just log it.
        for mac in macs:
            def cb(success, msg, n):
                logger.info(f"[{mac}] Training: {msg}")
            trainFocusModel(mac, cb)
            
        # Start Inference
        logger.info("Starting Inference...")
        from ifocus_sdk.APIs.FocusInference import startFocusInference, stopFocusInference
        
        # Map mac to player index for callback
        mac_to_idx = {mac: i for i, mac in enumerate(macs)}
        
        def inference_callback(mac, score):
            if mac in mac_to_idx:
                idx = mac_to_idx[mac]
                # Score is 0-100
                game_state["strengths"][idx] = int(score)
                # logger.info(f"Inference {mac}: {score}")

        for mac in macs:
            startFocusInference(mac, inference_callback)
        
        try:
            await run_stage("LIVE", 60)
        finally:
            for mac in macs:
                stopFocusInference(mac)
        
        logger.info("Game sequence finished.")
        self.window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    win = IFocusWindow()
    controller = GameController(win)
    controller.start()
    
    def cleanup():
        asyncio.ensure_future(controller.stop())
    app.aboutToQuit.connect(cleanup)
    
    win.show()
    
    with loop:
        loop.run_forever()

