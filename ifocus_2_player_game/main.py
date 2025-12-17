from __future__ import annotations

import asyncio
import sys
import time
import threading
from pathlib import Path
import logging

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# On Windows, prefer selector loop to avoid qasync/proactor re-entrancy issues
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
logger.setLevel(logging.INFO)  # Keep main at INFO, but allow DEBUG from SDK

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
        
        # Disable wearing check for now to collect data even if headset shows not worn
        logger.info("Disabling wearing check to allow data collection")
        setWearingCheckEnabled(False)

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
                
                # Always use multi-device path and bind subject to MAC to keep routing consistent
                success = await calibrationControl('connect', deviceId=[mac], subjectId=[mac])
                if success:
                    self.window.set_status(f"Connected to {mac}")
                    self.window.set_connection_state(mac, "connected")
                    self.connected_devices.add(mac)
                    
                    # Start streaming to get wearing status
                    # We use a dummy start to enable notifications
                    try:
                        # Provide deviceId so multi-device manager can bind notifications immediately
                        await calibrationControl('start', deviceId=mac, subjectId=mac, stateLabel='MONITOR')
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

    def on_play_requested(self, players_info: list[dict]):
        self.window.hide()
        asyncio.create_task(self._run_game_sequence(players_info))

    async def _run_game_sequence(self, players_info: list[dict]):
        macs = [p["mac"] for p in players_info]
        logger.info(f"Starting play sequence for {macs}")
        
        # Create mapping from MAC to player name for data collection
        mac_to_name = {p["mac"]: p["name"] for p in players_info}
        
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
            "running": True,
            "stage": {
                "id": "init",
                "type": "RELAX",
                "duration": 10
            }
        }

        # Configure initial session config for the UI to pick up players
        play_config.get_default_session_config = lambda: play_config.SessionConfig(
            task_type="RELAX",
            players=players,
            duration_seconds=10
        )
        
        # Start the pygame UI in its own event loop/thread to avoid re-entrancy with qasync/Qt
        def _run_ui():
            asyncio.run(play_ui.run_game_loop(game_state))

        ui_thread = threading.Thread(target=_run_ui, daemon=True)
        ui_thread.start()

        async def run_stage_logic(task_type, duration):
            logger.info(f"Starting stage: {task_type}")
            
            # Update game state to trigger UI transition
            game_state["stage"] = {
                "id": f"{task_type}_{time.time()}",
                "type": task_type,
                "duration": duration
            }
            
            # Start data collection using player names as subjectId
            for mac in macs:
                player_name = mac_to_name[mac]
                logger.info(f"Starting data collection for {player_name} (device {mac}) - {task_type}")
                await calibrationControl('start', deviceId=mac, subjectId=player_name, stateLabel=task_type)
            
            # Wait for duration
            end_time = asyncio.get_event_loop().time() + duration
            while asyncio.get_event_loop().time() < end_time:
                if not ui_thread.is_alive():
                    logger.warning("UI closed unexpectedly")
                    break
                
                # Update wearing status
                current_wearing = []
                for mac in macs:
                    parser = _multi_parsers.get(mac)
                    if parser:
                        current_wearing.append(parser.wearing_status)
                    else:
                        current_wearing.append(False)
                game_state["wearing"] = current_wearing
                
                await asyncio.sleep(0.1)
            
            # Stop data collection using player names
            for mac in macs:
                player_name = mac_to_name[mac]
                result = await calibrationControl('stop', deviceId=mac, subjectId=player_name)
                logger.info(f"Stopped data collection for {player_name}: {result}")

        async def run_live_stage(duration):
            """Live stage driven only by inference (no extra recording)."""
            task_type = "LIVE"
            logger.info(f"Starting stage: {task_type}")

            game_state["stage"] = {
                "id": f"{task_type}_{time.time()}",
                "type": task_type,
                "duration": duration
            }

            end_time = asyncio.get_event_loop().time() + duration
            while asyncio.get_event_loop().time() < end_time:
                if not ui_thread.is_alive():
                    logger.warning("UI closed unexpectedly")
                    break

                current_wearing = []
                for mac in macs:
                    parser = _multi_parsers.get(mac)
                    if parser:
                        current_wearing.append(parser.wearing_status)
                    else:
                        current_wearing.append(False)
                game_state["wearing"] = current_wearing

                await asyncio.sleep(0.1)

        async def run_inference_session(calibrate: bool, live_duration: int = 60):
            """Optionally recalibrate/train, then run inference + live stage."""

            if calibrate:
                await run_stage_logic("RELAX", 10)
                await run_stage_logic("FOCUS", 10)

                logger.info("Training models...")
                from ifocus_sdk.APIs.trainFocusModel import trainFocusModel

                for mac in macs:
                    player_name = mac_to_name[mac]
                    def cb(success, msg, n):
                        logger.info(f"[{player_name}] Training: {msg}")
                    trainFocusModel(player_name, cb)

            # Start Inference
            logger.info("Starting Inference...")
            from ifocus_sdk.APIs.FocusInference import startFocusInference, stopFocusInference

            mac_to_idx = {mac: i for i, mac in enumerate(macs)}
            inference_tasks = []
            for mac in macs:
                client = await calibrationControl('client', deviceId=mac)
                if client:
                    idx = mac_to_idx[mac]
                    player_name = mac_to_name[mac]

                    def make_callback(device_idx):
                        def cb(label, strength, timestamp):
                            game_state["strengths"][device_idx] = int(strength)
                        return cb

                    task = asyncio.create_task(
                        startFocusInference(player_name, client, updateHz=2.0, callback=make_callback(idx))
                    )
                    inference_tasks.append((mac, player_name, client, task))

            try:
                await run_live_stage(live_duration)
            finally:
                for mac, player_name, client, task in inference_tasks:
                    try:
                        await stopFocusInference(subjectId=player_name, client=client)
                    except Exception as e:
                        logger.warning(f"stopFocusInference failed for {player_name}/{mac}: {e}")
                for _, _, _, task in inference_tasks:
                    if not task.done():
                        task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        action = "start_fresh"
        first_run = True

        try:
            while True:
                calibrate = action == "start_fresh" or first_run
                first_run = False

                if action == "start_fresh" and not calibrate:
                    # should not happen, but guard
                    calibrate = True

                if action == "start_fresh" and not first_run:
                    # Wipe previous data/models when starting fresh after a round
                    for mac in macs:
                        player_name = mac_to_name[mac]
                        try:
                            await calibrationControl('reset', subjectId=player_name)
                            logger.info(f"Cleared calibration/model data for {player_name}")
                        except Exception as e:
                            logger.warning(f"Reset failed for {player_name}: {e}")

                # Run session (calibration+train+live or just live with existing model)
                await run_inference_session(calibrate=calibrate, live_duration=60)

                # Wait for user choice from UI (rematch or start fresh)
                chosen = None
                while chosen is None:
                    if not ui_thread.is_alive():
                        chosen = "quit"
                        break
                    chosen = game_state.get("action")
                    if chosen is None:
                        await asyncio.sleep(0.1)

                game_state["action"] = None
                action = chosen

                if chosen == "rematch":
                    # Replay live using existing models
                    continue
                elif chosen == "start_fresh":
                    action = "start_fresh"
                    continue
                else:
                    break

        finally:
            logger.info("Game sequence finished.")
            game_state["running"] = False
            if ui_thread.is_alive():
                ui_thread.join(timeout=2.0)
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

