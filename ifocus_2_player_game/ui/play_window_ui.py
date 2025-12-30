import sys
from pathlib import Path
import math
import random
import asyncio
import logging
import shutil
import subprocess
from typing import Optional, Dict, Any, List

import pygame

try:
    import play_window_config
except ImportError:
    from . import play_window_config


# -----------------------------
# Configuration
# -----------------------------
# Base logical window size; actual size is scaled
# using a similar principle to ifocus_ui.py.
BASE_WIDTH = 960
BASE_HEIGHT = 540
FPS = 60

# Strength controls vertical position of the character.
# 0   -> bottom
# 100 -> top
MIN_STRENGTH = 0
MAX_STRENGTH = 100
STRENGTH_STEP = 2


# -----------------------------
# Visual style (Apple-like)
# -----------------------------
SYSTEM_FONT_STACK = [
    "SF Pro Display",
    "SF Pro Text",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
]

logger = logging.getLogger(__name__)


def make_font(size: int, *, bold: bool = False) -> pygame.font.Font:
    """Return a high-quality, Apple-like system font.

    Uses a stack of common macOS / iOS fonts with sensible
    fallbacks so the UI feels crisp on most platforms.
    """

    return pygame.font.SysFont(SYSTEM_FONT_STACK, size, bold=bold)


def draw_glass_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    base_color: tuple[int, int, int] = (255, 255, 255),
    radius: int = 18,
    fill_alpha: int = 120,
    border_alpha: int = 200,
) -> None:
    """Draw a simple glassmorphism-style panel.

    This mimics "liquid glass" with a soft translucent fill and
    a subtle lighter border, without doing real background blur.
    """

    # Shadow
    shadow_offset = max(2, radius // 4)
    shadow_rect = rect.move(shadow_offset, shadow_offset)
    shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
    pygame.draw.rect(
        shadow_surf,
        (15, 23, 42, 60),  # soft dark shadow
        shadow_surf.get_rect(),
        border_radius=radius,
    )
    surface.blit(shadow_surf, shadow_rect.topleft)

    # Main glass panel
    panel_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
    r, g, b = base_color
    pygame.draw.rect(
        panel_surf,
        (r, g, b, fill_alpha),
        panel_surf.get_rect(),
        border_radius=radius,
    )
    pygame.draw.rect(
        panel_surf,
        (255, 255, 255, border_alpha),
        panel_surf.get_rect(),
        width=1,
        border_radius=radius,
    )
    surface.blit(panel_surf, rect.topleft)


def load_image(name: str) -> pygame.Surface:
    """Load image from the shared assets folder."""
    base_dir = Path(__file__).resolve().parents[1]  # .. / ifocus_2_player_game
    asset_path = base_dir / "assets" / name
    try:
        image = pygame.image.load(str(asset_path)).convert_alpha()
    except pygame.error as exc:  # pragma: no cover - runtime-only path
        raise SystemExit(f"Unable to load image '{name}': {exc}") from exc
    return image


def try_load_image(name: str):
    """Best-effort image loader that returns None if the asset is missing.

    Used for optional decorative/obstacle sprites so the game can still run
    even if a particular file isn't present.
    """

    base_dir = Path(__file__).resolve().parents[1]
    asset_path = base_dir / "assets" / name
    if not asset_path.exists():
        return None
    try:
        return pygame.image.load(str(asset_path)).convert_alpha()
    except pygame.error:
        return None


def load_sound(name: str) -> Optional[pygame.mixer.Sound]:
    """Load a sound effect from assets, converting if necessary."""

    base_dir = Path(__file__).resolve().parents[1]
    sound_path = base_dir / "assets" / name
    if not sound_path.exists():
        logger.warning("Sound effect %s not found.", name)
        return None

    def _try(path: Path) -> Optional[pygame.mixer.Sound]:
        try:
            return pygame.mixer.Sound(str(path))
        except pygame.error:
            return None

    sound = _try(sound_path)
    if sound:
        return sound

    suffix = sound_path.suffix.lower()
    if suffix in {".m4a", ".mp4", ".aac"}:
        converted_path = sound_path.with_suffix(".wav")
        if converted_path.exists():
            sound = _try(converted_path)
            if sound:
                return sound

        ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if ffmpeg:
            try:
                subprocess.run(
                    [ffmpeg, "-y", "-i", str(sound_path), str(converted_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                sound = _try(converted_path)
                if sound:
                    return sound
            except Exception as exc:
                logger.warning("Failed to convert %s via ffmpeg: %s", name, exc)
        else:
            logger.warning(
                "ffmpeg not found; unable to convert %s. Provide a WAV/OGG version for playback.",
                name,
            )
    else:
        logger.warning("Unable to load sound %s (unsupported format).", name)

    return None


def map_strength_to_y(
    strength: int,
    screen_height: int,
    sprite_height: int,
    top_margin: int = 40,
    bottom_margin: int = 40,
) -> int:
    """Map strength (0-100) to a vertical screen position.

    strength == 100 -> near top (small y)
    strength == 0   -> near bottom (large y)
    """

    strength = max(MIN_STRENGTH, min(MAX_STRENGTH, strength))

    min_y = top_margin
    max_y = screen_height - sprite_height - bottom_margin
    if max_y <= min_y:
        return min_y

    ratio = (strength - MIN_STRENGTH) / (MAX_STRENGTH - MIN_STRENGTH)
    # invert so that higher strength moves the character up
    y = max_y - int(ratio * (max_y - min_y))
    return y


async def run_game_loop(game_state: Optional[Dict[str, Any]] = None) -> None:
    """
    Main game loop.
    
    Args:
        game_state: Optional dictionary to share state with the controller.
                    Expected keys:
                    - 'strengths': List[int] (0-100)
                    - 'wearing': List[bool]
                    - 'running': bool (set to False to stop loop)
    """
    pygame.init()
    pygame.display.set_caption("iFocus Play Window")

    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        mixer_available = True
    except pygame.error:
        mixer_available = False

    # Derive a UI scale based on the current screen, similar in
    # spirit to IFocusWindow in ifocus_ui.py, but also ensuring
    # the window never exceeds the physical screen size.
    display_info = pygame.display.Info()
    screen_w, screen_h = display_info.current_w, display_info.current_h

    # Use maximized window mode with title bar (not fullscreen)
    # This allows users to close the window easily
    window_width = screen_w
    window_height = screen_h
    ui_scale = min(screen_w / 1440.0, screen_h / 900.0)
    if ui_scale <= 0:
        ui_scale = 0.5

    screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    # Session / player configuration (isolated from UI logic)
    session_cfg = play_window_config.get_default_session_config()
    players = session_cfg.players
    if not players:
        raise SystemExit("play_window_config must define at least one player.")

    # Initial Game Mode Behavior (will be updated by game_state)
    task_type = session_cfg.task_type.upper()
    
    # Defaults for LIVE mode
    show_score = True
    show_projectiles = True
    auto_exit_on_finish = False
    theme_color_override = None # None means use default logic or specific colors
    
    def update_stage_config(new_task_type: str):
        nonlocal show_score, show_projectiles, auto_exit_on_finish, theme_color_override, task_type
        task_type = new_task_type.upper()
        
        is_calibration = "RELAX" in task_type or "FOCUS" in task_type
        show_score = not is_calibration
        show_projectiles = not is_calibration
        auto_exit_on_finish = is_calibration
        
        if "RELAX" in task_type:
            theme_color_override = (59, 130, 246)  # Cool Blue
        elif "FOCUS" in task_type:
            theme_color_override = (249, 115, 22)  # Warm Orange
        else:
            theme_color_override = None

    update_stage_config(task_type)

    # Load images
    background = load_image("skyblue.png")
    cloud_img = load_image("cloud.png")
    cloud2_img = load_image("cloud2.png")
    dragon_img = load_image("dragonup.png")
    dragon_img2 = load_image("dragondown.png")
    wing_flap_sound = load_sound("wing_down.m4a") if mixer_available else None
    fireball_sound = load_sound("fireball.mp3") if mixer_available else None
    arrow_single_sound = load_sound("single_arrow.mp3") if mixer_available else None
    arrow_multi_sound = load_sound("multiple_arrow.mp3") if mixer_available else None
    hit_sound = load_sound("hit.mp3") if mixer_available else None

    if wing_flap_sound:
        wing_flap_sound.set_volume(0.9)
    if fireball_sound:
        fireball_sound.set_volume(1.0)
    if arrow_single_sound:
        arrow_single_sound.set_volume(1.0)
    if arrow_multi_sound:
        arrow_multi_sound.set_volume(1.0)
    if hit_sound:
        hit_sound.set_volume(1.0)

    music_started = False
    if mixer_available:
        music_path = Path(__file__).resolve().parents[1] / "assets" / "background_music.mp3"
        if music_path.exists():
            try:
                pygame.mixer.music.load(str(music_path))
                pygame.mixer.music.set_volume(0.18)
                pygame.mixer.music.play(-1)
                music_started = True
            except pygame.error as exc:
                logger.warning("Unable to load background music %s: %s", music_path.name, exc)
        else:
            logger.warning("Background music %s not found.", music_path.name)

    # Optional projectile / obstacle 3.3
    fireball_img = try_load_image("fireball.png")
    arrow_img = try_load_image("arrow.png")
    javelin_img = try_load_image("javelin.png")
    obstacle_img = try_load_image("obstacle.png")

    # Scale background to fill window
    background = pygame.transform.smoothscale(background, (window_width, window_height))

    # Downscale clouds and character so they fit nicely in the
    # scene, adapting to the window size.
    # Clouds
    def scale_to_width(img: pygame.Surface, target_w: int) -> pygame.Surface:
        w, h = img.get_size()
        if w == 0:
            return img
        scale = target_w / float(w)
        new_size = (int(w * scale), int(h * scale))
        return pygame.transform.smoothscale(img, new_size)

    cloud_img = scale_to_width(cloud_img, int(window_width * 0.18))
    cloud2_img = scale_to_width(cloud2_img, int(window_width * 0.22))

    # Dragon: scale by height so it does not dominate the screen
    dragon_h_target = int(window_height * 0.14)
    d_w, d_h = dragon_img.get_size()
    if d_h > 0:
        d_scale = dragon_h_target / float(d_h)
        dragon_size = (int(d_w * d_scale), int(d_h * d_scale))
        dragon_img = pygame.transform.smoothscale(dragon_img, dragon_size)
        dragon_img2 = pygame.transform.smoothscale(dragon_img2, dragon_size)

    # Projectile configuration (slow, readable motion)
    projectile_types = []
    # Make base speed quite slow so movement is gentle
    # Use config from game_state if available, otherwise default
    base_speed_multiplier = game_state.get("projectile_base_speed", 80) if game_state else 80
    base_speed = base_speed_multiplier * ui_scale

    # Scale projectiles so they sit nicely in the scene.
    proj_target_w = int(window_width * 0.07)

    # Load and configure projectiles
    projectile_configs = [
        ("fireball", fireball_img, 0.7),
        ("arrow", arrow_img, 0.8),
        ("javelin", javelin_img, 0.6),
        ("obstacle", obstacle_img, 0.5),
    ]
    
    for name, img, speed_mult in projectile_configs:
        if img is not None:
            img = scale_to_width(img, proj_target_w)
            if name == "obstacle":
                img = pygame.transform.flip(img, True, False)  # Face toward players
            projectile_types.append({"name": name, "image": img, "speed": base_speed * speed_mult})

    # Simple cloud layers for parallax-like motion
    cloud_speed_1 = 40  # pixels / second
    cloud_speed_2 = 80

    cloud1_x = 0.0
    cloud2_x = window_width * 0.5

    cloud1_y = window_height * 0.18
    cloud2_y = window_height * 0.32

    # Character strengths and scores (one per configured player)
    # Local state (fallback if game_state not provided)
    local_strengths = [50 for _ in players]
    local_wearing = [True for _ in players]
    
    # Interpolated strengths for smooth movement (avoids teleporting)
    interpolated_strengths = [50.0 for _ in players]
    
    scores = [0 for _ in players]
    hit_timers = [0.0 for _ in players]
    
    # Simulation state for calibration stages (RELAX and FOCUS)
    # During calibration, we simulate dragon movement to guide users
    simulation_active = False
    simulation_target_strengths = [50.0 for _ in players]  # Target values for simulation
    simulation_direction = [1.0 for _ in players]  # 1.0 for ascending, -1.0 for descending
    simulation_speed = 8.0  # Units per second (slow movement)
    simulation_reverse_timer = 0.0  # Timer for occasional reverse movements
    simulation_reverse_interval = 4.0  # Reverse direction every 4-6 seconds
    simulation_reverse_duration = 0.8  # Duration of reverse movement (shorter, more abrupt)

    # Wing animation timing
    wing_animation_timer = 0.0
    wing_flap_interval = 0.6  # seconds per wing state to match audio
    wing_was_down = False

    # Round timer (seconds)
    round_duration = max(1, getattr(session_cfg, "duration_seconds", 60))
    time_remaining = float(round_duration)
    second_accumulator = 0.0

    # Projectiles
    projectiles: list[dict] = []  # each: {image, rect, x, y, vx, vy, age, swirl_amp, swirl_freq, swirl_phase, name}
    # Start accumulator closer to spawn interval to spawn projectiles earlier
    projectile_spawn_accumulator = 1.0
    # Spawn very slowly so players are nudged to move,
    # but not overwhelmed.
    # Use config from game_state if available, otherwise default
    projectile_spawn_interval = game_state.get("projectile_spawn_interval", 8.0) if game_state else 8.0  # seconds
    # Remember which player was targeted last so we can avoid
    # picking the same one twice in a row when possible.
    last_projectile_target_idx = None

    # Fonts (Apple-like system stack), kept uniform in style
    # so text looks smooth and consistent across the HUD.
    base_font_size = max(18, int(18 * ui_scale))
    font = make_font(base_font_size)
    small_font = make_font(max(14, int(14 * ui_scale)))
    task_font = make_font(max(22, int(24 * ui_scale)), bold=True)

    # Helper: map strength (0-100) to a warm "temperature" color.
    # 0   -> relaxed (cool color), 50 -> mid, 100 -> focused (orangish).
    def strength_to_color(value: int) -> tuple[int, int, int]:
        if theme_color_override:
            return theme_color_override
            
        value = max(MIN_STRENGTH, min(MAX_STRENGTH, value))
        t = value / float(MAX_STRENGTH or 1)

        # Cool relaxed color (soft blue/teal)
        r0, g0, b0 = 59, 130, 246
        # Warm focused color (orange)
        r1, g1, b1 = 249, 115, 22

        r = int(r0 + (r1 - r0) * t)
        g = int(g0 + (g1 - g0) * t)
        b = int(b0 + (b1 - b0) * t)
        return r, g, b

    # Game state
    running = True
    state = "playing"  # or "game_over", "waiting_next_stage", "preparation", "training_results"
    winner_pulse_time = 0.0
    rematch_button_rect = None
    startfresh_button_rect = None
    
    current_stage_id = None
    
    # Preparation phase variables
    preparation_duration = 5.0  # 5 seconds countdown
    preparation_time_remaining = 0.0
    preparation_task_type = ""  # What task is coming up
    
    # Training results display
    training_results_duration = 4.0  # 4 seconds to display results
    training_results_time_remaining = 0.0

    # Use asyncio sleep for timing to avoid blocking the event loop with clock.tick
    target_frame_time = 1.0 / FPS
    last_time = asyncio.get_event_loop().time()

    while running:
        # Calculate dt using asyncio time
        current_time = asyncio.get_event_loop().time()
        dt = current_time - last_time
        
        # Cap dt to avoid huge jumps if lag occurs
        if dt > 0.1:
            dt = 0.1
            
        # Sleep to maintain FPS
        sleep_time = target_frame_time - (asyncio.get_event_loop().time() - current_time)
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
            # Update time after sleep
            current_time = asyncio.get_event_loop().time()
            dt = current_time - last_time
            if dt > 0.1: dt = 0.1
        else:
            await asyncio.sleep(0) # Yield at least once
            
        last_time = current_time

        # Update wing animation
        wing_animation_timer += dt
        wing_cycle = max(wing_flap_interval * 2.0, 1e-6)
        while wing_animation_timer >= wing_cycle:
            wing_animation_timer -= wing_cycle

        wing_down_active = wing_animation_timer >= wing_flap_interval
        if (
            wing_down_active
            and not wing_was_down
            and wing_flap_sound
            and state == "playing"
        ):
            wing_flap_sound.play()
        wing_was_down = wing_down_active

        # Select which dragon image to use based on animation timer
        current_dragon_img = dragon_img2 if wing_down_active else dragon_img

        # Sync with external game_state if provided
        if game_state:
            if not game_state.get("running", True):
                running = False
            
            # Check for stage transition
            stage_info = game_state.get("stage")
            if stage_info and stage_info.get("id") != current_stage_id:
                current_stage_id = stage_info.get("id")
                
                # Update configuration
                new_task_type = stage_info.get("type", "LIVE")
                
                # Check if this is the training results stage
                if new_task_type == "TRAINING_RESULTS":
                    state = "training_results"
                    training_results_time_remaining = training_results_duration
                    continue  # Skip rest of stage setup for training results
                
                update_stage_config(new_task_type)
                
                # Reset round state
                round_duration = max(1, stage_info.get("duration", 60))
                time_remaining = float(round_duration)
                second_accumulator = 0.0
                
                scores = [0 for _ in players]
                hit_timers = [0.0 for _ in players]
                projectiles.clear()
                # Start accumulator at 1.0 to spawn projectiles earlier
                projectile_spawn_accumulator = 1.0
                last_projectile_target_idx = None
                winner_pulse_time = 0.0
                
                # Reset simulation state and set starting positions based on task type
                simulation_reverse_timer = 0.0
                new_task_lower = new_task_type.lower()
                
                # Helper to set all player strengths
                def set_all_strengths(value: float):
                    for i in range(len(players)):
                        simulation_target_strengths[i] = value
                        local_strengths[i] = int(value)
                        interpolated_strengths[i] = value
                
                # Enter preparation state for RELAX and FOCUS stages
                if "relax" in new_task_lower or "focus" in new_task_lower:
                    state = "preparation"
                    preparation_time_remaining = preparation_duration
                    preparation_task_type = new_task_type
                else:
                    state = "playing"
                
                # Set initial positions
                if "relax" in new_task_lower:
                    set_all_strengths(100.0)  # Start from top
                elif "focus" in new_task_lower:
                    set_all_strengths(0.0)  # Start from bottom
                else:
                    set_all_strengths(50.0)  # Start from middle
            
            # Update strengths safely
            ext_strengths = game_state.get("strengths", [])
            for i in range(min(len(local_strengths), len(ext_strengths))):
                local_strengths[i] = ext_strengths[i]
                
            # Update wearing safely
            ext_wearing = game_state.get("wearing", [])
            for i in range(min(len(local_wearing), len(ext_wearing))):
                local_wearing[i] = ext_wearing[i]
        
        # Determine if simulation should be active based on current task type
        current_task_lower = task_type.lower()
        simulation_active = "relax" in current_task_lower or "focus" in current_task_lower
        
        # Simulation logic for calibration stages
        if simulation_active and state == "playing":
            simulation_reverse_timer += dt
            
            # Check if it's time to reverse direction
            if simulation_reverse_timer >= simulation_reverse_interval:
                simulation_reverse_timer = 0.0
                simulation_reverse_interval = 4.0 + random.random() * 2.0  # Randomize next interval
                for i in range(len(players)):
                    simulation_direction[i] *= -1.0
            
            in_reverse_phase = simulation_reverse_timer < simulation_reverse_duration
            base_direction = -1.0 if "relax" in current_task_lower else 1.0  # -1=descend, 1=ascend
            
            # Update simulated target strengths
            for i in range(len(players)):
                effective_direction = -base_direction if in_reverse_phase else base_direction
                simulation_target_strengths[i] += effective_direction * simulation_speed * dt
                simulation_target_strengths[i] = max(MIN_STRENGTH, min(MAX_STRENGTH, simulation_target_strengths[i]))
                local_strengths[i] = int(simulation_target_strengths[i])
        
        # Smoothly interpolate character positions
        for i in range(len(players)):
            diff = local_strengths[i] - interpolated_strengths[i]
            # Lower interpolation gain to make vertical movement less abrupt.
            interpolated_strengths[i] += diff * min(1.0, 4.0 * dt)

        # Event handling
        # Process pygame events carefully to avoid interfering with Qt event loop
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    if game_state is not None:
                        game_state["running"] = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        if game_state is not None:
                            game_state["running"] = False
                    # Removed K_UP/K_DOWN manual control
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if state == "game_over":
                        if rematch_button_rect and rematch_button_rect.collidepoint(event.pos):
                            # Signal controller to rematch using existing models
                            if game_state is not None:
                                game_state["action"] = "rematch"
                            state = "waiting_next_stage"
                        elif startfresh_button_rect and startfresh_button_rect.collidepoint(event.pos):
                            # Signal controller to wipe data and recalibrate
                            if game_state is not None:
                                game_state["action"] = "start_fresh"
                            state = "waiting_next_stage"
        except Exception:
            pass  # Ignore event processing errors to avoid blocking

        # Countdown logic for special states
        if state == "training_results":
            training_results_time_remaining -= dt
            if training_results_time_remaining <= 0:
                state = "waiting_next_stage"
                training_results_time_remaining = 0.0
        elif state == "preparation":
            preparation_time_remaining -= dt
            if preparation_time_remaining <= 0:
                state = "playing"
                preparation_time_remaining = 0.0

        # Timer and scoring: each full second that passes
        # awards +1 score point to every player while time remains.
        if state == "playing" and time_remaining > 0:
            time_remaining -= dt
            second_accumulator += dt
            while second_accumulator >= 1.0 and time_remaining > 0:
                second_accumulator -= 1.0
                if show_score:
                    for i in range(len(scores)):
                        scores[i] += 1

            if time_remaining <= 0:
                time_remaining = 0
                if auto_exit_on_finish:
                    # If controlled externally, we don't want to exit the loop, just wait for next stage
                    if game_state:
                        state = "waiting_next_stage"
                    else:
                        running = False # Exit loop immediately for calibration modes
                else:
                    state = "game_over"
                    winner_pulse_time = 0.0

        if state == "game_over":
            winner_pulse_time += dt

        # Update hit animation timers
        if state == "playing":
            for i in range(len(hit_timers)):
                if hit_timers[i] > 0:
                    hit_timers[i] = max(0.0, hit_timers[i] - dt)

        # Spawn and update projectiles only during active play
        if show_projectiles and projectile_types and state == "playing":
            projectile_spawn_accumulator += dt
            if projectile_spawn_accumulator >= projectile_spawn_interval:
                projectile_spawn_accumulator -= projectile_spawn_interval

                # Spawn a slow-moving projectile aimed roughly at a player.
                # Ensure we don't always target the same player twice in a row
                # when multiple players are present, for fairness.
                if len(players) > 1:
                    candidate_indices = [
                        i for i in range(len(players)) if i != last_projectile_target_idx
                    ]
                    target_idx = random.choice(candidate_indices)
                else:
                    target_idx = 0
                last_projectile_target_idx = target_idx
                proj_type = random.choice(projectile_types)
                img = proj_type["image"]
                if img is not None:
                    # 70% of projectiles target bottom/middle, 30% target upper area
                    # to discourage zoning out (staying relaxed at bottom)
                    if random.random() < 0.7:
                        # Target bottom and middle: strength 0-60
                        biased_strength = random.uniform(0, 60)
                    else:
                        # Target upper area: strength 60-100
                        biased_strength = random.uniform(60, 100)
                    
                    dragon_y_for_target = map_strength_to_y(
                        biased_strength,
                        window_height,
                        current_dragon_img.get_height(),
                        # Expanded range so targets can appear nearer the top and bottom edges.
                        top_margin=int(30 * ui_scale),
                        bottom_margin=int(0 * ui_scale),
                    )

                    # Choose a target point slightly toward the left side
                    target_x = int(window_width * 0.12)

                    # For fireballs and javelins, spawn a small non-linear
                    # cluster (staggered vertically) to cover more space.
                    if proj_type["name"] in ("fireball", "javelin"):
                        cluster_patterns = [
                            [-1, 0, 1],
                            [0, 1],
                            [-1, 1],
                            [0],
                        ]
                        pattern = random.choice(cluster_patterns)
                    else:
                        pattern = [0]

                    launch_sound = None
                    if mixer_available:
                        if proj_type["name"] == "fireball":
                            launch_sound = fireball_sound
                        elif proj_type["name"] in ("arrow", "javelin"):
                            if len(pattern) > 1:
                                launch_sound = arrow_multi_sound or arrow_single_sound
                            else:
                                launch_sound = arrow_single_sound

                    vertical_step = int(current_dragon_img.get_height() * 0.2)

                    for offset_index in pattern:
                        spawn_y = int(
                            dragon_y_for_target
                            + current_dragon_img.get_height() * 0.35
                            + offset_index * vertical_step
                        )

                        # Start just off the right edge
                        spawn_x = window_width + img.get_width()

                        # Aim roughly towards the chosen target point with
                        # slight random vertical offset to vary the arc.
                        aim_y = dragon_y_for_target + random.uniform(
                            -current_dragon_img.get_height() * 0.2,
                            current_dragon_img.get_height() * 0.2,
                        )
                        dx = target_x - spawn_x
                        dy = aim_y - spawn_y
                        dist = math.hypot(dx, dy) or 1.0
                        speed = proj_type["speed"]
                        vx = speed * dx / dist
                        vy = speed * dy / dist

                        rect = img.get_rect()
                        rect.center = (int(spawn_x), int(spawn_y))

                        # All projectiles the player "throws" (arrow, javelin, fireball)
                        # should move in a clean, linear path. We keep a slight swirl
                        # only for generic obstacles to make them feel a bit more organic.
                        if proj_type["name"] in ("arrow", "javelin", "fireball"):
                            swirl_amp = 0.0
                        elif proj_type["name"] == "obstacle":
                            swirl_amp = 10.0 * ui_scale
                        else:
                            swirl_amp = 0.0

                        projectiles.append(
                            {
                                "name": proj_type["name"],
                                "image": img,
                                "rect": rect,
                                "x": float(spawn_x),
                                "y": float(spawn_y),
                                "vx": vx,
                                "vy": vy,
                                "age": 0.0,
                                "swirl_amp": swirl_amp,
                                "swirl_freq": random.uniform(4.0, 8.0),
                                "swirl_phase": random.uniform(0.0, math.tau),
                                # Track which players this projectile has already hit
                                "hit_players": set(),
                            }
                        )

                    if launch_sound:
                        launch_sound.play()

        # Move projectiles (with optional swirl) and check collisions with player dragons
        if state == "playing":
            for proj in projectiles[:]:
                proj["age"] += dt

                # Base linear motion using standard integration
                proj["x"] += proj["vx"] * dt
                proj["y"] += proj["vy"] * dt
                base_y = proj["y"]

                # Swirl offset for a more natural, wavy trajectory (disabled when amp=0)
                if proj["swirl_amp"] > 0.0:
                    swirl = proj["swirl_amp"] * math.sin(
                        proj["age"] * proj["swirl_freq"] + proj["swirl_phase"]
                    )
                else:
                    swirl = 0.0
                final_y = base_y + swirl

                proj_rect = proj["rect"]
                proj_rect.centerx = int(proj["x"])
                proj_rect.centery = int(final_y)

                # Remove if it goes off-screen
                if (
                    proj_rect.right < 0
                    or proj_rect.bottom < 0
                    or proj_rect.top > window_height
                ):
                    projectiles.remove(proj)
                    continue

                # Collision against each player's approximate dragon bounds
                num_players = len(players)
                for idx in range(num_players):
                    # Place players further to the left and slightly increase
                    # spacing between them for clearer separation.
                    if num_players == 1:
                        dragon_center_x = window_width * 0.25
                    else:
                        left_band = window_width * 0.15
                        right_band = window_width * 0.55
                        t = idx / float(num_players - 1)
                        dragon_center_x = left_band + t * (right_band - left_band)
                    dragon_y = map_strength_to_y(
                        interpolated_strengths[idx],
                        window_height,
                        current_dragon_img.get_height(),
                        # Same expanded vertical range as in the draw pass.
                        top_margin=int(30 * ui_scale),
                        bottom_margin=int(0 * ui_scale),
                    )
                    dragon_x = int(dragon_center_x - current_dragon_img.get_width() / 2)
                    dragon_rect = pygame.Rect(
                        dragon_x,
                        int(dragon_y),
                        current_dragon_img.get_width(),
                        current_dragon_img.get_height(),
                    )
                    if dragon_rect.colliderect(proj_rect):
                        # Avoid repeatedly hitting the same player with the same projectile
                        hit_players = proj.setdefault("hit_players", set())
                        if idx in hit_players:
                            continue

                        hit_players.add(idx)

                        if state == "playing" and hit_sound:
                            hit_sound.play()

                        # Mark hit for swirl animation and deduct score
                        hit_timers[idx] = 0.6
                        if show_score:
                            scores[idx] = max(0, scores[idx] - 5)

        # Update cloud positions (wrap around screen)
        cloud1_x -= cloud_speed_1 * dt
        cloud2_x -= cloud_speed_2 * dt

        if cloud1_x < -cloud_img.get_width():
            cloud1_x = window_width
        if cloud2_x < -cloud2_img.get_width():
            cloud2_x = window_width

        # Draw
        screen.blit(background, (0, 0))

        # Clouds
        screen.blit(cloud_img, (int(cloud1_x), int(cloud1_y)))
        screen.blit(cloud2_img, (int(cloud2_x), int(cloud2_y)))

        # Projectiles (drawn above clouds, below dragons)
        for proj in projectiles:
            screen.blit(proj["image"], proj["rect"])

        # Task type badge (top-center), color-coded
        # Use current task_type from the nonlocal variable which gets updated by stage transitions
        current_task_type = task_type.lower()
        if "relax" in current_task_type:
            task_label = "Relax"
            task_color = (59, 130, 246) # Cool Blue
        elif "focus" in current_task_type:
            task_label = "Focus"
            task_color = (249, 115, 22) # Warm Orange
        elif "training" in current_task_type:
            task_label = "Training"
            task_color = (234, 179, 8)  # yellow
        elif "live" in current_task_type:
            task_label = "Live Play"
            task_color = (34, 197, 94)  # green
        else:
            task_label = task_type
            task_color = (107, 114, 128)  # gray

        task_surf = task_font.render(task_label, True, (17, 24, 39))
        task_rect = task_surf.get_rect()
        task_rect.centerx = window_width // 2
        task_rect.y = int(10 * ui_scale)
        padded_rect = task_rect.inflate(int(32 * ui_scale), int(16 * ui_scale))
        draw_glass_panel(
            screen,
            padded_rect,
            base_color=task_color,
            radius=int(22 * ui_scale),
            fill_alpha=80,
            border_alpha=180,
        )
        screen.blit(task_surf, task_rect)
        
        # Display "SIMULATED" indicator during calibration stages
        if simulation_active:
            simulated_label = "SIMULATED"
            simulated_surf = small_font.render(simulated_label, True, (239, 68, 68))  # Red color
            simulated_rect = simulated_surf.get_rect()
            simulated_rect.centerx = window_width // 2
            simulated_rect.top = padded_rect.bottom + int(2 * ui_scale)
            simulated_panel = simulated_rect.inflate(int(16 * ui_scale), int(6 * ui_scale))
            draw_glass_panel(
                screen,
                simulated_panel,
                base_color=(254, 226, 226),  # Light red background
                radius=int(10 * ui_scale),
                fill_alpha=150,
                border_alpha=200,
            )
            screen.blit(simulated_surf, simulated_rect)
            
            # Adjust timer position to be below the simulated label
            timer_top_offset = simulated_panel.bottom + int(4 * ui_scale)
        else:
            timer_top_offset = padded_rect.bottom + int(4 * ui_scale)

        # Timer display under the task badge (or under simulated indicator if present)
        seconds_left = int(time_remaining + 0.999)  # ceil for display
        timer_text = f"Time left: {seconds_left}s" if state == "playing" else "Time's up!"
        timer_surf = small_font.render(timer_text, True, (17, 24, 39))
        timer_rect = timer_surf.get_rect()
        timer_rect.centerx = window_width // 2
        timer_rect.top = timer_top_offset
        timer_panel = timer_rect.inflate(int(20 * ui_scale), int(6 * ui_scale))
        draw_glass_panel(
            screen,
            timer_panel,
            base_color=(255, 255, 255),
            radius=int(12 * ui_scale),
            fill_alpha=130,
            border_alpha=200,
        )
        screen.blit(timer_surf, timer_rect)

        # Draw each player's dragon (with optional swirl), label, and status
        num_players = len(players)
        for idx, player in enumerate(players):
            # Use interpolated strength for smooth movement
            current_strength = interpolated_strengths[idx]
            
            # Horizontal position: spread players across a band that is
            # clearly on the left side of the screen with a bit more
            # distance between them.
            if num_players == 1:
                dragon_center_x = window_width * 0.25
            else:
                left_band = window_width * 0.15
                right_band = window_width * 0.55
                t = idx / float(num_players - 1)
                dragon_center_x = left_band + t * (right_band - left_band)
            # Allow the dragon to travel much further up and down the screen
            # by using smaller top/bottom margins in the mapping.
            dragon_y = map_strength_to_y(
                current_strength,
                window_height,
                current_dragon_img.get_height(),
                top_margin=int(30 * ui_scale),
                bottom_margin=int(0 * ui_scale),
            )

            # Base center for the sprite
            base_center_y = dragon_y + current_dragon_img.get_height() / 2
            base_center = (int(dragon_center_x), int(base_center_y))

            if hit_timers[idx] > 0:
                # Swirl animation when hit
                t = 0.6 - hit_timers[idx]
                angle = 10 * math.sin(t * 18)
                offset_y = -8 * ui_scale * abs(math.sin(t * 12))
                center = (base_center[0], int(base_center[1] + offset_y))
                sprite = pygame.transform.rotozoom(current_dragon_img, angle, 1.0)
                dragon_rect = sprite.get_rect(center=center)
            else:
                sprite = current_dragon_img
                dragon_rect = sprite.get_rect(center=base_center)

            # Dragon sprite
            screen.blit(sprite, dragon_rect.topleft)

            # Player label (e.g. "P1: Alice") above the character
            label_text = f"P{player.number}: {player.name}"
            label_surf = font.render(label_text, True, (17, 24, 39))
            label_rect = label_surf.get_rect()
            label_rect.midbottom = (
                dragon_rect.centerx,
                dragon_rect.top - int(8 * ui_scale),
            )
            label_panel = label_rect.inflate(int(24 * ui_scale), int(10 * ui_scale))
            draw_glass_panel(
                screen,
                label_panel,
                base_color=(255, 255, 255),
                radius=int(14 * ui_scale),
                fill_alpha=90,
                border_alpha=160,
            )
            screen.blit(label_surf, label_rect)

            # Wearing status is now indicated on the focus bar widget
            # to keep the dragon area visually clean.

        # Focus strength bars on the right side of the window.
        bar_margin = int(24 * ui_scale)
        bar_width = int(24 * ui_scale)
        bar_spacing = int(18 * ui_scale)
        bar_max_height = int(window_height * 0.5)
        bar_bottom = window_height - int(80 * ui_scale)

        total_bar_width = num_players * bar_width + max(0, num_players - 1) * bar_spacing
        start_x = window_width - bar_margin - total_bar_width

        for idx, player in enumerate(players):
            bar_x = start_x + idx * (bar_width + bar_spacing)
            bar_bg_rect = pygame.Rect(
                bar_x,
                bar_bottom - bar_max_height,
                bar_width,
                bar_max_height,
            )

            # Background and outline (soft, glass-like)
            draw_glass_panel(
                screen,
                bar_bg_rect,
                base_color=(148, 163, 184),
                radius=int(12 * ui_scale),
                fill_alpha=60,
                border_alpha=160,
            )

            # Subtle internal grid for a more modern bar look
            grid_surf = pygame.Surface(bar_bg_rect.size, pygame.SRCALPHA)
            grid_color = (255, 255, 255, 50)
            segment_count = 4  # creates 5 horizontal grid lines incl. top/bottom
            for i in range(segment_count + 1):
                y = int(i * bar_bg_rect.height / segment_count)
                pygame.draw.line(
                    grid_surf,
                    grid_color,
                    (0, y),
                    (bar_bg_rect.width, y),
                    width=1,
                )
            surface_rect = grid_surf.get_rect()
            screen.blit(grid_surf, bar_bg_rect.topleft)

            # Filled part representing current strength (0 bottom -> relaxed,
            # 100 top -> fully focused) with a temperature-like gradient.
            ratio = max(0.0, min(1.0, local_strengths[idx] / float(MAX_STRENGTH or 1)))
            filled_height = int(bar_max_height * ratio)
            fill_rect = pygame.Rect(
                bar_x,
                bar_bottom - filled_height,
                bar_width,
                filled_height,
            )
            fill_color = strength_to_color(local_strengths[idx])
            pygame.draw.rect(
                screen,
                fill_color,
                fill_rect,
                border_radius=int(12 * ui_scale),
            )

            # Wearing status indicator integrated into the bar widget:
            # a small red/green dot at the top of the bar column.
            status_color = (52, 199, 89) if local_wearing[idx] else (255, 59, 48)
            status_radius = max(5, int(6 * ui_scale))
            status_center = (
                bar_bg_rect.centerx,
                bar_bg_rect.top - int(10 * ui_scale),
            )
            pygame.draw.circle(screen, status_color, status_center, status_radius)
            pygame.draw.circle(
                screen,
                (255, 255, 255),
                status_center,
                status_radius,
                width=2,
            )

            # Player id label under the bar (e.g. "P1")
            bar_label = f"P{player.number}"
            bar_label_surf = small_font.render(bar_label, True, (0, 0, 0))
            bar_label_rect = bar_label_surf.get_rect()
            bar_label_rect.midtop = (
                bar_x + bar_width // 2,
                bar_bottom + int(10 * ui_scale),
            )
            screen.blit(bar_label_surf, bar_label_rect)

        # Global bar legend in its own glass panels, clearly separated
        legend_pad_x = int(10 * ui_scale)
        legend_focused_pos = (
            start_x - legend_pad_x,
            bar_bottom - bar_max_height - int(8 * ui_scale),
        )
        legend_relaxed_pos = (
            start_x - legend_pad_x,
            bar_bottom + int(12 * ui_scale),
        )

        # Legend labels use the same smooth small font for uniformity
        focused_surf = small_font.render("Focused", True, (0, 0, 0))
        focused_rect = focused_surf.get_rect()
        focused_rect.midright = legend_focused_pos
        focused_panel = focused_rect.inflate(int(16 * ui_scale), int(8 * ui_scale))
        draw_glass_panel(
            screen,
            focused_panel,
            base_color=(59, 130, 246),
            radius=int(12 * ui_scale),
            fill_alpha=70,
            border_alpha=160,
        )
        screen.blit(focused_surf, focused_rect)

        relaxed_surf = small_font.render("Relaxed", True, (0, 0, 0))
        relaxed_rect = relaxed_surf.get_rect()
        relaxed_rect.midright = legend_relaxed_pos
        relaxed_panel = relaxed_rect.inflate(int(16 * ui_scale), int(8 * ui_scale))
        draw_glass_panel(
            screen,
            relaxed_panel,
            base_color=(148, 163, 184),
            radius=int(12 * ui_scale),
            fill_alpha=60,
            border_alpha=150,
        )
        screen.blit(relaxed_surf, relaxed_rect)

        # Display per-player score and focus prominently at top-left
        if show_score:
            # Starting position for score display (top-left area)
            score_start_x = int(24 * ui_scale)
            score_start_y = int(24 * ui_scale)
            score_line_height = int(70 * ui_scale)
            
            for idx, player in enumerate(players):
                # Position for this player's score panel
                current_y = score_start_y + idx * score_line_height
                
                # Create score text with larger font
                score_text = f"P{player.number} {player.name}"
                score_value = f"Score: {scores[idx]}  |  Focus: {local_strengths[idx]}"
                
                # Render player name/number
                name_surf = font.render(score_text, True, (17, 24, 39))
                name_rect = name_surf.get_rect()
                name_rect.topleft = (score_start_x + int(20 * ui_scale), current_y + int(8 * ui_scale))
                
                # Render score/focus values with task_font (bigger)
                value_surf = task_font.render(score_value, True, (17, 24, 39))
                value_rect = value_surf.get_rect()
                value_rect.topleft = (score_start_x + int(20 * ui_scale), name_rect.bottom + int(2 * ui_scale))
                
                # Calculate panel size to fit both lines
                panel_width = max(name_rect.width, value_rect.width) + int(40 * ui_scale)
                panel_height = int(60 * ui_scale)
                panel_rect = pygame.Rect(
                    score_start_x,
                    current_y,
                    panel_width,
                    panel_height
                )
                
                # Draw glass panel background
                draw_glass_panel(
                    screen,
                    panel_rect,
                    base_color=(255, 255, 255),
                    radius=int(18 * ui_scale),
                    fill_alpha=180,
                    border_alpha=220,
                )
                
                # Draw the text
                screen.blit(name_surf, name_rect)
                screen.blit(value_surf, value_rect)

        # Training results overlay
        if state == "training_results" and game_state:
            overlay = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
            overlay.fill((15, 23, 42, 220))  # Dark overlay
            screen.blit(overlay, (0, 0))

            panel_width = int(window_width * 0.7)
            panel_height = int(window_height * 0.6)
            panel_rect = pygame.Rect(0, 0, panel_width, panel_height)
            panel_rect.center = (window_width // 2, window_height // 2)
            draw_glass_panel(
                screen,
                panel_rect,
                base_color=(255, 255, 255),
                radius=int(24 * ui_scale),
                fill_alpha=250,
                border_alpha=255,
            )

            # Title
            title_text = "Training Complete"
            title_color = (34, 197, 94)  # Green
            title_surf = task_font.render(title_text, True, title_color)
            title_rect = title_surf.get_rect()
            title_rect.centerx = panel_rect.centerx
            title_rect.top = panel_rect.top + int(30 * ui_scale)
            screen.blit(title_surf, title_rect)

            # Display results for each player
            training_results = game_state.get("training_results", [])
            line_y = title_rect.bottom + int(40 * ui_scale)
            line_spacing = int(50 * ui_scale)
            
            for idx, result in enumerate(training_results):
                player_name = result.get("name", f"Player {idx+1}")
                accuracy = result.get("accuracy")
                samples = result.get("samples", 0)
                
                # Player name
                name_surf = font.render(f"P{idx+1}: {player_name}", True, (17, 24, 39))
                name_rect = name_surf.get_rect()
                name_rect.left = panel_rect.left + int(40 * ui_scale)
                name_rect.top = line_y
                screen.blit(name_surf, name_rect)
                
                # Accuracy and samples
                if accuracy is not None:
                    acc_percent = accuracy * 100
                    acc_text = f"Accuracy: {acc_percent:.1f}%"
                    # Color based on accuracy
                    if acc_percent >= 80:
                        acc_color = (34, 197, 94)  # Green
                    elif acc_percent >= 60:
                        acc_color = (234, 179, 8)  # Yellow
                    else:
                        acc_color = (239, 68, 68)  # Red
                else:
                    acc_text = "Accuracy: N/A"
                    acc_color = (107, 114, 128)  # Gray
                
                acc_surf = font.render(acc_text, True, acc_color)
                acc_rect = acc_surf.get_rect()
                acc_rect.left = name_rect.left
                acc_rect.top = name_rect.bottom + int(5 * ui_scale)
                screen.blit(acc_surf, acc_rect)
                
                # Sample count
                samples_text = f"Samples: {samples}"
                samples_surf = small_font.render(samples_text, True, (107, 114, 128))
                samples_rect = samples_surf.get_rect()
                samples_rect.left = acc_rect.right + int(20 * ui_scale)
                samples_rect.centery = acc_rect.centery
                screen.blit(samples_surf, samples_rect)
                
                line_y += line_spacing
            
            # "Starting Live Play..." message
            footer_text = "Starting Live Play..."
            footer_surf = small_font.render(footer_text, True, (107, 114, 128))
            footer_rect = footer_surf.get_rect()
            footer_rect.centerx = panel_rect.centerx
            footer_rect.bottom = panel_rect.bottom - int(20 * ui_scale)
            screen.blit(footer_surf, footer_rect)

        # Preparation countdown overlay
        if state == "preparation":
            overlay = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
            overlay.fill((15, 23, 42, 200))  # Dark overlay
            screen.blit(overlay, (0, 0))

            panel_width = int(window_width * 0.6)
            panel_height = int(window_height * 0.5)
            panel_rect = pygame.Rect(0, 0, panel_width, panel_height)
            panel_rect.center = (window_width // 2, window_height // 2)
            draw_glass_panel(
                screen,
                panel_rect,
                base_color=(255, 255, 255),
                radius=int(24 * ui_scale),
                fill_alpha=250,
                border_alpha=255,
            )

            prep_task_lower = preparation_task_type.lower()
            if "relax" in prep_task_lower:
                title_text = "RELAX Coming Up"
                instruction_text = "Relax to make the dragon go down"
                color = (59, 130, 246)  # Cool Blue
            elif "focus" in prep_task_lower:
                title_text = "FOCUS Coming Up"
                instruction_text = "Focus to make the dragon go up"
                color = (249, 115, 22)  # Warm Orange
            else:
                title_text = "Get Ready"
                instruction_text = ""
                color = (107, 114, 128)

            # Title
            title_surf = task_font.render(title_text, True, color)
            title_rect = title_surf.get_rect()
            title_rect.centerx = panel_rect.centerx
            title_rect.top = panel_rect.top + int(40 * ui_scale)
            screen.blit(title_surf, title_rect)

            # Countdown number (large)
            countdown_seconds = int(preparation_time_remaining + 0.999)
            countdown_font_size = max(80, int(80 * ui_scale))
            countdown_font = make_font(countdown_font_size, bold=True)
            countdown_surf = countdown_font.render(str(countdown_seconds), True, color)
            countdown_rect = countdown_surf.get_rect()
            countdown_rect.center = panel_rect.center
            screen.blit(countdown_surf, countdown_rect)

            # Instruction text
            if instruction_text:
                instruction_surf = font.render(instruction_text, True, (17, 24, 39))
                instruction_rect = instruction_surf.get_rect()
                instruction_rect.centerx = panel_rect.centerx
                instruction_rect.bottom = panel_rect.bottom - int(40 * ui_scale)
                screen.blit(instruction_surf, instruction_rect)

        # Game-over overlay: show final scores, declare winner, and offer rematch options.
        if state == "game_over":
            overlay = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
            overlay.fill((15, 23, 42, 160))
            screen.blit(overlay, (0, 0))

            panel_width = int(window_width * 0.7)
            panel_height = int(window_height * 0.55)
            panel_rect = pygame.Rect(0, 0, panel_width, panel_height)
            panel_rect.center = (window_width // 2, window_height // 2)
            draw_glass_panel(
                screen,
                panel_rect,
                base_color=(255, 255, 255),
                radius=int(24 * ui_scale),
                fill_alpha=210,
                border_alpha=230,
            )

            # Determine winner(s)
            max_score = max(scores) if scores else 0
            winner_indices = [i for i, s in enumerate(scores) if s == max_score]
            is_tie = len(winner_indices) > 1

            if is_tie:
                title_text = "It's a tie!"
            else:
                winner = players[winner_indices[0]]
                title_text = f"Winner: P{winner.number} {winner.name}"

            title_surf = task_font.render(title_text, True, (17, 24, 39))
            title_rect = title_surf.get_rect()
            title_rect.centerx = panel_rect.centerx
            title_rect.top = panel_rect.top + int(24 * ui_scale)
            screen.blit(title_surf, title_rect)

            # Get training results from game_state if available
            training_results = game_state.get("training_results", []) if game_state else []
            training_accuracy_map = {r.get("name", ""): r.get("accuracy") for r in training_results}

            # Score and accuracy lines for each player, with pulse animation on winner's row
            line_y = title_rect.bottom + int(30 * ui_scale)
            line_spacing = int(45 * ui_scale)
            for idx, player in enumerate(players):
                base_color = (17, 24, 39)
                if idx in winner_indices and not is_tie:
                    base_color = (234, 179, 8)  # Warm accent for the winner

                # Player name and score
                score_text = f"P{player.number} {player.name} — Score: {scores[idx]}"
                line_surf = font.render(score_text, True, base_color)
                line_rect = line_surf.get_rect()
                line_rect.centerx = panel_rect.centerx
                line_rect.top = line_y

                if idx in winner_indices and not is_tie:
                    # Pulse the winner row slightly
                    scale = 1.0 + 0.08 * math.sin(winner_pulse_time * 4.0)
                    line_surf = pygame.transform.rotozoom(line_surf, 0, scale)
                    line_rect = line_surf.get_rect(center=line_rect.center)

                screen.blit(line_surf, line_rect)
                
                # Model accuracy below the score
                player_name = player.name
                accuracy = training_accuracy_map.get(player_name)
                if accuracy is not None:
                    acc_percent = accuracy * 100
                    acc_text = f"Model Accuracy: {acc_percent:.1f}%"
                    # Color based on accuracy
                    if acc_percent >= 80:
                        acc_color = (34, 197, 94)  # Green
                    elif acc_percent >= 60:
                        acc_color = (234, 179, 8)  # Yellow
                    else:
                        acc_color = (239, 68, 68)  # Red
                else:
                    acc_text = "Model Accuracy: N/A"
                    acc_color = (107, 114, 128)  # Gray
                
                acc_surf = small_font.render(acc_text, True, acc_color)
                acc_rect = acc_surf.get_rect()
                acc_rect.centerx = panel_rect.centerx
                acc_rect.top = line_rect.bottom + int(4 * ui_scale)
                screen.blit(acc_surf, acc_rect)
                
                line_y += line_spacing

            # Buttons for rematch and starting fresh
            button_font = font
            padding_x = int(18 * ui_scale)
            padding_y = int(8 * ui_scale)

            rematch_label = "Rematch"
            startfresh_label = "Start Fresh"

            rematch_surf = button_font.render(rematch_label, True, (17, 24, 39))
            startfresh_surf = button_font.render(startfresh_label, True, (17, 24, 39))

            button_y = panel_rect.bottom - int(40 * ui_scale)

            rematch_text_rect = rematch_surf.get_rect()
            startfresh_text_rect = startfresh_surf.get_rect()

            total_buttons_width = (
                rematch_text_rect.width
                + startfresh_text_rect.width
                + padding_x * 4
                + int(32 * ui_scale)
            )
            buttons_start_x = panel_rect.centerx - total_buttons_width // 2

            rematch_button_rect = pygame.Rect(
                0,
                0,
                rematch_text_rect.width + padding_x * 2,
                rematch_text_rect.height + padding_y * 2,
            )
            rematch_button_rect.center = (
                buttons_start_x + rematch_button_rect.width // 2,
                button_y,
            )

            startfresh_button_rect = pygame.Rect(
                0,
                0,
                startfresh_text_rect.width + padding_x * 2,
                startfresh_text_rect.height + padding_y * 2,
            )
            startfresh_button_rect.center = (
                rematch_button_rect.right
                + int(32 * ui_scale)
                + startfresh_button_rect.width // 2,
                button_y,
            )

            draw_glass_panel(
                screen,
                rematch_button_rect,
                base_color=(255, 255, 255),
                radius=int(18 * ui_scale),
                fill_alpha=230,
                border_alpha=230,
            )
            draw_glass_panel(
                screen,
                startfresh_button_rect,
                base_color=(255, 255, 255),
                radius=int(18 * ui_scale),
                fill_alpha=230,
                border_alpha=230,
            )

            rematch_text_rect.center = rematch_button_rect.center
            startfresh_text_rect.center = startfresh_button_rect.center
            screen.blit(rematch_surf, rematch_text_rect)
            screen.blit(startfresh_surf, startfresh_text_rect)

        pygame.display.flip()

    if mixer_available and music_started:
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass

    pygame.quit()


def main() -> None:
    asyncio.run(run_game_loop())


if __name__ == "__main__":  # pragma: no cover - direct run
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        pygame.quit()
        raise SystemExit(exc) from exc
