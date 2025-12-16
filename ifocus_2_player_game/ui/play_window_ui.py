import sys
from pathlib import Path
import math
import random

import pygame

import play_window_config


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


def main() -> None:
    pygame.init()
    pygame.display.set_caption("iFocus Play Window")

    # Derive a UI scale based on the current screen, similar in
    # spirit to IFocusWindow in ifocus_ui.py, but also ensuring
    # the window never exceeds the physical screen size.
    display_info = pygame.display.Info()
    screen_w, screen_h = display_info.current_w, display_info.current_h

    base_scale = max(0.85, min(1.45, min(screen_w / 1440.0, screen_h / 900.0)))
    fit_scale = min(screen_w / BASE_WIDTH, screen_h / BASE_HEIGHT)
    ui_scale = min(base_scale, fit_scale)
    if ui_scale <= 0:
        ui_scale = 0.5

    window_width = int(BASE_WIDTH * ui_scale)
    window_height = int(BASE_HEIGHT * ui_scale)

    screen = pygame.display.set_mode((window_width, window_height))
    clock = pygame.time.Clock()

    # Session / player configuration (isolated from UI logic)
    session_cfg = play_window_config.get_default_session_config()
    players = session_cfg.players
    if not players:
        raise SystemExit("play_window_config must define at least one player.")

    # Load images
    background = load_image("skyblue.png")
    cloud_img = load_image("cloud.png")
    cloud2_img = load_image("cloud2.png")
    dragon_img = load_image("dragondown.png")

    # Optional projectile / obstacle sprites
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
    dragon_h_target = int(window_height * 0.22)
    d_w, d_h = dragon_img.get_size()
    if d_h > 0:
        d_scale = dragon_h_target / float(d_h)
        dragon_size = (int(d_w * d_scale), int(d_h * d_scale))
        dragon_img = pygame.transform.smoothscale(dragon_img, dragon_size)

    # Projectile configuration (slow, readable motion)
    projectile_types = []
    # Make base speed quite slow so movement is gentle
    base_speed = 80 * ui_scale

    # Scale projectiles so they sit nicely in the scene.
    # Slightly smaller so they feel less punishing.
    proj_target_w = int(window_width * 0.07)

    if fireball_img is not None:
        fireball_img = scale_to_width(fireball_img, proj_target_w)
        projectile_types.append(
            {"name": "fireball", "image": fireball_img, "speed": base_speed * 0.7}
        )
    if arrow_img is not None:
        arrow_img = scale_to_width(arrow_img, proj_target_w)
        projectile_types.append(
            {"name": "arrow", "image": arrow_img, "speed": base_speed * 0.8}
        )
    if javelin_img is not None:
        javelin_img = scale_to_width(javelin_img, proj_target_w)
        projectile_types.append(
            {"name": "javelin", "image": javelin_img, "speed": base_speed * 0.6}
        )
    if obstacle_img is not None:
        obstacle_img = scale_to_width(obstacle_img, proj_target_w)
        # Flip obstacles so they face toward the players
        obstacle_img = pygame.transform.flip(obstacle_img, True, False)
        projectile_types.append(
            {"name": "obstacle", "image": obstacle_img, "speed": base_speed * 0.5}
        )

    # Simple cloud layers for parallax-like motion
    cloud_speed_1 = 40  # pixels / second
    cloud_speed_2 = 80

    cloud1_x = 0.0
    cloud2_x = window_width * 0.5

    cloud1_y = window_height * 0.18
    cloud2_y = window_height * 0.32

    # Character strengths and scores (one per configured player)
    strengths = [50 for _ in players]
    scores = [0 for _ in players]
    hit_timers = [0.0 for _ in players]

    # Round timer (seconds)
    round_duration = max(1, getattr(session_cfg, "duration_seconds", 60))
    time_remaining = float(round_duration)
    second_accumulator = 0.0

    # Projectiles
    projectiles: list[dict] = []  # each: {image, rect, x, y, vx, vy, age, swirl_amp, swirl_freq, swirl_phase, name}
    projectile_spawn_accumulator = 0.0
    # Spawn very slowly so players are nudged to move,
    # but not overwhelmed.
    projectile_spawn_interval = 8.0  # seconds
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
    state = "playing"  # or "game_over"
    winner_pulse_time = 0.0
    rematch_button_rect = None
    startfresh_button_rect = None
    while running:
        dt = clock.tick(FPS) / 1000.0  # seconds since last frame

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP and state == "playing":
                    strengths = [
                        min(MAX_STRENGTH, s + STRENGTH_STEP) for s in strengths
                    ]
                elif event.key == pygame.K_DOWN and state == "playing":
                    strengths = [
                        max(MIN_STRENGTH, s - STRENGTH_STEP) for s in strengths
                    ]
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if state == "game_over":
                    if rematch_button_rect and rematch_button_rect.collidepoint(event.pos):
                        # Reset round state for a rematch with the same players
                        strengths = [50 for _ in players]
                        scores = [0 for _ in players]
                        hit_timers = [0.0 for _ in players]
                        time_remaining = float(round_duration)
                        second_accumulator = 0.0
                        projectiles.clear()
                        projectile_spawn_accumulator = 0.0
                        last_projectile_target_idx = None
                        winner_pulse_time = 0.0
                        state = "playing"
                    elif startfresh_button_rect and startfresh_button_rect.collidepoint(event.pos):
                        # Exit this play window; outer UI can start a new
                        # session or configuration as needed.
                        running = False

        # Timer and scoring: each full second that passes
        # awards +1 score point to every player while time remains.
        if state == "playing" and time_remaining > 0:
            time_remaining -= dt
            second_accumulator += dt
            while second_accumulator >= 1.0 and time_remaining > 0:
                second_accumulator -= 1.0
                for i in range(len(scores)):
                    scores[i] += 1

            if time_remaining <= 0:
                time_remaining = 0
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
        if projectile_types and state == "playing":
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
                    dragon_y_for_target = map_strength_to_y(
                        strengths[target_idx],
                        window_height,
                        dragon_img.get_height(),
                        # Give players more room to move vertically
                        top_margin=int(80 * ui_scale),
                        bottom_margin=int(4 * ui_scale),
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

                    vertical_step = int(dragon_img.get_height() * 0.2)

                    for offset_index in pattern:
                        spawn_y = int(
                            dragon_y_for_target
                            + dragon_img.get_height() * 0.35
                            + offset_index * vertical_step
                        )

                        # Start just off the right edge
                        spawn_x = window_width + img.get_width()

                        # Aim roughly towards the chosen target point with
                        # slight random vertical offset to vary the arc.
                        aim_y = dragon_y_for_target + random.uniform(
                            -dragon_img.get_height() * 0.2,
                            dragon_img.get_height() * 0.2,
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
                        strengths[idx],
                        window_height,
                        dragon_img.get_height(),
                        # Same expanded vertical range as in the draw pass
                        top_margin=int(80 * ui_scale),
                        bottom_margin=int(4 * ui_scale),
                    )
                    dragon_x = int(dragon_center_x - dragon_img.get_width() / 2)
                    dragon_rect = pygame.Rect(
                        dragon_x,
                        int(dragon_y),
                        dragon_img.get_width(),
                        dragon_img.get_height(),
                    )
                    if dragon_rect.colliderect(proj_rect):
                        # Avoid repeatedly hitting the same player with the same projectile
                        hit_players = proj.setdefault("hit_players", set())
                        if idx in hit_players:
                            continue

                        hit_players.add(idx)

                        # Mark hit for swirl animation and deduct score
                        hit_timers[idx] = 0.6
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
        task_type = session_cfg.task_type.lower()
        if task_type == "training":
            task_label = "Training"
            task_color = (234, 179, 8)  # yellow
        elif task_type == "live":
            task_label = "Live Play"
            task_color = (34, 197, 94)  # green
        else:
            task_label = session_cfg.task_type
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

        # Timer display under the task badge
        seconds_left = int(time_remaining + 0.999)  # ceil for display
        timer_text = f"Time left: {seconds_left}s" if state == "playing" else "Time's up!"
        timer_surf = small_font.render(timer_text, True, (17, 24, 39))
        timer_rect = timer_surf.get_rect()
        timer_rect.centerx = window_width // 2
        timer_rect.top = padded_rect.bottom + int(4 * ui_scale)
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
                strengths[idx],
                window_height,
                dragon_img.get_height(),
                top_margin=int(80 * ui_scale),
                bottom_margin=int(4 * ui_scale),
            )

            # Base center for the sprite
            base_center_y = dragon_y + dragon_img.get_height() / 2
            base_center = (int(dragon_center_x), int(base_center_y))

            if hit_timers[idx] > 0:
                # Swirl animation when hit
                t = 0.6 - hit_timers[idx]
                angle = 10 * math.sin(t * 18)
                offset_y = -8 * ui_scale * abs(math.sin(t * 12))
                center = (base_center[0], int(base_center[1] + offset_y))
                sprite = pygame.transform.rotozoom(dragon_img, angle, 1.0)
                dragon_rect = sprite.get_rect(center=center)
            else:
                sprite = dragon_img
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
            ratio = max(0.0, min(1.0, strengths[idx] / float(MAX_STRENGTH or 1)))
            filled_height = int(bar_max_height * ratio)
            fill_rect = pygame.Rect(
                bar_x,
                bar_bottom - filled_height,
                bar_width,
                filled_height,
            )
            fill_color = strength_to_color(strengths[idx])
            pygame.draw.rect(
                screen,
                fill_color,
                fill_rect,
                border_radius=int(12 * ui_scale),
            )

            # Wearing status indicator integrated into the bar widget:
            # a small red/green dot at the top of the bar column.
            status_color = (52, 199, 89) if player.wearing else (255, 59, 48)
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

        # Overlay per-player strength/score summary at the bottom in its own glass panel,
        # clearly indicating which values belong to which player.
        summary_parts = []
        for idx, player in enumerate(players):
            summary_parts.append(
                f"P{player.number} ({player.name}) - Focus: {strengths[idx]}  Score: {scores[idx]}"
            )
        # Use a simple bullet separator between players
        summary_text = "   •   ".join(summary_parts)
        summary_surf = small_font.render(summary_text, True, (17, 24, 39))
        summary_rect = summary_surf.get_rect()
        summary_rect.midbottom = (
            window_width // 2,
            window_height - int(12 * ui_scale),
        )
        summary_panel = summary_rect.inflate(int(32 * ui_scale), int(10 * ui_scale))
        draw_glass_panel(
            screen,
            summary_panel,
            base_color=(255, 255, 255),
            radius=int(16 * ui_scale),
            fill_alpha=150,
            border_alpha=200,
        )
        screen.blit(summary_surf, summary_rect)

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

            # Score lines for each player, with a subtle pulse animation
            # on the winner's row.
            line_y = title_rect.bottom + int(24 * ui_scale)
            line_spacing = int(28 * ui_scale)
            for idx, player in enumerate(players):
                base_color = (17, 24, 39)
                if idx in winner_indices and not is_tie:
                    # Warm accent for the winner
                    base_color = (234, 179, 8)

                score_text = f"P{player.number} {player.name} — Score: {scores[idx]}"
                line_surf = font.render(score_text, True, base_color)
                line_rect = line_surf.get_rect()
                line_rect.centerx = panel_rect.centerx
                line_rect.top = line_y

                if idx in winner_indices and not is_tie:
                    # Pulse the winner row slightly for a simple animation effect.
                    scale = 1.0 + 0.08 * math.sin(winner_pulse_time * 4.0)
                    line_surf = pygame.transform.rotozoom(line_surf, 0, scale)
                    line_rect = line_surf.get_rect(center=line_rect.center)

                screen.blit(line_surf, line_rect)
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

    pygame.quit()


if __name__ == "__main__":  # pragma: no cover - direct run
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        pygame.quit()
        raise SystemExit(exc) from exc
