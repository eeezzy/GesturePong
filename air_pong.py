import sys, time
import random
import cv2
import numpy as np
import pygame
import mediapipe as mp

# ==== Game area size ====
W, H = 900, 600

# ==== Right sidebar ====
SIDEBAR_W = 360
SB_BG = (12, 12, 16)
SB_FG = (220, 220, 220)

# ==== Camera preview ====
preview_w, preview_h = 320, 240
preview_margin = 16

# ==== Central ROI for input (camera horizontal ratio) ====
ROI_LEFT  = 0.40
ROI_RIGHT = 0.60
SHOW_LANDMARKS = True
SHOW_ROI_GUIDE = True

# ==== Mode settings ====
# AR mode: overlay the game on the live camera frame
AR_MODE = True

# ==== Wait for macOS camera permission ====
def open_camera_with_permission(indices=(0,1,2,3),
                                backend=cv2.CAP_AVFOUNDATION,
                                warmup_reads=2,
                                wait_seconds=15,
                                poll_interval=0.5):
    deadline = time.time() + wait_seconds
    printed_hint = False
    while time.time() < deadline:
        for idx in indices:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ok = False
                for _ in range(warmup_reads):
                    ok, _ = cap.read()
                    if not ok: break
                    time.sleep(0.05)
                if ok: return cap, idx, backend
                cap.release()
        if sys.platform == "darwin" and not printed_hint:
            print("📷 Waiting for macOS camera permission… (팝업이 뜨면 '허용')")
            printed_hint = True
        time.sleep(poll_interval)
    raise RuntimeError("웹캠을 열 수 없습니다.")

# ==== MediaPipe Hands & Drawing ====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=0
)
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# ==== Static gesture classification ====
def finger_extended(lm, tip, pip, wrist=0):
    tip_v = np.array([lm[tip].x - lm[wrist].x, lm[tip].y - lm[wrist].y])
    pip_v = np.array([lm[pip].x - lm[wrist].x, lm[pip].y - lm[wrist].y])
    return np.linalg.norm(tip_v) > np.linalg.norm(pip_v)

def classify_static_gesture(lm):
    if lm is None: return None
    F = {"index":(8,6,5), "middle":(12,10,9), "ring":(16,14,13), "pinky":(20,18,17)}
    ext = {n: finger_extended(lm, tip, pip) for n,(tip,pip,_) in F.items()}
    num_ext = sum(int(v) for v in ext.values())
    peace = ext["index"] and ext["middle"] and (not ext["ring"]) and (not ext["pinky"])
    if num_ext == 0: return "FIST"
    if num_ext == 4: return "PALM"
    if peace:        return "PEACE"
    return "OTHER"

def hand_center(lm):
    if lm is None: return None, None
    xs = [p.x for p in lm]; ys = [p.y for p in lm]
    return float(np.clip(np.mean(xs), 0, 1)), float(np.clip(np.mean(ys), 0, 1))


# ==== Score-based speed increase ====
STEP = 0.15
MAX_SPEED = 2.2
def speed_from_score(score: int) -> float:
    if score < 10: return 1.0
    bumps = 1 + (score - 10) // 5
    return min(MAX_SPEED, 1.0 + STEP * bumps)

# ==== Game elements ====
PADDLE_W, PADDLE_H = 130, 16
BALL_R = 9

THEMES = [
    ((20,20,30), (240,240,240)),
    ((15,25,45), (255,180,0)),
    ((30,10,35), (220,100,240)),
    ((5,30,25),  (120,220,210)),
]

def main():
    cap, _, _ = open_camera_with_permission()

    pygame.init()
    screen = pygame.display.set_mode((W + SIDEBAR_W, H))
    pygame.display.set_caption("Air-Pong + Side Camera")
    clock = pygame.time.Clock()

    theme_idx = 0
    bg, fg = THEMES[theme_idx]

    running = True
    paused = False
    score = 0

    # Power-up state
    active_effects = []  
    pickups = []      
    next_pickup_time = time.time() + 7.0
    fist_cooldown_until = 0.0

    paddle_x = (W - PADDLE_W) // 2
    paddle_y = H - 40
    ball_x, ball_y = W//2, H//2
    ball_vx, ball_vy = 320, -280

    last_label, label_time = None, 0.0
    debounce = 0.32

    # Latches: PALM (pause/resume), PEACE (theme). FIST/OTHER do nothing
    peace_latched = False
    palm_latched  = False

    font = pygame.font.SysFont("Arial", 22)
    small = pygame.font.SysFont("Arial", 16, bold=True)
    tiny  = pygame.font.SysFont("Arial", 12)

    # Sidebar layout
    sb_x0 = W
    cam_x = sb_x0 + preview_margin
    cam_y = preview_margin

    try:
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_a:
                        # Toggle AR mode
                        global AR_MODE
                        AR_MODE = not AR_MODE

            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            # Camera processing
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(frame_rgb)
            lm = res.multi_hand_landmarks[0].landmark if res.multi_hand_landmarks else None

            # Overlay (landmarks/ROI)
            overlay_rgb = frame_rgb.copy()
            if SHOW_LANDMARKS and res.multi_hand_landmarks:
                for hand_landmarks in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        overlay_rgb, hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
            if SHOW_ROI_GUIDE:
                h, w, _ = overlay_rgb.shape
                xL = int(ROI_LEFT * w); xR = int(ROI_RIGHT * w)
                cv2.line(overlay_rgb, (xL, 0), (xL, h), (0, 255, 0), 2)
                cv2.line(overlay_rgb, (xR, 0), (xR, h), (0, 255, 0), 2)

            # MediaPipe gesture classification + debounce
            raw_label = classify_static_gesture(lm)
            now = time.time()
            label = None
            if raw_label == last_label:
                if now - label_time >= debounce:
                    label = raw_label
            else:
                last_label = raw_label
                label_time = now

            # Trigger/release (only PALM/PEACE latch)
            peace_trigger = (label == "PEACE" and not peace_latched)
            palm_trigger  = (label == "PALM"  and not palm_latched)
            if raw_label != "PEACE": peace_latched = False
            if raw_label != "PALM":  palm_latched  = False

            # Gesture -> Action
            if palm_trigger:           # PALM -> Pause/Resume
                paused = not paused
                palm_latched = True
            elif peace_trigger:        # PEACE -> Change theme
                theme_idx = (theme_idx + 1) % len(THEMES)
                bg, fg = THEMES[theme_idx]
                peace_latched = True
            elif label == "FIST":     # FIST -> Immediate slow motion (cooldown)
                now_ts = time.time()
                if now_ts >= fist_cooldown_until:
                    active_effects.append(("SLOW", now_ts + 2.5))
                    fist_cooldown_until = now_ts + 6.0
            # FIST/OTHER -> Do nothing (keep game running)

            # Paddle movement (central ROI)
            cx, _ = hand_center(lm)
            if cx is not None:
                cx_clamped = float(np.clip(cx, ROI_LEFT, ROI_RIGHT))
                cx_norm = (cx_clamped - ROI_LEFT) / max(1e-6, (ROI_RIGHT - ROI_LEFT))
                target = int(cx_norm * (W - PADDLE_W))
                paddle_x = int(0.25 * target + 0.75 * paddle_x)

            # Update/apply effects
            now_ts = time.time()
            active_effects = [(t, until) for (t, until) in active_effects if until > now_ts]
            paddle_scale = 1.0
            ball_speed_mult = 1.0
            for t, _until in active_effects:
                if t == "ENLARGE":
                    paddle_scale = max(paddle_scale, 1.6)
                elif t == "SLOW":
                    ball_speed_mult = min(ball_speed_mult, 0.65)

            # Speed: score-based × effects
            speed_scale = speed_from_score(score) * ball_speed_mult

            # Physics
            dt = clock.get_time() / 1000.0
            if not paused:
                ball_x += ball_vx * dt * speed_scale
                ball_y += ball_vy * dt * speed_scale
                if ball_x < BALL_R: ball_x, ball_vx = BALL_R, abs(ball_vx)
                if ball_x > W - BALL_R: ball_x, ball_vx = W - BALL_R, -abs(ball_vx)
                if ball_y < BALL_R: ball_y, ball_vy = BALL_R, abs(ball_vy)

                # Current paddle width (effects applied)
                current_paddle_w = int(PADDLE_W * paddle_scale)
                paddle_x = int(np.clip(paddle_x, 0, W - current_paddle_w))

                if (paddle_y - BALL_R <= ball_y <= paddle_y + PADDLE_H) and \
                   (paddle_x <= ball_x <= paddle_x + current_paddle_w) and ball_vy > 0:
                    ball_y = paddle_y - BALL_R
                    ball_vy = -abs(ball_vy) * 1.03
                    score += 1

                # Pickup spawn (time-based)
                if now_ts >= next_pickup_time:
                    pickups.append({
                        'x': random.randint(40, W-40),
                        'y': -20,
                        'type': random.choice(["ENLARGE", "SLOW"]),
                    })
                    next_pickup_time = now_ts + random.uniform(6.0, 10.0)

                # Pickup falling and collision
                new_pickups = []
                for p in pickups:
                    p['y'] += 120 * dt
                    if p['y'] > H + 30:
                        continue
                    # Collision check
                    if (paddle_y - 10 <= p['y'] <= paddle_y + PADDLE_H) and \
                       (paddle_x <= p['x'] <= paddle_x + current_paddle_w):
                        dur = 5.0 if p['type'] == "ENLARGE" else 3.0
                        active_effects.append((p['type'], time.time() + dur))
                        continue
                    new_pickups.append(p)
                pickups = new_pickups

                if ball_y > H + 40:
                    paused = True

            # ==== Rendering ====
            # Game background or camera background
            if AR_MODE:
                cam_bg = cv2.resize(overlay_rgb, (W, H), interpolation=cv2.INTER_AREA)
                cam_surf = pygame.image.frombuffer(cam_bg.tobytes(), (W, H), 'RGB')
                screen.blit(cam_surf, (0, 0))
            else:
                screen.fill(bg, rect=pygame.Rect(0, 0, W, H))
            screen.fill(SB_BG, rect=pygame.Rect(W, 0, SIDEBAR_W, H))
            pygame.draw.line(screen, SB_FG, (W, 0), (W, H), width=2)

            # Game elements
            current_paddle_w = int(PADDLE_W * (1.6 if any(t=="ENLARGE" for t,_ in active_effects) else 1.0))
            pygame.draw.rect(screen, fg, (paddle_x, paddle_y, current_paddle_w, PADDLE_H), border_radius=8)
            pygame.draw.circle(screen, fg, (int(ball_x), int(ball_y)), BALL_R)

            # Pickup rendering
            for p in pickups:
                color = (120, 200, 120) if p['type'] == "ENLARGE" else (120, 160, 240)
                pygame.draw.circle(screen, color, (int(p['x']), int(p['y'])), 10)

            # Camera preview (sidebar)
            preview = cv2.resize(overlay_rgb, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
            surf = pygame.image.frombuffer(preview.tobytes(), (preview_w, preview_h), 'RGB')
            screen.blit(surf, (cam_x, cam_y))
            pygame.draw.rect(screen, SB_FG, (cam_x-1, cam_y-1, preview_w+2, preview_h+2), width=2, border_radius=6)
            title = small.render("Camera", True, SB_FG)
            screen.blit(title, (cam_x, cam_y + preview_h + 8))

            # HUD
            eff_txt = ",".join(t for t,_ in active_effects) or "-"
            info = f"Score: {score}   {'PAUSED' if paused else ''}   Gesture: {raw_label or ''}   Speed x{speed_scale:.2f}   FX: {eff_txt}"
            text = font.render(info, True, fg)
            screen.blit(text, (16, 12))

            # Sidebar help
            help_lines = [
                "Gestures:",
                "PALM = Pause/Resume",
                "PEACE = Theme",
                "FIST = SlowMo (CD)",
                "",
                "Keys:",
                "A = Toggle AR Mode",
            ]
            yoff = cam_y + preview_h + 32
            for line in help_lines:
                txt = tiny.render(line, True, SB_FG)
                screen.blit(txt, (cam_x, yoff))
                yoff += 16

            pygame.display.flip()
            clock.tick(60)
    finally:
        cap.release()
        pygame.quit()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()