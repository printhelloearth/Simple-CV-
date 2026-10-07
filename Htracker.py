import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import urllib.request

model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        model_path
    )
    print("Model downloaded.")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = vision.HandLandmarker.create_from_options(options)

grid_size = 8
voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
active_layer = grid_size // 2
brush_size = 1
camera_pitch = 30
camera_yaw = 45

last_fist_time = 0.0 
fist_cooldown = 0.4
last_swipe_time = 0.0
swipe_cooldown = 0.6
last_pinch = False
last_two_hand_center = None
hand_history = []

cube_mode = False
cube_position = [grid_size // 2, grid_size // 2, grid_size // 2]
cube_size = 2
cube_grabbed = False
cube_rotation = [0.0, 0.0]

print("Hand-Tracked Voxel Builder")
print("Press 'c' to toggle Cube Manipulation mode")
print("Open hand: move cursor / move cube")
print("Closed fist: place voxel / grab cube")
print("Pinch: move cube while maintaining size")
print("Two hands open: rotate camera")
print("Swipe: select/delete layer / change cube depth")
print("Press 'q' to quit and view the voxel structure.")

cap = cv2.VideoCapture(0)

finger_tips = [4, 8, 12, 16, 20]
finger_mcps = [2, 5, 9, 13, 17]


def hand_distance(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def is_hand_open(hand):
    open_count = 0
    for tip, mcp in zip(finger_tips, finger_mcps):
        if hand[tip].y < hand[mcp].y:
            open_count += 1
    palm_width = abs(hand[5].x - hand[17].x)
    return open_count >= 4 and palm_width > 0.12

def is_hand_fist(hand):
    close_count = 0
    for tip, mcp in zip(finger_tips, finger_mcps):
        if hand[tip].y > hand[mcp].y:
            close_count += 1
    return close_count >= 4

def is_hand_pinch(hand):
    thumb = hand[4]
    index = hand[8]
    base = hand[0]
    hand_size = hand_distance(base, hand[9])
    if hand_size <= 0:
        return False
    return hand_distance(thumb, index) < max(0.06, hand_size * 0.18)

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def place_voxel(grid_x, grid_y, grid_z, size):
    half = size // 2
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            for dz in range(-half, half + 1):
                x = clamp(grid_x + dx, 0, grid_size - 1)
                y = clamp(grid_y + dy, 0, grid_size - 1)
                z = clamp(grid_z + dz, 0, grid_size - 1)
                voxel_grid[x, y, z] = True

def draw_text(frame, text, position, color=(255, 255, 255)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def draw_cube_cursor(frame, center, size, color=(0, 255, 255), depth_scale=8):
    cx, cy = center
    half = int(size * 12)
    depth = int(size * depth_scale)

    front_tl = (cx - half, cy - half)
    front_tr = (cx + half, cy - half)
    front_bl = (cx - half, cy + half)
    front_br = (cx + half, cy + half)

    back_tl = (front_tl[0] + depth, front_tl[1] - depth)
    back_tr = (front_tr[0] + depth, front_tr[1] - depth)
    back_bl = (front_bl[0] + depth, front_bl[1] - depth)
    back_br = (front_br[0] + depth, front_br[1] - depth)

    pts = [front_tl, front_tr, front_br, front_bl]
    cv2.polylines(frame, [np.array(pts)], True, color, 2)
    pts_back = [back_tl, back_tr, back_br, back_bl]
    cv2.polylines(frame, [np.array(pts_back)], True, color, 2)
    for p1, p2 in zip(pts, pts_back):
        cv2.line(frame, p1, p2, color, 2)

def draw_cube_object(frame, center, z, size, color=(0, 255, 0)):
    screen_depth = max(2, int(size * 8 + (z - grid_size / 2) * 2))
    draw_cube_cursor(frame, center, size, color, depth_scale=max(1, screen_depth // max(size, 1)))

def map_cube_to_screen(position, frame_w, frame_h):
    x, y, z = position
    screen_x = int(np.clip(x / (grid_size - 1), 0, 1) * frame_w)
    screen_y = int(np.clip(y / (grid_size - 1), 0, 1) * frame_h)
    return screen_x, screen_y

def is_near_cube(hand, cube_center, threshold=0.15):
    palm_x = hand[0].x
    palm_y = hand[0].y
    dx = palm_x - cube_center[0]
    dy = palm_y - cube_center[1]
    return np.sqrt(dx * dx + dy * dy) < threshold

def detect_swipe(history):
    if len(history) < 2:
        return None
    old_time, old_x = history[0]
    new_time, new_x = history[-1]
    dt = new_time - old_time
    if dt <= 0:
        return None
    dx = new_x - old_x
    if abs(dx) > 0.25 and dt < 0.6:
        return 'right' if dx > 0 else 'left'
    return None

def map_to_grid(x, y):
    grid_x = int(np.clip(x * grid_size, 0, grid_size - 1))
    grid_y = int(np.clip(y * grid_size, 0, grid_size - 1))
    return grid_x, grid_y

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = landmarker.detect(mp_image)

    frame_h, frame_w = frame.shape[:2]
    mode_text = "No hands detected"
    swipe_text = ""
    camera_text = ""
    current_cursor = None

    hands = results.hand_landmarks if results.hand_landmarks else []
    open_hands = [hand for hand in hands if is_hand_open(hand)]
    fist_hands = [hand for hand in hands if is_hand_fist(hand)]
    pinch_hands = [hand for hand in hands if is_hand_pinch(hand)]

    cube_center_screen = map_cube_to_screen(cube_position, frame_w, frame_h)
    draw_cube_object(frame, cube_center_screen, cube_position[2], cube_size, (0, 255, 0))
    draw_text(frame, f"Cube: X={cube_position[0]} Y={cube_position[1]} Z={cube_position[2]} Size={cube_size}", (20, 30), (0, 255, 0))
    if len(open_hands) == 2:
        center_x = (open_hands[0][0].x + open_hands[1][0].x) / 2.0
        center_y = (open_hands[0][0].y + open_hands[1][0].y) / 2.0
        if last_two_hand_center is not None:
            dx = center_x - last_two_hand_center[0]
            dy = center_y - last_two_hand_center[1]
            camera_yaw += dx * 180
            camera_pitch += -dy * 160
            camera_pitch = clamp(camera_pitch, -85, 85)
            camera_text = f"Camera rotate: yaw={int(camera_yaw)%360}, pitch={int(camera_pitch)}"
            if cube_mode:
                cube_rotation[0] += dx * 4
                cube_rotation[1] += dy * 4
        last_two_hand_center = (center_x, center_y)
        mode_text = "Two hands open: rotate camera"
        cv2.circle(frame, (int(center_x * frame_w), int(center_y * frame_h)), 12, (255, 255, 0), -1)
    else:
        last_two_hand_center = None

    if len(open_hands) >= 1 and len(open_hands) < 2:
        hand = open_hands[0]
        palm_x = hand[0].x
        palm_y = hand[0].y
        grid_x, grid_y = map_to_grid(palm_x, palm_y)
        current_cursor = (grid_x, grid_y, active_layer)
        center = (int(palm_x * frame_w), int(palm_y * frame_h))
        hand_history.append((time.time(), palm_x))
        if len(hand_history) > 10:
            hand_history.pop(0)

        if cube_mode:
            if cube_grabbed:
                cube_position[0] = grid_x
                cube_position[1] = grid_y
                mode_text = "Grabbed cube: move it with your hand"
            elif is_near_cube(hand, (palm_x, palm_y)):
                cube_grabbed = True
                mode_text = "Fist or near cube: grab cube"
            elif is_hand_fist(hand): 
                cube_grabbed = False 
                mode_text = "No Action: move hand away from cube or open hand to grab"
            else:
                draw_cube_cursor(frame, center, brush_size, (0, 255, 255))
                mode_text = "Open hand: move cursor"
            swipe_direction = detect_swipe(hand_history)
            if swipe_direction and time.time() - last_swipe_time > swipe_cooldown:
                if swipe_direction == 'right':
                    cube_position[2] = clamp(cube_position[2] + 1, 0, grid_size - 1)
                    swipe_text = f"Swipe right: cube depth {cube_position[2]}"
                else:
                    cube_position[2] = clamp(cube_position[2] - 1, 0, grid_size - 1)
                    swipe_text = f"Swipe left: cube depth {cube_position[2]}"
                last_swipe_time = time.time()
                hand_history.clear()
        else:
            draw_cube_cursor(frame, center, brush_size, (0, 255, 255))
            draw_text(frame, "Open hand: move cursor", (20, 60), (0, 255, 255))
            draw_text(frame, f"Cursor: X={grid_x} Y={grid_y} Z={active_layer}", (20, 90), (0, 255, 255))
            swipe_direction = detect_swipe(hand_history)
            if swipe_direction and time.time() - last_swipe_time > swipe_cooldown:
                if swipe_direction == 'right':
                    active_layer = (active_layer + 1) % grid_size
                    swipe_text = f"Swipe right: selected layer {active_layer}"
                else:
                    voxel_grid[:, :, active_layer] = False
                    swipe_text = f"Swipe left: deleted layer {active_layer}"
                last_swipe_time = time.time()
                hand_history.clear()

    elif len(open_hands) == 0:
        hand_history.clear()

    if cube_mode:
        if len(fist_hands) >= 1:
            fist_hand = fist_hands[0]
            palm_x = fist_hand[0].x
            palm_y = fist_hand[0].y
            grid_x, grid_y = map_to_grid(palm_x, palm_y)
            if is_near_cube(fist_hand, (cube_position[0] / (grid_size - 1), cube_position[1] / (grid_size - 1))):
                cube_grabbed = True
                cube_position[0] = grid_x
                cube_position[1] = grid_y
                draw_text(frame, "Fist: grabbing cube", (20, 90), (0, 255, 0))
        else:
            cube_grabbed = False
    else:
        if len(fist_hands) >= 1:
            if current_cursor is None and len(fist_hands) > 0:
                hand = fist_hands[0]
                palm_x = hand[0].x
                palm_y = hand[0].y
                grid_x, grid_y = map_to_grid(palm_x, palm_y)
                current_cursor = (grid_x, grid_y, active_layer)
            if current_cursor is not None and time.time() - last_fist_time > fist_cooldown:
                place_voxel(*current_cursor, brush_size)
                last_fist_time = time.time()
                draw_text(frame, "Placed voxel", (20, 90), (0, 255, 0))

    if len(pinch_hands) >= 1:
        pinch_hand = pinch_hands[0]
        thumb = pinch_hand[4]
        index = pinch_hand[8]
        pinch_dist = hand_distance(thumb, index)
        reference = max(hand_distance(pinch_hand[0], pinch_hand[9]), 0.01)
        new_brush_size = int(np.clip(pinch_dist / reference * 8, 1, 4))
        if cube_mode:
            palm_x = pinch_hand[0].x
            palm_y = pinch_hand[0].y
            grid_x, grid_y = map_to_grid(palm_x, palm_y)
            cube_position[0] = grid_x
            cube_position[1] = grid_y
            draw_text(frame, f"Pinch move cube: X={cube_position[0]} Y={cube_position[1]}", (20, 120), (0, 255, 0))
        else:
            if new_brush_size != brush_size:
                brush_size = new_brush_size
            draw_text(frame, f"Pinch resize: {brush_size}x{brush_size}x{brush_size}", (20, 120), (0, 200, 255))
        last_pinch = True
    else:
        last_pinch = False

    if swipe_text:
        draw_text(frame, swipe_text, (20, 150), (255, 200, 0))
    if camera_text:
        draw_text(frame, camera_text, (20, 180), (255, 200, 0))

    draw_text(frame, f"Active layer: {active_layer}", (20, frame_h - 70), (200, 200, 200))
    draw_text(frame, f"Brush size: {brush_size}", (20, frame_h - 40), (200, 200, 200))
    draw_text(frame, f"Mode: {'Cube Manipulation' if cube_mode else 'Voxel Builder'}", (20, frame_h - 100), (255, 255, 0))

    cv2.imshow('Hand-Tracked Voxel Builder', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        cube_mode = not cube_mode
        cube_grabbed = False
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
landmarker.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(voxel_grid, edgecolor='k')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Built Voxel Structure (Layer {active_layer}, Brush {brush_size})')
ax.view_init(camera_pitch, camera_yaw)
plt.show()

