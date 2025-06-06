# -----------------------------------
# ------------- IMPORTS -------------
# -----------------------------------

from ultralytics import YOLO
import cv2
import math
import os
import numpy as np
from collections import deque
from functools import reduce
from sys import argv
from PIL import Image, ImageDraw, ImageFont

# -----------------------------------
# ------- HELPER FUNCTIONS ----------
# -----------------------------------

def image_to_court(x, y, H): # for right lock videos
    pt = np.array([x, y, 1.0])
    mapped = H @ pt
    mapped /= mapped[2]
    return float(mapped[0]), float(mapped[1])  # court_x, court_y

def is_in_paint(x, y):
    return PAINT_X_MIN <= x <= PAINT_X_MAX and PAINT_Y_MIN <= y <= PAINT_Y_MAX

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def inc_dist(point, points_array):
    distances = [dist(point, pt) for pt in points_array]
    return all(distances[i] < distances[i+1] for i in range(len(distances) - 1))

def ball_above_rim(ball, rim):
    return ball[1] < rim[1]

def ball_below_rim(ball, rim):
    return ball[1] > rim[3]

def shot_made(above_rim, below_rim, rim):
    x1, y1, x2 = rim[0], rim[1], rim[2]
    cx1, cy1, cx2, cy2 = above_rim[0], above_rim[1], below_rim[0], below_rim[1]

    try:
        m = (cy2 - cy1) / (cx2 - cx1)
        b = cy1 - m * cx1
        x = (y1 - b) / m
        return x1 < x < x2
    except ZeroDivisionError:
        return False

def write_text(img, text, location, font_face, font_scale, text_color, background_color, thickness):
    (tw, th), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    top_left = (location[0], location[1] - th - baseline)
    bottom_right = (location[0] + tw, location[1] + baseline)
    cv2.rectangle(img, top_left, bottom_right, background_color, -1)
    cv2.putText(img, text, location, font_face, font_scale, text_color, thickness)

def get_available_filename(output_dir, base_name, extension):
    counter = 1
    output_path = os.path.join(output_dir, f"{base_name}.{extension}")

    while os.path.exists(output_path):
        output_path = os.path.join(output_dir, f"{base_name}{counter}.{extension}")
        counter += 1

    return output_path

def get_input_video():
    vid = argv[1]
    video_path = 'input_vids/' + vid
    return video_path

def get_output_video(cap):
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = get_available_filename('output_vids', 'output', 'mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    return out, frame_height

# -----------------------------------
# ----------- MAIN CODE -------------
# -----------------------------------

H = np.loadtxt("homography_matrix.txt")

PAINT_X_MIN = 50
PAINT_X_MAX = 600
PAINT_Y_MIN = 630
PAINT_Y_MAX = 880

paint_touch_attempts = 0
paint_touch_makes = 0

cap = cv2.VideoCapture(get_input_video())

# Read and rotate first frame to determine correct dimensions
success, img = cap.read()
if not success:
    raise Exception("Could not read the first frame from video")

img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
first_frame = img  # store for use in first iteration

frame_height, frame_width = img.shape[:2]
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter with rotated dimensions
output_path = get_available_filename('output_vids', 'output', 'mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

model = YOLO("model.pt")

classes = ["ball", "made", "person", "rim", "shoot"]

total_attempts = 0
total_made = 0

frame = 0

paint_touch_detected = False
paint_touch_frame = -1
paint_sequences = []


# format = [x_center, y_center, frame]
ball_pos = deque(maxlen=30)
shoot_pos = deque(maxlen=30)

# format = [x1, y1, x2, y2, frame]
rim_pos = deque(maxlen=30)

above_rim_pos = None
overlay = None

popup_text = None
popup_timer = 0

release_calculated = False  # Flag to ensure the release angle is calculated only once
release_angle_degrees = None  # Store the calculated release angle
release_arrow_start = None  # Store the starting point for the arrow
release_arrow_end = None  # Store the ending point for the arrow


# Store the entire ball trajectory
full_ball_trajectory = []

# get total frames for progress bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    if frame == 0:
        img = first_frame  # Already read and rotated
        print(f"Resolution: {img.shape[1]}x{img.shape[0]}")
    else:
        success, img = cap.read()
        if not success:
            break
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    results = model(img, stream=True)

    potential_rims = []
    person_bbox = None

    # Store all detected player bounding boxes
    players = []

    # Store ball position
    ball_position = None   

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            current_class = classes[cls]

            cx = x1 + w // 2
            cy = y1 + h // 2

            bbox_font_scale = 1.5
            bbox_thickness = 2

            if current_class == "person" and conf > 0.6:
                person_bbox = [x1, y1, x2, y2]
                players.append((x1, y1, x2, y2))

            print(f"Players detected: {players}")  # Debugging

            if current_class == "ball" and conf > 0.4:
                ball_pos.append([cx, cy, frame])
                full_ball_trajectory.append((cx, cy))
                ball_position = (cx, cy)
                
                # Store ball's bounding box (x1, y1, x2, y2)
                ball_bbox = (x1, y1, x2, y2)  # This was missing before!

                cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

            if current_class == "rim" and conf > 0.6:
                potential_rims.append(([x1, y1, x2, y2, frame], conf))

            if current_class != "rim" and conf > 0.5:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                write_text(img, f'{current_class} {conf}', (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_PLAIN, bbox_font_scale, (255, 255, 255), (0, 0, 0), bbox_thickness)

    # Identify the ballhandler - If any part of the ball's box is inside a player's bounding box
    ballhandler = None

    if ball_position and players:
        ball_x1, ball_y1, ball_x2, ball_y2 = ball_bbox  # Get ball's bounding box

        for player in players:
            px1, py1, px2, py2 = player  # Get player's bounding box

            # Check if the ball's bounding box intersects with the player's bounding box
            if not (px2 < ball_x1 or px1 > ball_x2 or py2 < ball_y1 or py1 > ball_y2):
                ballhandler = player
                break  # Assign only one player as ballhandler and exit loop

        # if not ballhandler:
        #     # If no direct match, find closest player to ball
        #     min_distance = float('inf')
        #     for player in players:
        #         px1, py1, px2, py2 = player
        #         player_center = ((px1 + px2) // 2, (py1 + py2) // 2)
        #         distance = dist(ball_position, player_center)

        #         if distance < min_distance:
        #             min_distance = distance
        #             ballhandler = player
        #             print(f"Closest Ballhandler: {ballhandler}")  # Debugging

    # Draw the ballhandler with a distinct color
    if ballhandler:
        print("BALLHANDLER CONFIRMED")  # Debugging
        x1, y1, x2, y2 = ballhandler
        bh_cx = (x1 + x2) // 2
        bh_cy = (y1 + y2) // 2
        court_x, court_y = image_to_court(bh_cx, bh_cy, H)
        print(f"Ballhandler at (court): x={court_x:.2f}, y={court_y:.2f}")

        if -8 <= court_x <= 8 and 0 <= court_y <= 19:
            write_text(img, "Paint Touch", (x1, y2 + 20), 
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), (0, 0, 255), 2)
            print(f"Paint touch detected at court coords: ({court_x:.2f}, {court_y:.2f})")
            paint_touch_frame = frame
            paint_touch_detected = True

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)  # Blue bounding box
        write_text(img, "Ballhandler", (x1, y1 - 15), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), (0, 0, 0), 2)
    else:
        print("No Ballhandler Detected")  # Debugging

    if len(potential_rims) > 0:
        mp_rim, conf = reduce(lambda p, c: c, potential_rims)
        rim_pos.append(mp_rim)
        x1, y1, x2, y2 = mp_rim[0], mp_rim[1], mp_rim[2], mp_rim[3]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        write_text(img, f'Rim {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, 
                   (255, 255, 255), (0, 0, 0), 2)

    if above_rim_pos and rim_pos and ball_pos and ball_below_rim(ball_pos[-1], rim_pos[-1]):
        made = shot_made(above_rim_pos, ball_pos[-1], rim_pos[-1])

        if paint_touch_detected and frame - paint_touch_frame < 100:
            paint_touch_attempts += 1
            if made:
                paint_touch_makes += 1
                paint_sequences.append({"frame": paint_touch_frame, "result": "make"})
                print("Paint Touch → Shot Made")
            else:
                paint_sequences.append({"frame": paint_touch_frame, "result": "miss"})
                print("Paint Touch → Shot Missed")
            paint_touch_detected = False

        if made:
            total_made += 1
            #popup_text = "Made!"
        else:
            #popup_text = "Missed!"

        total_attempts += 1
        above_rim_pos = None
        popup_timer = 30


        # # Reset for the next shot
        # release_calculated = False
        # release_angle_degrees = None
        # release_arrow_start = None
        # release_arrow_end = None

    # Reset release_calculated when a new ball is detected in the player's bounding box
    if person_bbox and ball_pos:  
        cx, cy = ball_pos[-1][:2]
        if person_bbox[0] < cx < person_bbox[2] and person_bbox[1] < cy < person_bbox[3]:
            release_calculated = False  # Ready for a new calculation
            release_angle_degrees = None
            release_arrow_start = None
            release_arrow_end = None
            print("ball is within person's grasp")
        
    if rim_pos and ball_pos and ball_above_rim(ball_pos[-1], rim_pos[-1]):
        above_rim_pos = ball_pos[-1]

    scaling_factor = frame_height / 1080 
    font_scale = 2.5 * scaling_factor
    thickness = int(3 * scaling_factor)

    if total_attempts > 0:
        accuracy = (total_made / total_attempts) * 100
        shots_text = f'Shots: {total_made}/{total_attempts} ({accuracy:.1f}%)'
    else:
        shots_text = f'Shots: {total_made}/{total_attempts}'
    
    write_text(img, shots_text, (50, int(150 * scaling_factor)), 
               cv2.FONT_HERSHEY_PLAIN, font_scale, (255, 255, 255), (0, 0, 0), thickness)

    if paint_touch_attempts > 0:
        paint_pct = (paint_touch_makes / paint_touch_attempts) * 100
        paint_text = f'Paint Touch Shots: {paint_touch_makes}/{paint_touch_attempts} ({paint_pct:.1f}%)'
    else:
        paint_text = 'Paint Touch Shots: 0/0'

    write_text(img, paint_text, (50, int(200 * scaling_factor)), 
            cv2.FONT_HERSHEY_PLAIN, font_scale, (255, 255, 255), (0, 0, 0), thickness)

    if popup_timer > 0 and popup_text:
        text_location = (int(img.shape[1] / 2) - 100, int(img.shape[0] / 2))
        write_text(img, popup_text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 
                   2.5, (255, 255, 255), (0, 0, 0), 4)
        popup_timer -= 1
    else:
        popup_text = None

    if overlay is None:
        overlay = np.zeros_like(img, dtype=np.uint8)

    overlay = np.zeros_like(img, dtype=np.uint8)

    # Detect when the ball exits the person's bounding box
    release_frame = None
    if person_bbox and len(full_ball_trajectory) > 10 and not release_calculated:
        for i, (cx, cy) in enumerate(full_ball_trajectory):
            if person_bbox[0] < cx < person_bbox[2] and person_bbox[1] < cy < person_bbox[3]:
                release_frame = i
            else:
                if release_frame is not None:  # Ball just exited the bounding box
                    break

    # Calculate the release angle only when Delta x > 0
    #print("release_frame:" + str(release_frame))
    if release_frame is not None and release_frame + 10 < len(full_ball_trajectory):
        # Get release point and point 10 frames later
        x1, y1 = full_ball_trajectory[release_frame][:2]
        x2, y2 = full_ball_trajectory[release_frame + 5][:2]

        # Check if the ball is moving forward (Delta x > 0)
        if x2 - x1 > 0:
            # Calculate the release angle
            delta_x = x2 - x1
            delta_y = y2 - y1
            release_angle_radians = math.atan2(-delta_y, delta_x)  # Negative delta_y due to inverted y-axis
            release_angle_degrees = math.degrees(release_angle_radians)

            # Store the arrow coordinates for display
            release_arrow_start = (int(x1), int(y1))
            release_arrow_end = (int(x2), int(y2))

            # Set the flag to prevent recalculation
            release_calculated = True

        #print(f"Release Frame: {release_frame}, Release Angle: {release_angle_degrees}")

        # Display the release angle continuously with the arrow
    #print("release_calculated: " + str(release_calculated))
    if release_calculated and release_angle_degrees is not None:
        # Draw the arrow for the release angle
        if release_arrow_start and release_arrow_end:
            cv2.arrowedLine(img, release_arrow_start, release_arrow_end, (0, 255, 0), 3, tipLength=0.2)

        #print("Hi, release angle = " + str(release_angle_degrees))
        # Display the release angle text
        write_text(img, f"Release Angle: {release_angle_degrees:.1f} deg", (50, int(250 * scaling_factor)), 
                cv2.FONT_HERSHEY_PLAIN, font_scale, (255, 255, 255), (0, 0, 0), thickness)


    # Draw the entire ball trajectory (Yellow)
    if len(full_ball_trajectory) > 1:
        for i in range(1, len(full_ball_trajectory)):
            prev_x, prev_y = full_ball_trajectory[i - 1]
            curr_x, curr_y = full_ball_trajectory[i]
            cv2.line(overlay, (prev_x, prev_y), (curr_x, curr_y), (0, 255, 255), 4)

    # write_text(img, f"Time: {cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0:.2f}s Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}", 
    #            (50, 100), cv2.FONT_HERSHEY_PLAIN, font_scale, (255, 255, 255), (0, 0, 0), thickness)

    if total_frames > 0:
        progress = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) / total_frames
        bar_length = 500
        bar_x = 50
        bar_y = frame_height - 70
        bar_height = 30
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_length * progress), bar_y + bar_height), (255, 0, 255), -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_height), (255, 255, 255), 3)

    frame += 1
    print("Current Frame: " + str(frame))
    #cv2.rectangle(img, (PAINT_X_MIN, PAINT_Y_MIN), (PAINT_X_MAX, PAINT_Y_MAX), (0, 255, 255), 2)
    combined_img = cv2.addWeighted(img, 1.0, overlay, 1, 0)

    cv2.imshow("Image", combined_img)
    out.write(combined_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.waitKey(-1)

cap.release()
out.release()
cv2.destroyAllWindows()
