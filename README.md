# 🏀 Automated Basketball Analytics for Cal Poly MBB

> Final Senior Project – CSC 492  
> Cal Poly, San Luis Obispo  
> By Yashwant Sathish Kumar  
> Advisor: Dr. Jonathan Ventura

---

## 📽️ Project Overview

This project was developed in collaboration with the **Cal Poly Men’s Basketball (MBB) team** to automate key performance metrics that coaches currently track **manually**. Using computer vision, I built a real-time system that analyzes practice footage to:

- Detect and classify shot attempts
- Estimate release angles
- Identify paint touches
- Automatically determine made/missed outcomes

The goal is to **save coaches time**, **reduce human error**, and provide **visual feedback** to players for more efficient development. This builds upon prior work from EE 428 and extends it with advanced tracking logic and court-space analysis.

---

## 💡 Key Features

### 🎯 Shot Outcome Detection
Uses bounding box tracking and geometric heuristics to classify shots as **made** or **missed**, based on the ball’s trajectory through the rim.

### 📐 Shot Angle Estimation
Calculates the **release angle** of a shot using the ball’s position at release and five frames later. It visualizes the release vector with a green arrow and displays the angle in degrees.

### 🏀 Paint Touch Detection
Maps player coordinates to **real-world court space** using homography. When a player dribbles into the paint (x: -8 to 8 ft, y: 0 to 19 ft), it records a paint touch and links it to shot outcomes.

### 👤 Ballhandler Identification
Determines which player is holding the ball by checking **bounding box overlap** between the player and the ball.

### 🖼️ Visual Overlays
Displays:
- 🟡 Ball trajectory (yellow lines)
- 🔴 Ball position (red dots)
- 🟩 Shot angle (green arrow)
- 📊 Stats on makes, attempts, and paint-specific percentages

---

## 🧠 Technical Stack

- **Object Detection**: YOLOv8 (via Ultralytics)
- **Computer Vision**: OpenCV
- **Tracking**: Custom trajectory tracking with `deque`
- **Geometric Analysis**: Trigonometry for angle estimation
- **Homography Mapping**: Manual calibration using court landmarks
- **Dataset**: Custom-annotated videos of Cal Poly Men’s Basketball practices, labeled using Roboflow

---

## 📊 Evaluation Metrics

| Task                 | Precision | Recall | F1-Score |
|----------------------|----------:|-------:|---------:|
| Shot Outcome (32 shots) | 88.9%     | 93.8%  | 91.3%    |
| Paint Touch (15 plays)  | 92.9%     | 86.7%  | 89.7%    |

- **Shot Angle Accuracy**:  
  - Mean angle: 54.7°  
  - Avg deviation: ±5.2° from HomeCourt app  
  - 90% of angles fell in valid range (40°–70°)

---

## 📎 Example Output

> ![Shot angle and paint detection](example_frame.jpg)  
> *Green arrow indicates release angle, with in-paint detection labeled.*

---

## 📂 File Highlights

- `main.py` – Core program for tracking, detection, and visualization.
- `homography_matrix.txt` – Matrix for mapping pixel coordinates to real-world court space.
- `model.pt` – YOLOv8 trained weights via Roboflow (not included in repo).
- `input_vids/` – Folder for practice/game footage.
- `output_vids/` – Generated videos with overlayed analytics.

---

## 📌 Lessons Learned

This project pushed me to balance model accuracy with real-world video noise and frame ambiguity. Key takeaways:

- Manual homography calibration worked better than automated feature matching due to lighting inconsistencies.
- Frame-by-frame tracking needs post-processing to handle rim interactions and avoid overcounting makes.
- Player-ball overlap frames (e.g., during handoffs) are tough—future improvements could use pose estimation or optical flow.

---

## 🔐 Ethical Considerations

- **Privacy**: Only internal team footage was used; future applications should involve player consent.
- **Bias**: I trained the YOLO model on a diverse dataset to avoid appearance-based detection bias.
- **Usage**: Coaches should treat the data as one layer of feedback—not an absolute judgment of performance.

---

## 🚀 Future Improvements

- Add temporal smoothing to reduce false positives (e.g., rim bounces).
- Improve release point detection using optical flow or pose estimation.
- Automate court calibration for new gyms or angles.
- Expand analytics to include off-ball movement and defensive contests.

---
