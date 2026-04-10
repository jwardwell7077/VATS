# CLAUDE.md — Project Memory

## Developer: Jon Ward

### System Specs
- **GPU:** NVIDIA GeForce RTX 4070 — 12 GB VRAM
- **CPU:** AMD Ryzen 7 3700X — 8-Core @ 3.60 GHz
- **RAM:** 32 GB DDR4 @ 3200 MT/s
- **Storage:** 15.92 TB (11.24 TB used)
- **OS:** Dual boot — Windows + Linux
- **Dev Environment:** WSL2 Ubuntu (on Windows side)

### Background
- CS degree, summa cum laude
- Technical electives: Robotics, ML, Algorithms, AI, Computer Vision (OpenCV)
- Capstone: TurtleBot "ROSIE" — ROS, LIDAR, autonomous mapping/localization, robotic arm
- Strong physics background
- 1 year career break, currently interviewing

### Interview Context
- Role: Senior Software Engineer — AI Systems / Linux Infrastructure (Chandler, AZ)
- Project: Automated turret — drone detection via RF receiver array + video
- Second interview: Friday April 10, 2026 (2pm-6pm)
- Includes: written portion, technical questions, live coding (array reversal)
- First interview went very well

### Drone Detection Resources
- **Roboflow dataset (people holding drones):** https://universe.roboflow.com/tracker-qjlj1/drones_new
- **Roboflow dataset (drone detect):** https://universe.roboflow.com/kang-igjbn/dronedetect-9grse
- **HuggingFace multi-class model:** Javvanny/yolov8m_flying_objects_detection (drone/plane/heli/bird, 85% drone accuracy)
- **HuggingFace XL model:** doguilmak/Drone-Detection-YOLOv8x (single class, best generalization)
- **Plan:** Fine-tune YOLOv8 Nano on these datasets + 20-30 photos of toy drone. RTX 4070 handles training in ~30 min.
- **Demo setup:** Webcam + broken toy drone held by hand. Laptop for demo (CPU inference is fine for YOLOv8n).
