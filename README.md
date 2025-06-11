# Parking Space and Car Counting (YOLOv8-based)

This repository contains solutions to real-time parking space detection and car counting using YOLOv8. It covers the following tasks:

---

## 🚩 Problem Statements & Objectives

### Task 1: YOLOv8 Training and Inference on Images(With fine tuning)
- **Objective:** Train a YOLOv8s model to detect parking slots as `empty` or `occupied`.
- **Output:** Bounding boxes with confidence scores and class labels.

### Task 2: Model Export & GPU Inference (ONNX)
- **Objective:** Export the trained model to ONNX format and evaluate performance using GPU with ONNXRuntime.
- **Output:** Frame-wise inference with FPS benchmarking.

### Parking Space & Car Counting (without fine tuning)
- **Objective:** Use a steady surveillance video feed to detect available parking spaces in real time.
- **Method:** Manually define empty slots once → Apply YOLOv8s per frame → Check car presence inside predefined slots.
- **Output:** Dynamic counter showing available vs. occupied spaces (e.g., 12 → 11).
- 🧠 How It Works

        1. **YOLOv8 Inference**: Pretrained YOLOv8 model detects objects in each frame.
        2. **Centroid Calculation**: For each detection, centroid `(cx, cy)` is calculated.
        3. **AOI Detection**: Using `cv2.pointPolygonTest()`, we check if a vehicle centroid lies within any AOI.
        4. **Slot Matching**: If it lies inside an AOI, the vehicle is considered to occupy that slot.

---

## 📦 Data Sources & Preprocessing

- **Source:** Custom parking lot dataset (annotated with `empty` and `occupied`).
- **Split:**
  - Train: 70%
  - Validation: 20%
  - Test: 10%
- **Preprocessing:**
  - Resize all images to 640x640
  - Normalization [0, 1] pixel values
  - Format: YOLO `.txt` annotations with bounding boxes

---

## 🧠 Model Architectures & Training Pipeline

- **Backbone:** [YOLOv8s](https://github.com/ultralytics/ultralytics)
- **Input Size:** 640x640
- **Epochs:** 50  
- **Batch Size:** 8  
- **Loss Function:** YOLOv8's native object detection loss (classification + box regression)
- **Optimizer:** SGD with momentum  
- **Augmentation:** HSV shift, flip, mosaic, mixup (Ultralytics defaults)
  
---


## Model Performance

Evaluation metrics of the trained YOLOv8s model on the validation set:

| Class     | recision | Recall | mAP50 | mAP50-95 |
|-----------|----------|--------|-------|----------|
| all       |   0.927  | 0.922  | 0.962 | 0.837    |
| empty     |   0.888  | 0.898  | 0.944 | 0.779    |
| occupied  |   0.966  | 0.946  | 0.980 | 0.895    |


---

## 📹 Demo

![Real-Time Car Counting Demo](media/parking_demo.gif)

The video shows how available parking slots are updated dynamically using YOLOv8 and manual empty-slot mapping.


