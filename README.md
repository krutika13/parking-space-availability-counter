# parking-space-availability-counter
Parking Space &amp; Car Counting with YOLOv8  A computer vision project using YOLOv8s to detect parked cars, count cars, and identify available parking spaces in real-time. Supports ONNX and TFLite model export for edge deployment. Includes evaluation tools like F1 scores and precision-recall curves.

## Structure

- **Task1/** – YOLOv8 training and image-based predictions to generalize across scenes.  
- **Task2/** – Export of the trained model to ONNX format with GPU inference for real-time detection.  
- **Parking_Space_and_Car_Counting_Steady_Camera/** – A full pipeline that assumes a fixed surveillance camera. Instead of training, parking slots are manually boxed once. YOLOv8s is applied on every frame to detect cars. If a car's detection overlaps with an empty slot, it marks the slot as occupied. The available parking count is updated in real-time (e.g., from 12 to 11 as cars arrive).

Both approaches aim to achieve the same goal—Task1 and Task2 rely on a trained model for generalization, while the steady camera method offers a lightweight and static alternative.

## Model Performance

Evaluation metrics of the trained YOLOv8s model on the validation set:

| Class     | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-----------|--------|-----------|-----------|--------|-------|----------|
| all       | 558    | 11950     | 0.927     | 0.922  | 0.962 | 0.837    |
| empty     | 503    | 3650      | 0.888     | 0.898  | 0.944 | 0.779    |
| occupied  | 484    | 8300      | 0.966     | 0.946  | 0.980 | 0.895    |
