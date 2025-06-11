from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model on a custom dataset.")
    parser.add_argument('--model_cfg', type=str, default='yolov8s.yaml', help='Path to model YAML config file')
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to dataset config file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--project', type=str, default='results', help='Directory to save training results')
    parser.add_argument('--name', type=str, default='car_detection', help='Subdirectory for this run')
    parser.add_argument('--exist_ok', action='store_true', help='Overwrite existing project/name folder if it exists')
    args = parser.parse_args()

    model = YOLO(args.model_cfg)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok
    )

if __name__ == "__main__":
    main()
