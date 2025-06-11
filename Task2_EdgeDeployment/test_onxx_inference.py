import onnxruntime
import cv2
import numpy as np
import time

session = onnxruntime.InferenceSession('best.onnx', providers=['CUDAExecutionProvider'])
input_name = session.get_inputs()[0].name

img = cv2.imread('image1.jpg')
img_resized = cv2.resize(img, (640, 640))
img_input = img_resized.transpose(2, 0, 1).astype(np.float32)
img_input = np.expand_dims(img_input, axis=0) / 255.0

for _ in range(5):  # warm-up
    session.run(None, {input_name: img_input})

start = time.time()
for _ in range(30):
    session.run(None, {input_name: img_input})
end = time.time()

fps = 30 / (end - start)
print(f"Inference FPS on GPU: {fps:.2f}")
