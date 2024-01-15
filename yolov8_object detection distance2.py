import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

def initialize_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    return pipeline

def get_frames(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not depth_frame or not color_frame:
        return None, None
    return color_frame, depth_frame

def detect_and_annotate(color_frame, depth_frame, model):
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    results = model.predict(color_image, show_labels=False, classes=0, conf=0.65)
    annotated_frame = results[0].plot()

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            distance = depth_frame.get_distance(int(center_x), int(center_y))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"{distance:.2f}m", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    return annotated_frame

def main():
    pipeline = initialize_camera()
    model = YOLO("/home/yang/yolov8n.pt")
    try:
        while True:
            color_frame, depth_frame = get_frames(pipeline)
            if color_frame is None or depth_frame is None:
                continue

            annotated_frame = detect_and_annotate(color_frame, depth_frame, model)
            cv2.imshow('yolov8', annotated_frame)
            if cv2.waitKey(1) & 0xFF in {ord('q'), 27}:
                break
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
