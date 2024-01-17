import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("/home/yang/yolov8n.pt")

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not depth_frame or not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    results = model.predict(color_image, show_labels = False, classes = 0, conf = 0.7)
    annotated_frame = results[0].plot()

    person_count = 0

    for result in results:
        boxes = result.boxes.cpu().numpy()
        person_count += sum(1 for box in boxes if box.cls == 0)
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            distance = depth_frame.get_distance(int(center_x), int(center_y))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"{distance:.2f}.m", (int(center_x), int(center_y)), cv2.FONT_HERSHEY_DUPLEX, 1.0,
            (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"number of people:{person_count}", (370, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1)

    cv2.imshow('yolov8', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

