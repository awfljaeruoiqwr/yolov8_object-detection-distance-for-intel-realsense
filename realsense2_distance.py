import cv2
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

point = (400, 300)

def show_distance(event, x, y, args, params):
    global point
    point = (x, y)

cv2.namedWindow("depth frame")
cv2.setMouseCallback("depth frame", show_distance)

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    distance = depth_image[point[1], point[0]]

    center_x = color_image.shape[1] // 2
    center_y = color_image.shape[0] // 2

    cv2.line(color_image, (center_x, 0), (center_x, color_image.shape[0]), (0, 255, 0), 2)
    cv2.line(color_image, (0, center_y), (color_image.shape[1], center_y), (0, 255, 0), 2)

    cv2.putText(color_image, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


    cv2.imshow("depth frame", color_image)
    key = cv2.waitKey(1)
    if key == 27:
        break

