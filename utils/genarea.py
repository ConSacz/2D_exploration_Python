import cv2
import numpy as np

def genarea(image_name):
    # Đọc ảnh gốc (dạng màu)
    image_path = f'./maps/{image_name}.png'
    I_raw = cv2.imread(image_path)  # ảnh đọc theo định dạng BGR (OpenCV mặc định)

    # Resize về 101 x 101
    I_raw = cv2.resize(I_raw, (100, 100), interpolation=cv2.INTER_AREA)

    # Chuyển sang ảnh xám
    I_gray = cv2.cvtColor(I_raw, cv2.COLOR_BGR2GRAY)

    # Chuyển sang kiểu double trong khoảng [0,1]
    Obstacle_Area = I_gray.astype(np.float64) / 255.0

    # Các pixel >0 được gán là 1 (interest area), 0 giữ nguyên (obstacle area)
    Obstacle_Area[Obstacle_Area > 0] = 1

    return Obstacle_Area
