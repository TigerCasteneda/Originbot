# reading Video
import sys
from pathlib import Path
import cv2
import numpy as np
import uuid

def draw_circle(event, x, y, flags, param):    
    global frame, non_transparent_area
    if event == cv2.EVENT_LBUTTONDOWN:
        # 检查点击是否在指定高度范围内
        if h1 <= y <= h2:
            # 在原始帧上绘制圆点
            cv2.circle(frame, (x, y), 25, (0,0,255), -1)
            name = 'xy_%03d_%03d_%s' % (x, y, uuid.uuid1())
            cv2.imwrite('./image_dataset/' + name + '.jpg', non_transparent_area)
            cv2.imshow('Video', frame)
            cv2.waitKey(300)

h1 = 70   # 修改高度值
h2 = 518
DEFAULT_VIDEO_NAME = 'record4.avi'
video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / DEFAULT_VIDEO_NAME
video_file = str(video_path)

# 打开视频文件
cap = cv2.VideoCapture(video_file)  # 修改成车道线视频文件所在位置
ret = cap.isOpened()
# 检查视频是否成功打开
if not ret:
    print(f"Error: Could not open video: {video_file}")
    print(f"Current working directory: {Path.cwd()}")
    print("Put your video file in the current folder or pass its path as the first argument.")
    sys.exit(1)
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_circle)

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        print("Finished playing video.")
        break

    black_screen = np.zeros_like(frame) 
            # 复制本色显示的部分
    non_transparent_area = frame[h1:h2, :, :].copy() 
            # 将非透明区域放置到黑色屏幕上
    black_screen[h1:h2, :, :] = non_transparent_area 
            # 将当前帧与黑色屏幕混合，以实现50%透明度（仅对非本色显示部分）
    alpha = 0.3  # 透明度因子
    mask = np.ones_like(frame, dtype=np.float32)
    mask[h1:h2, :, :] = 0  # 本色显示部分不混合 
            # 使用掩码进行混合
    frame = cv2.addWeighted(frame, alpha, black_screen, 1 - alpha, 0, mask)
            # 显示混合后的帧
    
    # 显示帧
    cv2.imshow('Video', frame)

    # 按下 'ESC' 键退出
    if cv2.waitKey(800) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows() 