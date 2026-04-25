# reading Video
import sys
from pathlib import Path
import cv2
import numpy as np
h1 = 70   # 修改高度值
h2 = 518

DEFAULT_VIDEO_NAME = 'record4.avi'

def main():
    # 使用当前工作目录中的视频文件，或从命令行参数读取路径
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
   
    while(ret):
        ret, frame = cap.read()
        if ret == True:            
            # 创建一个与帧相同大小的黑色屏幕，用于混合以实现透明度
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
            blended_frame = cv2.addWeighted(frame, alpha, black_screen, 1 - alpha, 0, mask)
            # 显示混合后的帧
            cv2.imshow('blended_frame', blended_frame)                
            k = cv2.waitKey(200)
            if( k == ord('q')):
                break 
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()