import time  # 用来帧率检查
import cv2  # 用来进行摄像头捕捉
import mediapipe as mp  # 导入BlazePose的Module
import numpy as np  # 加入数学基本库
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from scipy.signal import savgol_filter

# 初始化
jointNum = 33
alpha = 0.1
valuescy = []
values, valuesLow, valuesKalman, valuesSavgol = [], [], [], []
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
mpPose = mp.solutions.pose  # 导入BlazePose模型
mpDraw = mp.solutions.drawing_utils  # 对识别到的人体进行描边，可视化给操作人员
pose = mpPose.Pose(min_detection_confidence=0.97, min_tracking_confidence=0.97)

def Capinit(cap, pose, opcode):
    begin = time.time()
    sequenceNum = 0
    while True:
        success, img = cap.read()  # 若成功读取到人体动态，success会变为true
        if success:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将读取到的这一帧转为RGB图像用于传入模型
            results = pose.process(imgRGB)  # 模型输出了33个人体采样点
            if results.pose_landmarks is not None:  # 检查是否识别出姿态
                if len(results.pose_landmarks.landmark) == jointNum:  # 确保检测到了33个关键点，才进到数据传送
                    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS, landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=4), connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 255), thickness=3))
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    if id == 11:
                        sequenceNum = sequenceNum + 1
                        # 主数据预处理
                        h, w, c = img.shape  # h,w,c 获得当前摄像头显示图像的宽高比
                        cx, cy = lm.x * w, lm.y * 480
                        if opcode == 0:
                            continue
                        if opcode == 1:
                            values.append(str(time.time() - begin) + ',' + str(cy))
                        if opcode == 2:
                            # 使用一阶低通滤波器
                            if sequenceNum == 1:
                                valuesLow.append(str(time.time() - begin) + ',' + str(cy))
                            if sequenceNum > 1:
                                valuesLow.append(
                                    str(time.time() - begin) + ',' + str(alpha * cy + (1 - alpha) * previousCy))
                            previousCy = cy
                        if opcode == 3:
                            # 使用卡尔曼滤波器进行状态估计
                            if sequenceNum == 1:
                                state_mean, state_cov = cy, 0.001
                                valuesKalman.append(str(time.time() - begin) + ',' + str(state_mean))
                            else:
                                state_mean, state_cov = kf.filter_update(state_mean, state_cov, cy)
                                state_mean_value = state_mean[0][0]
                                valuesKalman.append(str(time.time() - begin) + ',' + str(state_mean_value))
                        if opcode == 4:
                            window_length = 7  # 窗口长度（奇数）
                            polyorder = 3  # 多项式阶数
                            valuescy.append(cy)
                            if sequenceNum >= window_length:
                                y_smoothed = savgol_filter(valuescy[-window_length:], window_length, polyorder)
                                valuesSavgol.append(str(time.time() - begin) + ',' + str(y_smoothed[-1]))
                            else:
                                valuesSavgol.append(str(time.time() - begin) + ',' + str(cy))
            else:
                continue
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

# 第0轮
cap = cv2.VideoCapture("C:/Users/ALIENWARE/Desktop/Study/AI/Tewst.mp4")  # 用外置摄像头实时捕捉
cap.set(cv2.CAP_PROP_FPS, 30)
Capinit(cap, pose, 0)

# 第1轮
cap = cv2.VideoCapture("C:/Users/ALIENWARE/Desktop/Study/AI/Tewst.mp4")  # 用外置摄像头实时捕捉
cap.set(cv2.CAP_PROP_FPS, 30)
Capinit(cap, pose, 1)

# 第2轮
cap = cv2.VideoCapture("C:/Users/ALIENWARE/Desktop/Study/AI/Tewst.mp4")  # 用外置摄像头实时捕捉
cap.set(cv2.CAP_PROP_FPS, 30)
Capinit(cap, pose, 2)

# 第3轮
cap = cv2.VideoCapture("C:/Users/ALIENWARE/Desktop/Study/AI/Tewst.mp4")  # 用外置摄像头实时捕捉
cap.set(cv2.CAP_PROP_FPS, 30)
Capinit(cap, pose, 4)

# 第4轮
cap = cv2.VideoCapture("C:/Users/ALIENWARE/Desktop/Study/AI/Tewst.mp4")  # 用外置摄像头实时捕捉
cap.set(cv2.CAP_PROP_FPS, 30)
Capinit(cap, pose, 3)

def Flatten(values, numOrders, numCys):
    for value in values:
        numOrder, numCy = value.split(',')
        numOrders.append(float(numOrder))
        numCys.append(float(numCy))

numOrders1, numOrders2, numOrders3, numOrders4 = [], [], [], []
numCys1, numCys2, numCys3, numCys4 = [], [], [], []

Flatten(values, numOrders1, numCys1)
Flatten(valuesLow, numOrders2, numCys2)
Flatten(valuesKalman, numOrders3, numCys3)
Flatten(valuesSavgol, numOrders4, numCys4)

plt.plot(numOrders1, numCys1, label='Original', color='blue')
plt.plot(numOrders2, numCys2, label='Lowpass Filtered', color='green', alpha=0.5)
plt.plot(numOrders3, numCys3, label='Kalman Filtered', color='red', alpha=0.8)
plt.plot(numOrders4, numCys4, label='Savgol Filtered', color='yellow', alpha=0.7)
plt.xlabel('Time Sequences')
plt.ylabel('Y-Axis position')
plt.title('Time Sequences vs Left Shoulder Keypoint Y-Position')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=4)
plt.grid()
plt.show()

d = np.mean(np.array(numOrders2) - np.array(numOrders1))
e = np.mean(np.array(numOrders3) - np.array(numOrders1))
f = np.mean(np.array(numOrders4) - np.array(numOrders1))

print(d, e, f)