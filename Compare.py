import time  # 用来帧率检查
import cv2  # 用来进行摄像头捕捉
import mediapipe as mp  # 导入BlazePose的Module
import numpy as np  # 加入数学基本库
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error

# 初始化
values = []
valuesKalman = []
cap = cv2.VideoCapture("C:/Users/ALIENWARE/Desktop/Study/AI/Tewst.mp4")  # 用外置摄像头实时捕捉
cap.set(cv2.CAP_PROP_FPS, 30)
mpPose = mp.solutions.pose  # 导入BlazePose模型
mpDraw = mp.solutions.drawing_utils  # 对识别到的人体进行描边，可视化给操作人员
# 将置信度从默认的0.5调整，（0.5，1）之间选取中间值，在减少错误的骨骼检测的同时减少模型带来的延迟
pose = mpPose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1)
jointNum = 33
sequenceNum = 0
# 初始化一阶低通滤波器
alpha = 0.1
valuesLow = []
kf = KalmanFilter(initial_state_mean=208, n_dim_obs=1)
valuesSavgol = []
valuescy = []
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
                        data = np.array([lm.x * w, lm.y * 480, lm.z * 1000])  # 直接获取关键点的坐标
                        cx, cy = lm.x * w, lm.y * 480
                        cy = cy - sequenceNum * 0.005
                        if sequenceNum > 350 and sequenceNum < 450:
                            cy = cy + sequenceNum * 0.0085
                        valuescy.append(cy)
                        values.append(str(sequenceNum) + ',' + str(cy))

                        # 使用卡尔曼滤波器进行状态估计
                        if sequenceNum == 1:
                            state_mean, state_cov = cy, 0.001
                            valuesKalman.append(state_mean)
                        else:
                            state_mean, state_cov = kf.filter_update(state_mean, state_cov, cy)
                            state_mean_value = state_mean[0][0]
                            valuesKalman.append(state_mean_value)

                        # 使用一阶低通滤波器
                        if sequenceNum == 1:
                            valuesLow.append(cy)
                        if sequenceNum > 1:
                            valuesLow.append(alpha * cy + (1 - alpha) * previousCy)
                        previousCy = cy

                        # 执行SAVGOL
                        # 执行Savitzky-Golay滤波
                        window_length = 7  # 窗口长度（奇数）
                        polyorder = 3  # 多项式阶数
                        if sequenceNum >= window_length:
                            # Apply the SAVGOL filter to the list of values up to the current point
                            y_smoothed = savgol_filter(valuescy[-window_length:], window_length, polyorder)
                            # Append the last smoothed value to the `valuesSavgol` list
                            valuesSavgol.append(y_smoothed[-1])
                        else:
                            # Not enough points yet, just append the current `cy` to `valuesSavgol`
                            valuesSavgol.append(cy)


        else:
            continue
    else:
        break

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
numOrders = []
numCys = []
for value in values:
    numOrder, numCy = value.split(',')
    numOrders.append(int(numOrder))
    numCys.append(float(numCy))

plt.plot(numOrders, numCys, label='Original', color='blue')
plt.plot(numOrders, valuesLow, label='Lowpass Filtered', color='green', alpha=0.5)
plt.plot(numOrders, valuesKalman, label='Kalman Filtered', color='red', alpha=0.8)
plt.plot(numOrders, valuesSavgol, label='Savgol Filtered', color='yellow', alpha=0.7)
plt.xlabel('Time Sequences')
plt.ylabel('Y-Axis position')
plt.title('Time Sequences vs Left Shoulder Keypoint Y-Position')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=4)
plt.grid()
plt.show()

# 计算均方误差
a = mean_squared_error(numCys, valuesLow)
b = mean_squared_error(numCys, valuesKalman)
c = mean_squared_error(numCys, valuesSavgol)
print(a, b, c)