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
valuesMix = []
mix = []
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
def Kalman1D(observations, damping, previous, cov):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations
    transition_matrix = 1
    transition_covariance = cov
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state, state_cov = kf.filter_update(previous, transition_covariance, observations)
    return pred_state[0][0]

mpPose = mp.solutions.pose  # 导入BlazePose模型
mpDraw = mp.solutions.drawing_utils  # 对识别到的人体进行描边，可视化给操作人员
pose = mpPose.Pose(min_detection_confidence=0.97, min_tracking_confidence=0.97)

values1, values2, values3, values4 = [], [], [], []

def Capinit(cap, pose, opcode):
    begin = time.time()
    sequenceNum = 0
    while True:
        now = time.time()
        success, img = cap.read()  # 若成功读取到人体动态，success会变为true
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
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
                        if opcode == 1:
                            values1.append(str(time.time() - begin) + ',' + str(cy))
                            if sequenceNum > 1:
                                valuesLow.append(
                                    str(time.time() - begin) + ',' + str(alpha * cy + (1 - alpha) * previousCy))
                            previousCy = cy
                        if opcode == 2:
                            values2.append(str(now - begin) + ',' + str(cy))
                            if sequenceNum == 1:
                                state_mean, state_cov = cy, 0.01
                                state_mean, state_cov = kf.filter_update(state_mean, state_cov, cy)
                                state_mean_value = state_mean[0][0]
                                valuesKalman.append(str(time.time() - begin) + ',' + str(state_mean_value))
                            else:
                                state_mean, state_cov = kf.filter_update(state_mean, state_cov, cy)
                                state_mean_value = state_mean[0][0]
                                valuesKalman.append(str(time.time() - begin) + ',' + str(state_mean_value))
                        if opcode == 3:
                            values4.append(str(now - begin) + ',' + str(cy))
                            if sequenceNum == 1:
                                mix.append(cy)
                            if sequenceNum > 1:
                                Mix = 0.7 * cy + (1 - 0.7) * previousCy2
                                state_mean_value2 = Kalman1D(Mix, 0.013, mix[sequenceNum - 2], 0.03)
                                mix.append(state_mean_value2)
                                valuesMix.append(str(time.time() - begin) + ',' + str(state_mean_value2))
                            previousCy2 = cy
                        if opcode == 4:
                            window_length = 7
                            polyorder = 3
                            valuescy.append(cy)
                            values3.append(str(now - begin) + ',' + str(cy))
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

# opcode
cap = cv2.VideoCapture(1)  # 用外置摄像头实时捕捉
cap.set(cv2.CAP_PROP_FPS, 30)  # 尝试将帧率设置为30帧每秒
Capinit(cap, pose, 3)

def Flatten(values, numOrders, numCys):
    for value in values:
        numOrder, numCy = value.split(',')
        numOrders.append(float(numOrder))
        numCys.append(float(numCy))

numOrders1, numOrders2, numOrders3, numOrders4 = [], [], [], []
numCys1, numCys2, numCys3, numCys4 = [], [], [], []
numCy1, numCy2, numCy3, numCy4 = [], [], [], []
num1, num2, num3, num4 = [], [], [], []
Flatten(valuesLow, numOrders1, numCys1)
Flatten(values1, num1, numCy1)
Flatten(valuesKalman, numOrders2, numCys2)
Flatten(values2, num2, numCy2)
Flatten(valuesSavgol, numOrders3, numCys3)
Flatten(values3, num3, numCy3)
Flatten(valuesMix, numOrders4, numCys4)
Flatten(values4, num4, numCy4)
# plt.plot(num1, numCy1, label='Original_Lowpass', color='blue')
# plt.plot(numOrders1, numCys1, label='Lowpass Filtered', color='green', alpha=1)
# plt.plot(num2, numCy2, label='Original', color='blue')
# plt.plot(numOrders2, numCys2, label='Kalman Filtered', color='red', alpha=1)
# plt.plot(num3, numCy3, label='Original', color='blue')
# plt.plot(numOrders3, numCys3, label='Savgol Filtered', color='orange', alpha=1)
plt.plot(num4, numCy4, label='Original', color='blue')
plt.plot(numOrders4, numCys4, label='Mixed Filtered', color='orange', alpha=1)

plt.xlabel('Time Sequences')
plt.ylabel('Y-Axis position')
plt.title('Time Sequences vs Left Shoulder Keypoint Y-Position')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
plt.grid()
plt.show()

num4.pop()

d = np.mean(np.array(numOrders1) - np.array(num1))
e = np.mean(np.array(numOrders2) - np.array(num2))
f = np.mean(np.array(numOrders3) - np.array(num3))
g = np.mean(np.array(numOrders4) - np.array(num4))

print(d, e, f, g)