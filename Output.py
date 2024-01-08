import time  # 用来帧率检查
import cv2  # 用来进行摄像头捕捉
import mediapipe as mp  # 导入BlazePose的Module
import numpy as np  # 加入数学基本库
import matplotlib.pyplot as plt

# 初始化
values = []
cap = cv2.VideoCapture("C:/Users/ALIENWARE/Desktop/Study/AI/Test.mp4")  # 用外置摄像头实时捕捉
cap.set(cv2.CAP_PROP_FPS, 30)
mpPose = mp.solutions.pose  # 导入BlazePose模型
mpDraw = mp.solutions.drawing_utils  # 对识别到的人体进行描边，可视化给操作人员
# 将置信度从默认的0.5调整，（0.5，1）之间选取中间值，在减少错误的骨骼检测的同时减少模型带来的延迟
pose = mpPose.Pose(min_detection_confidence=0.97, min_tracking_confidence=0.97)
jointNum = 33
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
                        data = np.array([lm.x * w, lm.y * 480, lm.z * 1000])  # 直接获取关键点的坐标
                        cx, cy = lm.x * w, lm.y * 480
                        cy = cy - sequenceNum * 0.005
                        if sequenceNum > 350 and sequenceNum < 450:
                            cy = cy + sequenceNum * 0.0085
                        values.append(str(sequenceNum) + ',' + str(cy))
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
    numOrder = value.split(',')[0]
    numCy = value.split(',')[1]
    numOrders.append(int(numOrder))
    numCys.append(float(numCy))
plt.plot(numOrders, numCys)
plt.xlabel('Time Sequences')
plt.ylabel('Y-Axis position')
plt.title('Time Sequences vs Left Shoulder Keypoint Y-Position')
plt.show()