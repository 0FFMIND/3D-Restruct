import time  # 用来帧率检查
import cv2  # 用来进行摄像头捕捉
import mediapipe as mp  # 导入BlazePose的Module
import threading  # 启用线程，进行并行处理的优化
import queue  # 进程之间虽然不能共享内存，但可以通过Queue在进程之间传递数据
import socket  # 进行UDP的基础传输
import numpy as np  # 加入数学基本库

# 初始化
cap = cv2.VideoCapture(1)  # 用外置摄像头实时捕捉
cap.set(cv2.CAP_PROP_FPS, 30)  # 尝试将帧率设置为30帧每秒
mpPose = mp.solutions.pose  # 导入BlazePose模型
mpDraw = mp.solutions.drawing_utils  # 对识别到的人体进行描边，可视化给操作人员
# 将置信度从默认的0.5调整，（0.5，1）之间选取中间值，在减少错误的骨骼检测的同时减少模型带来的延迟
pose = mpPose.Pose(min_detection_confidence=0.95, min_tracking_confidence=0.95)
pTime = 0  # 用于帧率检查
jointNum = 33  # 会跟踪33个关节点
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建UDP套接字
serverAddressPort = ("127.0.0.1", 5053)  # 设置本地/外部ip，并使用5053端口单向广播数据
order = 0

# 引入Kalman滤波器进行线性滤波，过滤掉噪声和干扰的影响，通过实际测量和状态预测进行预测修正
kalmanParamQ = 0.070  # Q表示噪声协方差，Q较大系统会认为模型不准
kalmanParamR = 0.095  # R表示测量协方差，R较大系统会认为观察不准，这里R>Q，认为模型暂时准确
K = np.zeros((jointNum, 3), dtype=np.float32)
P = np.zeros((jointNum, 3), dtype=np.float32)
X = np.zeros((jointNum, 3), dtype=np.float32)
# 引入低通滤波器，允许低频信号通过，从而过滤掉短暂的，突发的动作，使得关节点数据更加平稳
lowPassParam = 0.7  # 该参数大，滤波效果差，但对数据反应快速，参数较小，滤波效果好，对数据反应较慢，故选取中间数
# 空白数据包，每一次向接收端发一个储存6帧信息的矩阵，每一帧存储33个骨骼位点，3个维度(x,y,z)
PrevPose3D = np.zeros((6, jointNum, 3), dtype=np.float32)

# 引用并行线程的概念，不影响主线程工作，并行线程的主要概念需引入缓存池
dataBuffer = queue.Queue()


def send_data():
    while True:
        buffer = dataBuffer.get()
        if buffer is None:
            break
        sock.sendto(str(buffer).encode(), serverAddressPort)
        dataBuffer.task_done()


sendThread = threading.Thread(target=send_data)
sendThread.start()

while True:
    success, img = cap.read()  # 若成功读取到人体动态，success会变为true
    if success:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将读取到的这一帧转为RGB图像用于传入模型
        # 将图片输入模型后返回的预测点可视化，方便技术人员进行检查
        results = pose.process(imgRGB)  # 模型输出了33个人体采样点
        if results.pose_landmarks is not None:  # 检查是否识别出姿态
            if len(results.pose_landmarks.landmark) == jointNum:  # 确保检测到了33个关键点，才进到数据传送
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                # 用于检测帧率是否稳定
                cTime = time.time()
                time_delta = cTime - pTime
                fps = 1 / time_delta
                pTime = cTime
                # 将读取到的fps写入视频流旁边, 方便技术人员进行检查
                cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    # 主数据预处理
                    h, w, c = img.shape  # h,w,c 获得当前摄像头显示图像的宽高比
                    data = np.array([lm.x * h, lm.y * w, lm.z * 1000])  # 直接获取关键点的坐标
                    if order == 0:
                        preData = data
                    else:
                        data[0] = data[0] * lowPassParam + (1 - lowPassParam) * preData[0]
                        data[1] = data[1] * lowPassParam + (1 - lowPassParam) * preData[1]
                        data[2] = data[2] * lowPassParam + (1 - lowPassParam) * preData[2]
                    order += 1
                    '''这里1000是为了坐标匹配定义的，可以根据需求更改'''
                    # 进行卡尔曼滤波器的递归
                    K[id] = (P[id] + kalmanParamQ) / (P[id] + kalmanParamQ + kalmanParamR)
                    P[id] = kalmanParamR * (P[id] + kalmanParamQ) / (P[id] + kalmanParamQ + kalmanParamR)
                    smooth_kps = X[id] + (data - X[id]) * K[id]  # 进行滤波操作
                    X[id] = smooth_kps  # 用于下一次迭代
                    # 用于滤波的数据应该是3D的，每一行代表一个关节点的x，y，z坐标
                    PrevPose3D[0, id, :] = smooth_kps  # 获得了经过卡尔曼滤波的估计值
                    sendData = ','.join(PrevPose3D[0].flatten().astype(str).tolist())
                    dataBuffer.put(sendData)
                    # 可视化的操作，给测试人员直观的映像
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
            else:  # 没有检测到33个关键点/关键点被遮盖
                continue  # 选择跳过这一帧
        # 进行Image图像框的展示，进行无限次循环，直到按下按键'q'中止
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
