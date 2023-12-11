using System;
using System.Collections.Generic;
using UnityEngine;

public class Main : MonoBehaviour
{
    Animator animator; // 用来恢复到默认站姿
    public RuntimeAnimatorController rac; // 同样，该参数用来恢复到默认站姿
    private Transform root, spine, neck, head, leye, reye, 
        lshoulder, lelbow, lwrist, lpinky, lindex, lthumb,
        rshoulder, relbow, rwrist, rpinky, rindex,rthumb,
        lhip, lknee, lankle, ltoe, rhip, rknee, rankle, rtoe;
    public Vector3 forward;
    private Quaternion midRoot, midSpine, midNeck, midHead, 
        midLshoulder, midLelbow, midLwrist, midLpinky, midLindex,
        midRshoulder, midRelbow, midRwrist, midRpinky, midRindex,
        midLhip, midLknee, midLankle, midLheel, midRhip, midRknee, midRankle, midRheel;
    private Vector3 centerRootPos; // 人物中央点
    // 因为在Unity中，人物Avatar没有能挂载鼻子，脚后跟坐标的骨骼点，故这里手动将坐标绑定
    public Transform nose;  // 鼻子 0
    public Transform lheel;  // 左后跟 29
    public Transform rheel;  // 右后跟 30
    // 将之前接收UDP数据的Receiver挂载到这里udpReceiver上，取得接收端string的数据，从而进行后续的处理
    public UDPReceiver udpReceiver;
    private bool onceCheck = false;
    private Vector3 offsetInit;
    // 测试可视化用的圆球点
    //public GameObject dotGroup;
    
    void Start()
    {
        // 动画相关，若没有接收到骨骼动画，则恢复到默认站姿
        animator = GetComponent<Animator>();
        //躯干
        root = animator.GetBoneTransform(HumanBodyBones.Hips); // 人物重心点根骨骼
        centerRootPos = root.position;
        spine = animator.GetBoneTransform(HumanBodyBones.Spine); // 重心点上方躯干
        neck = animator.GetBoneTransform(HumanBodyBones.Neck); // 颈部
        head = animator.GetBoneTransform(HumanBodyBones.Head); // 头部
        leye = animator.GetBoneTransform(HumanBodyBones.LeftEye); // 左眼 2
        reye = animator.GetBoneTransform(HumanBodyBones.RightEye); // 右眼 5
        //左臂
        lshoulder = animator.GetBoneTransform(HumanBodyBones.LeftUpperArm); // 左肩 11
        lelbow = animator.GetBoneTransform(HumanBodyBones.LeftLowerArm); // 左肘 13
        lwrist = animator.GetBoneTransform(HumanBodyBones.LeftHand); // 左腕 15
        lpinky = animator.GetBoneTransform(HumanBodyBones.LeftLittleProximal); //左小拇指 17
        lindex = animator.GetBoneTransform(HumanBodyBones.LeftIndexProximal); // 左食指 19
        lthumb = animator.GetBoneTransform(HumanBodyBones.LeftThumbIntermediate); // 左拇指 21
        //右臂
        rshoulder = animator.GetBoneTransform(HumanBodyBones.RightUpperArm); // 右肩 12
        relbow = animator.GetBoneTransform(HumanBodyBones.RightLowerArm); // 右肘 14
        rwrist = animator.GetBoneTransform(HumanBodyBones.RightHand); // 右腕 16
        rpinky = animator.GetBoneTransform(HumanBodyBones.RightLittleProximal); // 右小拇指 18
        rindex = animator.GetBoneTransform(HumanBodyBones.RightIndexProximal); // 右食指 20
        rthumb = animator.GetBoneTransform(HumanBodyBones.RightThumbIntermediate); // 右拇指 22
        //左腿
        lhip = animator.GetBoneTransform(HumanBodyBones.LeftUpperLeg); // 左臀部 23
        lknee = animator.GetBoneTransform(HumanBodyBones.LeftLowerLeg); // 左膝 25
        lankle = animator.GetBoneTransform(HumanBodyBones.LeftFoot); //左踝 27
        ltoe = animator.GetBoneTransform(HumanBodyBones.LeftToes); // 左足尖 31
        //右腿
        rhip = animator.GetBoneTransform(HumanBodyBones.RightUpperLeg); // 右臀部 24
        rknee = animator.GetBoneTransform(HumanBodyBones.RightLowerLeg); //右膝 26
        rankle = animator.GetBoneTransform(HumanBodyBones.RightFoot); // 右踝 28
        rtoe = animator.GetBoneTransform(HumanBodyBones.RightToes); // 右足尖 32
        //首先，输入根骨骼，左臀部，右臀部的三点三维坐标，放入数学函数计算出目前人体正前方朝向的法向量
        forward = TriangleNormal(root.position, lhip.position, rhip.position);
        // 其次，比起仅呈现每个骨骼关键点的位移，为更好模拟真实人物的骨骼旋转，我们引入中间变化矩阵
        // 根骨骼不与其他骨骼节点连接，因此直接用forward三维向量创建Quaternion四元数看向人体正前方的旋转角
        // 再乘上根骨骼本身的旋转创建中间变化矩阵
        midRoot = Quaternion.Inverse(root.rotation) * Quaternion.LookRotation(forward);
        // 如果存在可以用来做参考的向量坐标，这里四元旋转角的计算选择躯干到颈部的向量方向，再计算出与人正前方朝向法向量的旋转角再去中间矩阵，其余均同样
        midSpine = Quaternion.Inverse(spine.rotation) * Quaternion.LookRotation(spine.position - neck.position, forward); //躯干
        midNeck = Quaternion.Inverse(neck.rotation) * Quaternion.LookRotation(neck.position - head.position, forward); //颈部
        midHead = Quaternion.Inverse(head.rotation) * Quaternion.LookRotation(head.position - leye.position,forward); //头部
        // 肩部以下
        midLshoulder = Quaternion.Inverse(lshoulder.rotation) * Quaternion.LookRotation(lshoulder.position - lelbow.position, forward);
        midLelbow = Quaternion.Inverse(lelbow.rotation) * Quaternion.LookRotation(lelbow.position - lwrist.position, forward);
        midLwrist = Quaternion.Inverse(lwrist.rotation) * Quaternion.LookRotation(
            lwrist.position - lpinky.position,
            TriangleNormal(lpinky.position, lindex.position, lthumb.position)
            );
        midLpinky = Quaternion.Inverse(lpinky.rotation) * Quaternion.LookRotation(
            lpinky.position - lindex.position,
            TriangleNormal(lpinky.position, lindex.position, lthumb.position)
            );
        midLindex = Quaternion.Inverse(lindex.rotation) * Quaternion.LookRotation(
            lindex.position - lthumb.position,
            TriangleNormal(lpinky.position, lindex.position, lthumb.position)
            );
        midRshoulder = Quaternion.Inverse(rshoulder.rotation) * Quaternion.LookRotation(rshoulder.position - relbow.position, forward);
        midRelbow = Quaternion.Inverse(relbow.rotation) * Quaternion.LookRotation(relbow.position - rwrist.position, forward);
        midRwrist = Quaternion.Inverse(rwrist.rotation) * Quaternion.LookRotation(
            rwrist.position - rpinky.position,
            TriangleNormal(rpinky.position, rindex.position, rthumb.position)
            );
        midRpinky = Quaternion.Inverse(rpinky.rotation) * Quaternion.LookRotation(
            rpinky.position - rindex.position,
            TriangleNormal(rpinky.position, rindex.position,rthumb.position)
            );
        midRindex = Quaternion.Inverse(rindex.rotation) * Quaternion.LookRotation(
            rindex.position - rthumb.position,
            TriangleNormal(rpinky.position, rindex.position, rthumb.position)
            );
        // 臀部以下
        midLhip = Quaternion.Inverse(lhip.rotation) * Quaternion.LookRotation(lhip.position - lknee.position, forward);
        midLknee = Quaternion.Inverse(lknee.rotation) * Quaternion.LookRotation(lknee.position - lankle.position, forward);
        midLankle = Quaternion.Inverse(lankle.rotation) * Quaternion.LookRotation(lankle.position - lheel.position, forward);
        midLheel = Quaternion.Inverse(lheel.rotation) * Quaternion.LookRotation(lheel.position - ltoe.position, lankle.position - lheel.position);
        midRhip = Quaternion.Inverse(rhip.rotation) * Quaternion.LookRotation(rhip.position - rknee.position, forward);
        midRknee = Quaternion.Inverse(rknee.rotation) * Quaternion.LookRotation(rknee.position - rankle.position, forward);
        midRankle = Quaternion.Inverse(rankle.rotation) * Quaternion.LookRotation(rankle.position - rheel.position, forward);
        midRheel = Quaternion.Inverse(rheel.rotation) * Quaternion.LookRotation(rheel.position - rankle.position, rankle.position - rheel.position);
    }

    // 输入三点，计算三角形法向量
    Vector3 TriangleNormal(Vector3 a, Vector3 b, Vector3 c)
    {
        return Vector3.Cross((a - b), a - c).normalized;
    }
    // 在接收端每帧调用更新姿势的函数
    void Update()
    {
        // 更新姿势
        updatePose();
    }
    void updatePose()
    {
        if (udpReceiver.data != null) // 当接收端的data数据不为空的时候，使用临时变量接收data
        {
            string data = udpReceiver.data;
            try // 使用try-catch进行异常捕获，以保证不影响主数据流传输的进行
            {
                string[] points = data.Split(','); // 每一次会获得99个点的字符串，将字符串data按','分隔开，即得到了按顺序排列的单个数据点的字符串
                if (points.Length == 99)
                {
                    try
                    {
                        Dictionary<int,Vector3> poseDot3d = new Dictionary<int, Vector3>(); // 创建一个字典储存33个数据点的列表，每一个点里面储存了x,y,z坐标
                        for (int i = 0; i < points.Length / 3; i++) //重复33次
                        {
                            // 此时的points为字符串的类型，因此用字符串转换为double后储存在poseDot3d列表中
                            float pointX = float.Parse(points[i * 3]) * 0.6f;
                            float pointY = - float.Parse(points[i * 3 + 1]) * 0.4f;
                            float pointZ = float.Parse(points[i * 3 + 2]) * 0.05f;
                            Vector3 oneDot = new Vector3(pointX, pointY, pointZ);
                            poseDot3d.Add(i,oneDot);
                        }
                        // 现在进行接受点坐标与unity人形坐标进行转换，由于BlazePose自带的33个骨骼点无法完全对应unity人形avatar的驱动点，因此需要坐标转换
                        // 手动计算四个部分的位置(不在BlazePose的33个特征点中，故使用left shoulder和right shoulder的中间点加上向量偏移找到neck
                        Vector3 dotNeck = (poseDot3d[11] + poseDot3d[12]) * 0.5f;
                        Vector3 dotHips = (poseDot3d[23] + poseDot3d[24]) * 0.5f;
                        Vector3 dotMid = (dotNeck + dotHips) * 0.5f;
                        Vector3 dotHead = ((poseDot3d[2] + poseDot3d[5]) * 0.5f + (poseDot3d[11] + poseDot3d[12]) * 0.5f) * 0.5f;
                        Vector3 dotSpine = (dotHips + dotMid) * 0.5f;
                        Vector3 dotRoot = (dotHips + dotSpine) * 0.5f;
                        // 预测位移，人物除开单纯的做出动作的同时会左右移动，只需要跟踪它的dotRoot节点即可
                        if (!onceCheck)
                        {
                            offsetInit = dotRoot; // 记录首次位移
                            onceCheck = true;
                            // 第一次不更新位移
                        }
                        else
                        {
                            Vector3 minOffset = new Vector3(-100f, -1f, -50f);
                            Vector3 maxOffset = new Vector3(100f, 1f, 50f); // y轴是朝向地面的轴,
                            // 进行迭代直到左右位移的量在限制范围内
                            Vector3 offset = dotRoot - offsetInit;
                            bool isOKX = false;
                            while (!isOKX)
                            {
                                if (minOffset.x < offset.x && offset.x < maxOffset.x)
                                {
                                    isOKX = true;
                                }
                                else
                                {
                                    offset.x *= 0.8f;
                                }
                            }
                            bool isOKY = false;
                            while (!isOKY)
                            {
                                if (minOffset.y < offset.y && offset.y < maxOffset.y)
                                {
                                    isOKY = true;
                                }
                                else
                                {
                                    offset.y *= 0.8f;
                                }
                            }
                            bool isOKZ = false;
                            while (!isOKZ)
                            {
                                if (minOffset.z < offset.z && offset.z < maxOffset.z)
                                {
                                    isOKZ = true;
                                }
                                else
                                {
                                    offset.z *= 0.8f;
                                }
                            }
                            // 更新位移
                            root.position = centerRootPos + offset;
                        }
                        ///
                        /// 测试用圆球点，可视化成功
                        ///
                        //for (int i = 0; i < 33; i++)
                        //{
                        //    dotGroup.transform.GetChild(i).gameObject.GetComponent<Transform>().position = poseDot3d[i] + dotGroup.GetComponent<Transform>().position;
                        //}
                        // 更新旋转
                        Vector3 forward = TriangleNormal(dotRoot, poseDot3d[23], poseDot3d[24]);
                        root.rotation = Quaternion.LookRotation(forward) * Quaternion.Inverse(midRoot);
                        // 下面按照最开始计算的中心旋转矩阵复原，仅更新骨骼旋转
                        spine.rotation = Quaternion.LookRotation(dotSpine - dotNeck, forward) * Quaternion.Inverse(midSpine);
                        neck.rotation = Quaternion.LookRotation(dotNeck - dotHead, forward) * Quaternion.Inverse(midNeck);
                        head.rotation = Quaternion.LookRotation(dotHead - poseDot3d[2],forward) * Quaternion.Inverse(midHead);
                        lshoulder.rotation = Quaternion.LookRotation(poseDot3d[11] - poseDot3d[13], forward) * Quaternion.Inverse(midLshoulder);
                        rshoulder.rotation = Quaternion.LookRotation(poseDot3d[12] - poseDot3d[14], forward) * Quaternion.Inverse(midRshoulder);
                        lelbow.rotation = Quaternion.LookRotation(poseDot3d[13] - poseDot3d[15], forward) * Quaternion.Inverse(midLelbow);
                        relbow.rotation = Quaternion.LookRotation(poseDot3d[14] - poseDot3d[16], forward) * Quaternion.Inverse(midRelbow);
                        lwrist.rotation = Quaternion.LookRotation(
                            poseDot3d[15] - poseDot3d[17], TriangleNormal(poseDot3d[17], poseDot3d[19], poseDot3d[21])) * Quaternion.Inverse(midLwrist);
                        rwrist.rotation = Quaternion.LookRotation(
                            poseDot3d[16] - poseDot3d[18], TriangleNormal(poseDot3d[18], poseDot3d[20], poseDot3d[22])) * Quaternion.Inverse(midRwrist);
                        lpinky.rotation = Quaternion.LookRotation(
                            poseDot3d[17] - poseDot3d[19],
                            TriangleNormal(poseDot3d[17], poseDot3d[19], poseDot3d[21])) * Quaternion.Inverse(midLpinky);
                        rpinky.rotation = Quaternion.LookRotation(
                            poseDot3d[18] - poseDot3d[20],
                            TriangleNormal(poseDot3d[18], poseDot3d[20], poseDot3d[22])) * Quaternion.Inverse(midRpinky);
                        lindex.rotation = Quaternion.LookRotation(
                            poseDot3d[19] - poseDot3d[21],
                            TriangleNormal(poseDot3d[17], poseDot3d[19], poseDot3d[21])) * Quaternion.Inverse(midLindex);
                        rindex.rotation = Quaternion.LookRotation(
                            poseDot3d[20] - poseDot3d[22],
                            TriangleNormal(poseDot3d[18], poseDot3d[20], poseDot3d[22])) * Quaternion.Inverse(midRindex);
                        // 臀部以下
                        lhip.rotation = Quaternion.LookRotation(poseDot3d[23] - poseDot3d[25], forward) * Quaternion.Inverse(midLhip);
                        rhip.rotation = Quaternion.LookRotation(poseDot3d[24] - poseDot3d[26], forward) * Quaternion.Inverse(midRhip);
                        lknee.rotation = Quaternion.LookRotation(poseDot3d[25] - poseDot3d[27], forward) * Quaternion.Inverse(midLknee);
                        rknee.rotation = Quaternion.LookRotation(poseDot3d[26] - poseDot3d[28], forward) * Quaternion.Inverse(midRknee);
                        lankle.rotation = Quaternion.LookRotation(poseDot3d[27] - poseDot3d[29], forward) * Quaternion.Inverse(midLankle);
                        rankle.rotation = Quaternion.LookRotation(poseDot3d[28] - poseDot3d[30], forward) * Quaternion.Inverse(midLankle);
                        lheel.rotation = Quaternion.LookRotation(poseDot3d[29] - poseDot3d[31], poseDot3d[31] - poseDot3d[29]) * Quaternion.Inverse(midLheel);
                        rheel.rotation = Quaternion.LookRotation(poseDot3d[30] - poseDot3d[32], poseDot3d[32] - poseDot3d[30]) * Quaternion.Inverse(midRheel);

                    }
                    catch (Exception)
                    {
                        return;
                    }
                }
                else
                {
                    animator.runtimeAnimatorController = rac;
                }
            }
            catch (Exception)
            {
                return;
            }
        }       
    }        
}