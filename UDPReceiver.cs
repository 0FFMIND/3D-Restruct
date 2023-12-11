using UnityEngine;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
// 前面的using语句是导入UDPReceiver脚本必要的命名空间，使得函数引用可以正常进行
public class UDPReceiver : MonoBehaviour
{
    int port = 5053; // 这里用本机作为发收器，进行发收的端口为5053，数据收发采用同一端口
    public string data; // 接收到的数据是用99个double变量构成的string

    void Start()
    {
        Thread receiveThread = new Thread(
            new ThreadStart(ReceiveData)); // 初始化，构建ThreadStart委托，并告诉其调用ReceiveData函数
        receiveThread.IsBackground = true; // 将此线程设为后台线程
        receiveThread.Start(); // 开启线程中ReceiveData中的while循环

    }
    void ReceiveData()
    {
        using (UdpClient client = new UdpClient(port)) // 使用using语句可以在client不再需要时自动调用其Dispose方法，释放资源
        {
            while (true)
            {
                try
                {
                    IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0); // 表示client会接收任何IP传入本机的数据
                    byte[] dataByte = client.Receive(ref anyIP); // 若client接收到传入的数据，即将该数据存在byte[]数组里面
                    data = Encoding.UTF8.GetString(dataByte); // 将byte数组进行解码后转为string，并最终赋值给data变量
                }
                catch (ThreadAbortException)
                {
                    return; // 如果捕获到ThreadAbortException，那么结束方法执行，结束线程
                }
                catch (Exception err)
                {
                    print(err.ToString()); // 若捕获到其他异常则打印到控制台，但不干扰while循环的进行
                }
            }
        }
    }

}