import matplotlib.pyplot as plt
import numpy as np

# 原始数据
bandwidths = [10, 5, 2, 1.6, 1.2, 1, 0.5]
time_delays = [0.002, 0.005, 0.008, 0.012, 0.014, 0.017, 0.027]
minus = [1 - 0.02, 1 - 0.05, 1 - 0.08, 1 - 0.12-(300-242)/300,
         1 - 0.14 - (300-182)/300, 1 - 0.17 - (300-152)/300,
         1 - 0.27 - (300-91)/300]

# 计算 x 位置，使得数据点可以在图上均匀分布
x = np.arange(len(bandwidths))  # [0, 1, 2, ..., len(bandwidths)-1]

fig, ax1 = plt.subplots()

# 柱状图
color = 'tab:blue'
ax1.set_xlabel('Bandwidths(Mb/s)')
ax1.set_ylabel('Metaverse User Experience(100%)', color='blue')
bars = ax1.bar(x, minus, width=0.7*0.8,  color='blue', alpha=0.9, label='MUE')
ax1.set_xticks(x)
ax1.set_xticklabels(bandwidths)  # 将 x 位置标签设置为原始 bandwidths 数值
ax1.tick_params(axis='y', labelcolor='blue')
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom', color='black')

# 添加第二个 y 轴用于线状图
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Network Time-Delays(s)', color='red')
ax2.plot(x, time_delays, label='Time Delays', color='red', marker='o')
ax2.tick_params(axis='y', labelcolor='red')

# 添加图例
fig.tight_layout()  # 调用 tight_layout 以防止标签之间的重叠

# 图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

plt.title('Network Delay and MUE for Different Bandwidth.', pad=20)

# 显示网格
plt.grid()

# 展示图形
plt.show()