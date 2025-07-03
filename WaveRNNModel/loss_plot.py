import matplotlib.pyplot as plt
import re
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 简体中文（根据系统调整）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 从txt文件读取日志数据
def parse_log_file(file_path):
    epochs = []
    losses = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 使用正则匹配有效行
            match = re.search(
                r'Epoch:\s+(\d+)/*.*Loss:\s+(\d+\.\d+)',
                line.strip()
            )
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epochs.append(epoch)
                losses.append(loss)
    
    return epochs, losses

# 文件路径
log_file = "E:\\智能语音处理系统\\Noise-suppression-and-speech-recognition-systems-master\\WaveRNNModel\\checkpoints\\ljspeech_lsa_smooth_attention.tacotron\\log_test.txt"

# 提取数据
try:
    epochs_read, losses = parse_log_file(log_file)
    print(epochs_read)
    epochs=np.arange(len(epochs_read))
    print(epochs)
except FileNotFoundError:
    print(f"错误：文件 {log_file} 不存在，请检查路径！")
    exit()
except Exception as e:
    print(f"解析文件时出错: {str(e)}")
    exit()

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, 'b-', linewidth=2, label='训练损失')

# 图表美化
plt.title('训练损失随轮次变化曲线', fontsize=14)
plt.xlabel('训练轮次 (Epoch)', fontsize=12)
plt.ylabel('损失值 (Loss)', fontsize=12)
#plt.xticks(range(1, len(epochs)))  # 强制显示所有epoch刻度
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 标注最低损失
min_loss = min(losses)
min_idx = losses.index(min_loss)
plt.annotate(
    f'最低损失: {min_loss:.3f}',
    xy=(epochs[min_idx], min_loss),
    xytext=(epochs[min_idx]-3, min_loss+0.1),
    arrowprops=dict(arrowstyle='->', color='red'),
    fontsize=10,
    color='red'
)

plt.tight_layout()
plt.show()