import matplotlib.pyplot as plt
from scipy.io import wavfile

# 读取WAV文件
file_path = 'output.wav'  # 请替换为你的文件路径
sample_rate, audio_data = wavfile.read(file_path)

# 获取音频信息
duration = len(audio_data) / sample_rate  # 计算音频时长
num_channels = audio_data.shape[1] if len(audio_data.shape) > 1 else 1  # 获取声道数

print(f"采样率: {sample_rate}Hz")
print(f"音频时长: {duration:.2f}秒")
print(f"声道数: {num_channels}")

# 创建时间轴
time = [i/sample_rate for i in range(len(audio_data))]

# 绘制振幅图
plt.figure(figsize=(12, 6))

if num_channels == 1:
    plt.plot(time, audio_data, linewidth=0.5)
    plt.title("Mono Audio Amplitude Waveform")
else:
    # 如果是立体声，分开绘制左右声道
    plt.plot(time, audio_data[:, 0], linewidth=0.5, label='Left Channel')
    plt.plot(time, audio_data[:, 1], linewidth=0.5, label='Right Channel')
    plt.title("Stereo Audio Amplitude Waveform")
    plt.legend()

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(alpha=0.5)
plt.xlim(0, duration)
plt.tight_layout()
plt.show()