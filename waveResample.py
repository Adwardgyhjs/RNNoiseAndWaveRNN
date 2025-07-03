import librosa
import soundfile as sf

# 读取音频文件（自动处理不同格式）
audio_path = 'E:\智能语音处理系统\Noise-suppression-and-speech-recognition-systems-master\WaveRNNModel\model_outputs\ljspeech_lsa_smooth_attention.tacotron'
y, orig_sr = librosa.load(audio_path, sr=None)  # sr=None保持原始采样率

# 重采样到16000Hz
y_resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=16000)

# 保存结果
sf.write('output_16k.wav', y_resampled, 16000)