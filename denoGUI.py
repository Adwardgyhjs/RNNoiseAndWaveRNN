import signal
import time
import tkinter
import wave
from tkinter import *
import tkinter.messagebox
import pyaudio
import winsound
from tqdm import tqdm
import librosa
from tkinter import filedialog, messagebox
import alg_denoise
import simpleaudio as sa
import alg_tts
from denoise import sigint_handler
import subprocess
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class denoWindow:
    def __init__(self, win, ww, wh):
        self.win = win
        self.ww = ww
        self.wh = wh
        self.win.title("语音降噪模块")
        self.win.geometry("%dx%d+%d+%d" % (ww, wh, 200, 50))
        self.img_src_path = None
        self.audio_data= None
        self.sample_rate=None
        self.filepath=None

        self.figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax1 = self.figure.add_subplot(221)  # 原始波形
        self.ax3 = self.figure.add_subplot(222)   # 原始频谱
        self.ax2 = self.figure.add_subplot(223)   # 降噪波形
        self.ax4 = self.figure.add_subplot(224)   # 降噪频谱
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.win)
        self.canvas.get_tk_widget().place(x=50, y=70, width=900, height=350)

        self.textlabe = Label(text="语音降噪模块", fg="white", bg='black', font=("微软雅黑", 21))
        self.textlabe.place(x=420, y=35)

        #self.button = Button(self.win, text='开始录音', width=10, height=2, command=self.start)
        #self.button.place(x=183, y=450)
        self.button=Button(self.win,text='加载带噪音频',width=10,height=2,command=self.load)
        self.button.place(x=300, y=450)

        #self.button1 = Button(self.win, text='加噪处理', width=10, height=2, command=self.noise)
        #self.button1.place(x=300, y=450)

        self.button2 = Button(self.win, text='降噪处理', width=10, height=2, command=self.denoise)
        self.button2.place(x=417, y=450)

        self.button3 = Button(self.win, text='播放带噪音频', width=10, height=2, command=self.playno)
        self.button3.place(x=534, y=450)


        self.button4 = Button(self.win, text='播放降噪音频', width=10, height=2, command=self.playdeno)
        self.button4.place(x=650, y=450)

    def load(self):
        # 弹出文件选择对话框
        filepath = filedialog.askopenfilename(
            title="选择带噪音频文件",
            filetypes=[("WAV 文件", "*.wav"), ("MP3 文件", "*.mp3"), ("所有文件", "*.*")]
        )
    
        # 如果用户取消选择
        if not filepath:
            return
        
        try:
            # 读取音频文件
            data, samplerate = sf.read(filepath)
            #assert samplerate == 48000, "需48kHz采样率"
            # 将音频数据和参数保存到实例变量中
            self.audio_data = data
            self.sample_rate = samplerate
            self.filepath = filepath
            
            self.plot_waveform(self.ax1, self.audio_data, "原始音频波形")
            self.plot_spectrum(self.ax3, self.audio_data, self.sample_rate, "spectrum")

            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"成功加载: {filepath.split('/')[-1]}")
                
                
        except Exception as e:
            tkinter.messagebox.showerror("加载错误", f"读取音频文件失败:\n{str(e)}")

        #input_audio, sr = sf.read('input.wav', dtype='int16')
        #assert sr == 48000, "需48kHz采样率"

        # 保存为RAW格式供RNNoise处理
        sf.write('input.raw', self.audio_data, self.sample_rate, subtype='PCM_16', format='RAW')
        tkinter.messagebox.showinfo('提示', '加载成功！')
    def plot_spectrum(self, ax, data, sample_rate, title):
        """绘制频谱图"""
        ax.clear()
        try:
            # 处理多声道数据
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # 截取前15秒数据
            max_samples = min(len(data), 15 * sample_rate)
            data_segment = data[:max_samples]
            
            # 计算STFT
            n_fft = 2048
            hop_length = 512
            D = librosa.stft(data_segment.astype(float), 
                            n_fft=n_fft, 
                            hop_length=hop_length)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            # 绘制频谱
            img = librosa.display.specshow(S_db, 
                                         sr=sample_rate,
                                         hop_length=hop_length,
                                         x_axis='time',
                                         y_axis='log',
                                         ax=ax,
                                         cmap='inferno')
            ax.set_title(title)
            self.figure.colorbar(img, ax=ax, format="%+2.0f dB")
            
        except Exception as e:
            print(f"频谱绘制错误: {str(e)}")

    def noise(self):
        global objDenoise, exit
        exit = False
        signal.signal(signal.SIGINT, sigint_handler)
        objDenoise = alg_denoise.espDenoise()
        objDenoise.initWiener()
        objDenoise.denoiseWiener("E:/audio/pyaudio.wav", "E:/audio/doing_the_dishes.wav",  "E:/audio/noise.wav") #加噪
        while exit is False:
            if objDenoise.getState() == -1:
                time.sleep(0.5)
                print("state runing, wait")
                continue
            else:
                break
        #print("going to exit")
        objDenoise.close()
        tkinter.messagebox.showinfo('提示', '存储成功！')
    """
    def start(self):
        wave_out_path = "E:\\智能语音处理系统\\Noise-suppression-and-speech-recognition-systems-master\\"
        record_second = 5
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        wf = wave.open(wave_out_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        print("* recording")
        for i in tqdm(range(0, int(RATE / CHUNK * record_second))):
            data = stream.read(CHUNK)
            wf.writeframes(data)
        print("* done recording")
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()
        tkinter.messagebox.showinfo('提示', '处理成功')
    """

    def denoise(self):
        #global objDenoise, exit
        #exit = False
        signal.signal(signal.SIGINT, sigint_handler)

        """
        objDenoise = alg_denoise.espDenoise()
        objDenoise.initWiener()
        objDenoise.denoiseWienerOneInput("E:/audio/pyaudio.wav", "E:/audio/denoise.wav")  # 减噪


        while exit is False:
            if objDenoise.getState() == -1:
                time.sleep(0.5)
                print("state runing, wait")
                continue
            else:
                break
        
        #print("going to exit")
        objDenoise.close()
        """

        subprocess.run([
            './rnnoise-master/VisualStudio2019/x64/Release/rnnoise_demo.exe',
            'input.raw',
            'output.raw'
        ])

      
        output_audio, _ = sf.read('output.raw', channels=1, samplerate=self.sample_rate, 
                                subtype='PCM_16', format='RAW')
        sf.write('output.wav', output_audio, self.sample_rate)
        #print("exit main program")

        self.plot_waveform(self.ax2, output_audio, "降噪后音频波形")
        self.plot_spectrum(self.ax4, output_audio, self.sample_rate, "spectrum")
        self.canvas.draw()

        tkinter.messagebox.showinfo('提示', '处理成功并保存于当前文件夹')

    def plot_waveform(self, ax, data, title):
        """绘制波形图"""
        ax.clear()
        """
        # 显示前15秒的波形
        max_samples = min(len(data), 15 * self.sample_rate)
        time = np.arange(max_samples) / self.sample_rate
        
        ax.plot(time, data[:max_samples], linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("时间 (秒)")
        ax.set_ylabel("振幅")
        ax.set_ylim(-1, 1) if data.dtype == np.float32 else ax.set_ylim(-32768, 32767)
        ax.grid(True)
        """

        duration = len(data) / self.sample_rate  # 计算音频时长
        num_channels = data.shape[1] if len(data.shape) > 1 else 1  # 获取声道数
        # 绘制振幅图
        #plt.figure(figsize=(12, 6))
        # 创建时间轴
        time = [i/self.sample_rate for i in range(len(data))]
        if num_channels == 1:
            ax.plot(time, data, linewidth=0.5)
            ax.set_title("Mono Audio Amplitude Waveform")
        else:
            # 如果是立体声，分开绘制左右声道
            ax.plot(time, data[:, 0], linewidth=0.5, label='Left Channel')
            ax.plot(time, data[:, 1], linewidth=0.5, label='Right Channel')
            #ax.title("Stereo Audio Amplitude Waveform")
            ax.legend()

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.5)
        ax.set_xlim(0, duration)
        #ax.tight_layout()
        #ax.show()


    def playdeno(self):
        filename = './output.wav'
        winsound.PlaySound(filename, winsound.SND_FILENAME)

    #def play(self):
        #filename = 'E:/audio/pyaudio.wav'
        #winsound.PlaySound(filename, winsound.SND_FILENAME)

    def playno(self):
        filename = self.filepath
        winsound.PlaySound(filename, winsound.SND_FILENAME)

if __name__ == '__main__':
    win = Tk()
    ww = 1000
    wh = 600
    img_gif = tkinter.PhotoImage(file="2.gif")
    label_img = tkinter.Label(win, image=img_gif, width="998", height="600")
    label_img.place(x=0, y=0)
    denoWindow(win, ww, wh)
    screenWidth, screenHeight = win.maxsize()
    geometryParam = '%dx%d+%d+%d' % (
        ww, wh, (screenWidth - ww) / 2, (screenHeight - wh) / 2)
    win.geometry(geometryParam)
    win.mainloop()
