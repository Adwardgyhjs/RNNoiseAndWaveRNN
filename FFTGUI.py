import signal
import time
import tkinter
import wave
from tkinter import *
import tkinter.messagebox
import matplotlib.pyplot as plt
import librosa
import numpy as np
import pyaudio
import winsound
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.io import wavfile
from tqdm import tqdm
import soundfile as sf
from tkinter import filedialog, messagebox

import alg_denoise
import alg_tts
from denoise import sigint_handler


class FFTWindow:
    def __init__(self, win, ww, wh):
        self.fig = fig = Figure(figsize=(4, 3), dpi=80)
        self.win = win
        self.ww = ww
        self.wh = wh
        self.win.title("FFT降噪模块")
        self.win.geometry("%dx%d+%d+%d" % (ww, wh, 200, 50))
        self.img_src_path = None
        self.audio_data = None
        self.sample_rate = None
        self.outrate=None
        self.filepath = None
        self.start_time=0
        self.end_time=20
        self.outputfilepath='E:/智能语音处理系统/Noise-suppression-and-speech-recognition-systems-master/output.wav'

        self.figure = plt.Figure(figsize=(8, 4), dpi=100)
        self.ax1 = self.figure.add_subplot(211)  # 原始波形
        #self.ax3 = self.figure.add_subplot(222)  # 原始频谱
        self.ax2 = self.figure.add_subplot(212)  # 降噪波形
        #self.ax4 = self.figure.add_subplot(224)  # 降噪频谱
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.win)
        self.canvas.get_tk_widget().place(x=50, y=70, width=900, height=350)

        self.textlabe = Label(text="FFT降噪模块", fg="white", bg='black', font=("微软雅黑", 21))
        self.textlabe.place(x=420, y=35)

        #self.canvas = FigureCanvasTkAgg(fig, win)
        #self.canvas.get_tk_widget().place(x=140, y=130)

        #self.canvas2 = FigureCanvasTkAgg(fig, win)
        #self.canvas2.get_tk_widget().place(x=560, y=130)

        #self.button = Button(self.win, text='开始录音', width=10, height=2, command=self.start)
        #self.button.place(x=183, y=450)
        self.button=Button(self.win,text='加载带噪音频',width=10,height=2,command=self.load)
        self.button.place(x=300, y=450)

        #self.button1 = Button(self.win, text='添加白噪', width=10, height=2, command=self.noise)
        #self.button1.place(x=300, y=450)

        self.button2 = Button(self.win, text='FFT降噪', width=10, height=2, command=self.denoise)
        self.button2.place(x=417, y=450)

        #self.button3 = Button(self.win, text='播放原音频', width=10, height=2, command=self.play)
        #self.button3.place(x=534, y=450)

        self.button4 = Button(self.win, text='播放降噪音频', width=10, height=2, command=self.playdeno)
        self.button4.place(x=650, y=450)

        self.button5 = Button(self.win, text='播放带噪音频', width=10, height=2, command=self.playno)
        self.button5.place(x=763, y=450)

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
            if int(len(data)/samplerate)<self.end_time:
                self.end_time=int(len(data)/samplerate)
            plot_times=[self.start_time,self.end_time]
            print("plot_times",plot_times)
            self.plot_waveform(self.ax1, self.audio_data, "原始音频波形",plot_times)
           # self.plot_spectrum(self.ax3, self.audio_data, self.sample_rate, "spectrum", plot_times)

            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"成功加载: {filepath.split('/')[-1]}")
                
                
        except Exception as e:
            tkinter.messagebox.showerror("加载错误", f"读取音频文件失败:\n{str(e)}")

        #input_audio, sr = sf.read('input.wav', dtype='int16')
        #assert sr == 48000, "需48kHz采样率"

        # 保存为RAW格式供RNNoise处理
        sf.write('input.raw', self.audio_data, self.sample_rate, subtype='PCM_16', format='RAW')
        tkinter.messagebox.showinfo('提示', '加载成功！')

    def plot_spectrum(self, ax, data, sample_rate, title, plottime):
        """绘制频谱图"""
        ax.clear()
        
        try:
            # 处理多声道数据
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # 截取时间段
            start = int(plottime[0] * sample_rate)
            end = int(plottime[1] * sample_rate)
            data_segment = data[start:end]
            
            # 计算STFT
            n_fft = 2048
            hop_length = 512
            D = librosa.stft(data_segment.astype(float), n_fft=n_fft, hop_length=hop_length)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            # 绘制频谱
            img = librosa.display.specshow(S_db, sr=sample_rate,
                                         hop_length=hop_length,
                                         x_axis='time', y_axis='log',
                                         ax=ax, cmap='viridis')
            ax.set_title(title)
            self.figure.colorbar(img, ax=ax, format="%+2.0f dB")
            
        except Exception as e:
            print(f"绘制频谱图出错: {str(e)}")

    def plot_waveform(self, ax, data, title,plottime):
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
        plotdurtime=plottime[1]-plottime[0]
        time = [i/self.sample_rate for i in range(plottime[0]*self.sample_rate,plottime[1]*self.sample_rate)]
        #if flag==1:
            #duration=len(data) / self.outrate
            #time=[i/self.sample_rate for i in range(len(data))]
            #print("flag 1")
        print("test3",len(data))
        print("test4",duration)
        num_channels = data.shape[1] if len(data.shape) > 1 else 1  # 获取声道数
        # 绘制振幅图
        #plt.figure(figsize=(12, 6))
        # 创建时间轴
        
        if num_channels == 1:
            ax.plot(time, data[plottime[0]*self.sample_rate:plottime[1]*self.sample_rate], linewidth=0.5)
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
        ax.set_xlim(plottime[0], plottime[1])
        #ax.tight_layout()
        #ax.show()

    def noise(self):
        rate, data = wavfile.read(r'E:/audio/FFTpyaudio.wav')
        noise = np.random.normal(0, 500, data.shape)
        newaudio = data + noise
        wavfile.write("E:/audio/FFTnoise.wav", rate, newaudio.astype(np.int16))
        tkinter.messagebox.showinfo('提示', '存储成功')
        #self.clear2()
        axc = self.fig.add_subplot(111)
        axc.set_xticks([])
        axc.set_yticks([])
        axc.plot(newaudio, 'g')
        self.canvas2.draw()

    def start(self):
        wave_out_path = "E:/audio/FFTpyaudio.wav"
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
        tkinter.messagebox.showinfo('提示', '录制成功')
        #self.clear1()
        rate, data = wavfile.read(r'E:/audio/FFTpyaudio.wav')
        axc = self.fig.add_subplot(111)
        axc.set_xticks([])
        axc.set_yticks([])
        axc.plot(data)
        self.canvas.draw()

    def denoise(self):
        file = self.filepath
        sig, fs = librosa.load(file, sr=self.sample_rate)

        # Filtering
        dt = 1 / fs
        t = np.arange(self.start_time, self.end_time, dt)
        n = len(t)

        fhat = np.fft.fft(sig, n)  # Compute the FFT
        PSD = fhat * np.conj(fhat) / n  # Power spectrum (power per freq)
        freq = (1 / (dt * n)) * np.arange(n)  # Create x-axis of frequencies in Hz
        L = np.arange(1, np.floor(n / 20), dtype='int')
        indices = PSD > 0.1  # Find all freqs with large power
        PSDclean = PSD * indices  # Zero out all others
        fhat = indices * fhat  # Zero out small Fourier coeffs.
        ffilt = np.fft.ifft(fhat)

        nsig = ffilt.real
        abs_spectrogram = np.abs(librosa.stft(nsig))
        audio_signal = librosa.griffinlim(abs_spectrogram)
        sf.write(self.outputfilepath, audio_signal, fs)
        tkinter.messagebox.showinfo('提示', '存储成功')


        #output_audio, _ = sf.read(self.outputfilepath, channels=1, samplerate=self.sample_rate, 
                                #subtype='PCM_16', format='RAW')
        #sf.write('output.wav', output_audio, self.sample_rate)

        rate, data = wavfile.read(self.outputfilepath)
        print("rate=",rate)
        print("self sample=",self.sample_rate)
        #self.outrate=rate
        print("test1:",len(data))
        data.astype(data.dtype).tofile("output.raw")
        print("test2:",len(data))
        output_audio, _ = sf.read("output.raw", channels=1, samplerate=self.sample_rate, 
                                subtype='PCM_16', format='RAW')
        plot_times=[self.start_time,self.end_time]
        print("plot_times",plot_times)
        self.plot_waveform(self.ax2, output_audio, "降噪后音频波形",plot_times)
        #self.plot_spectrum(self.ax4, output_audio, self.sample_rate, "spectrum", plot_times)
        self.canvas.draw()

        #axc = self.fig.add_subplot(111)
        #axc.set_xticks([])
       # axc.set_yticks([])
        #axc.plot(data)
        #self.canvas.draw()


    def playdeno(self):
        filename = self.outputfilepath
        winsound.PlaySound(filename, winsound.SND_FILENAME)

    def play(self):
        filename = 'E:/audio/FFTpyaudio.wav'
        winsound.PlaySound(filename, winsound.SND_FILENAME)

    def playno(self):
        filename = self.filepath
        winsound.PlaySound(filename, winsound.SND_FILENAME)

    def clear1(self):
        self.canvas.get_tk_widget().delete(tkinter.ALL)

    def clear2(self):
        self.canvas2.get_tk_widget().delete(tkinter.ALL)


if __name__ == '__main__':
    win = Tk()
    ww = 1000
    wh = 600
    img_gif = tkinter.PhotoImage(file="5.gif")
    label_img = tkinter.Label(win, image=img_gif, width="1000", height="600")
    label_img.place(x=0, y=0)
    FFTWindow(win, ww, wh)
    screenWidth, screenHeight = win.maxsize()
    geometryParam = '%dx%d+%d+%d' % (
        ww, wh, (screenWidth - ww) / 2, (screenHeight - wh) / 2)
    win.geometry(geometryParam)
    win.mainloop()