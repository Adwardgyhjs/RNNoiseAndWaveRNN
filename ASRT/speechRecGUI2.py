import os
import tkinter
import wave
from tkinter import *
import tkinter.messagebox
import pyaudio
import winsound
import soundfile as sf
from tqdm import tqdm
import librosa
#import ASRT
#from ASRT import speech_model
import speech_recognition as sr
from ASRT.predict_speech_file import predict
from tkinter import filedialog, messagebox

recognizer = sr.Recognizer()

class srWindow:
    def __init__(self, win, ww, wh):
        self.win = win
        self.ww = ww
        self.wh = wh
        self.win.title("语音识别模块")
        self.win.geometry("%dx%d+%d+%d" % (ww, wh, 200, 50))
        self.img_src_path = None
        self.audio_path = None

        self.textlabe = Label(text="语音识别模块", fg="white", bg='black', font=("微软雅黑", 21))
        self.textlabe.place(x=420, y=35)

        self.text = Text(height=15, width=50)
        self.text.place(x=325, y=180)

        self.button1 = Button(self.win, text='加载音频', width=10, height=2, command=self.load)
        self.button1.place(x=325, y=450)

        self.button3 = Button(self.win, text='播放音频', width=10, height=2, command=self.play)
        self.button3.place(x=465, y=450)

        self.button2 = Button(self.win, text='开始识别', width=10, height=2, command=self.speechrec)
        self.button2.place(x=605, y=450)

    def play(self):
        if self.audio_path and os.path.exists(self.audio_path):
            try:
                winsound.PlaySound(self.audio_path, winsound.SND_FILENAME)
            except Exception as e:
                tkinter.messagebox.showerror("播放失败", str(e))
        else:
            tkinter.messagebox.showwarning("提示", "请先选择音频文件")

    def load(self):
        filepath = filedialog.askopenfilename(filetypes=[("音频文件", "*.wav *.flac")])
        if filepath:
            try:
                # 验证文件格式
                with sf.SoundFile(filepath) as f:
                    rate = f.samplerate
                    frames = f.frames
                    if rate!=16000:
                        print("采样率非16000hz，重采样")
                        y, orig_sr = librosa.load(filepath, sr=None)  # sr=None保持原始采样率

                        # 重采样到16000Hz
                        y_resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=16000)
                        
                        filepath="E:/智能语音处理系统/Noise-suppression-and-speech-recognition-systems-master/output_resample16k.wav"
                        # 保存结果
                        sf.write(filepath, y_resampled, 16000)
                
                self.audio_path = filepath
                self.text.insert('end', f"采样率：{rate}Hz\n总时长：{frames/rate:.2f}秒\n")
                
            except Exception as e:
                tkinter.messagebox.showerror("错误", f"不支持的音频格式：{str(e)}")

    def pyaudio(self):
        wave_out_path = "E:/audio/speech.wav"
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
        tkinter.messagebox.showinfo('提示', '存储成功')

    def speechrec(self):
        with sr.AudioFile(self.audio_path) as source:
            audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language='en-US')  # 支持多语言
            self.text.insert('end', f"识别结果：{text}\n")
        except sr.UnknownValueError:
            self.text.insert('end', f"识别失败：{str(e)}\n")
        except sr.RequestError as e:
            self.text.insert('end', f"识别失败：{str(e)}\n")


    #def speechrec(self):
        #res = predict()
        #self.text.insert('end',res)

if __name__ == '__main__':
    win = Tk()
    ww = 1000
    wh = 600
    img_gif = tkinter.PhotoImage(file="3.gif")
    label_img = tkinter.Label(win, image=img_gif, width="1000", height="600")
    label_img.place(x=0, y=0)
    srWindow(win, ww, wh)
    screenWidth, screenHeight = win.maxsize()
    geometryParam = '%dx%d+%d+%d' % (
        ww, wh, (screenWidth - ww) / 2, (screenHeight - wh) / 2)
    win.geometry(geometryParam)
    win.mainloop()
