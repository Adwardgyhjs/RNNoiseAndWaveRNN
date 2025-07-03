import tkinter
from tkinter import *
from tkinter import ttk
import tkinter.messagebox
import os
import winsound
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import alg_tts
import sys
from pathlib import Path

# 添加项目根目录（project/）到 sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 导入 WaveRNN_master 包（触发其 __init__.py 中的路径设置）
import WaveRNNModel

import torch
from WaveRNNModel.gen_tacotron import gen_tacotron_from_inputtext
import numpy as np

class ttsWindow:
    def __init__(self, win, ww, wh):
        self.win = win
        self.ww = ww
        self.wh = wh
        self.win.title("语音合成模块")
        self.win.geometry("%dx%d+%d+%d" % (ww, wh, 200, 50))
        self.img_src_path = None
        self.save_path=None

        self.waveform_frame = ttk.Frame(self.win)
        self.waveform_frame.place(x=200, y=80, width=600, height=100)  # 调整位置

        self.textlabe = Label(text="语音合成模块", fg="white", bg='black', font=("微软雅黑", 21))
        self.textlabe.place(x=420, y=35)

        self.text = Text(height=15, width=50)
        self.text.place(x=320, y=200)

        #self.text2 = Text(height=15, width=50)
        #self.text2.place(x=520, y=200)

        self.button1 = Button(self.win, text='合成语音', width=10, height=2, command=self.tts)
        self.button1.place(x=335, y=450)

        #self.button2 = Button(self.win, text='存取合成音频', width=10, height=2, command=self.load)
        #self.button2.place(x=460, y=450)

        #self.button2 = Button(self.win, text='存取1号音频', width=10, height=2, command=self.loadone)
        #self.button2.place(x=460, y=450)

        #self.button2 = Button(self.win, text='存取2号音频', width=10, height=2, command=self.loadtwo)
        #self.button2.place(x=585, y=450)

        self.button2 = Button(self.win, text='播放合成音频', width=10, height=2, command=self.play)
        self.button2.place(x=585, y=450)

    def tts(self):
        #alg_tts.init()
        #self.string1 = self.text.get('0.0', 'end')
        #self.string2 = self.text2.get('0.0', 'end')
        #alg_tts.ttsSpeak(self.string1)
        #alg_tts.ttsSpeak(self.string2)
        input_text = self.text.get('0.0', 'end')
        hp_file="E:\\智能语音处理系统\\Noise-suppression-and-speech-recognition-systems-master\\WaveRNNModel\\hparams.py"
        custom_args = [
            "--input_text", input_text,
            "--hp_file",hp_file,
            "wavernn", 
        ]
        print(custom_args)
        self.save_path=gen_tacotron_from_inputtext(args_list=custom_args)
        if self.save_path:
            self.show_waveform(self.save_path)
        tkinter.messagebox.showinfo('提示','成功生成语音')

    """
    def loadone(self):
        alg_tts.init()
        self.string1 = self.text.get('0.0', 'end')
        alg_tts.ttsSaveToFile(self.string1, "E:/audio/tts.wav")
        tkinter.messagebox.showinfo('提示','存储成功')

    def loadtwo(self):
        alg_tts.init()
        self.string1 = self.text2.get('0.0', 'end')
        alg_tts.ttsSaveToFile(self.string1, "E:/audio/tts2.wav")
        tkinter.messagebox.showinfo('提示','存储成功')

    """

    #def load(self):
        

    def play(self):
        filename = str(self.save_path)
        print(type(filename))
        winsound.PlaySound(filename, winsound.SND_FILENAME)
    def show_waveform(self, save_path):
        # 清除旧内容
        for widget in self.waveform_frame.winfo_children():
            widget.destroy()
        
        # 读取音频
        try:
            sample_rate, data = wavfile.read(save_path)
            # 处理立体声
            if len(data.shape) > 1:
                data = data[:, 0]
        except Exception as e:
            tkinter.messagebox.showerror("错误", f"读取音频失败: {str(e)}")
            return

        # 绘制波形
        fig = plt.figure(figsize=(10, 2), dpi=50)
        plt.plot(data, color='blue', linewidth=0.5)
        plt.axis('off')
        plt.tight_layout()

        # 嵌入到界面
        canvas = FigureCanvasTkAgg(fig, master=self.waveform_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)


if __name__ == '__main__':
    win = Tk()
    ww = 1000
    wh = 600
    img_gif = tkinter.PhotoImage(file="4.gif")
    label_img = tkinter.Label(win, image=img_gif, width="1000", height="600")
    label_img.place(x=0, y=0)
    ttsWindow(win, ww, wh)
    screenWidth, screenHeight = win.maxsize()
    geometryParam = '%dx%d+%d+%d' % (
        ww, wh, (screenWidth - ww) / 2, (screenHeight - wh) / 2)
    win.geometry(geometryParam)
    win.mainloop()
