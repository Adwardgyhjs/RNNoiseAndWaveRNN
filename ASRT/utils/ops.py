#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2016-2099 Ailemon.net
#
# This file is part of ASRT Speech Recognition Tool.
#
# ASRT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# ASRT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASRT.  If not, see <https://www.gnu.org/licenses/>.
# ============================================================================

"""
@author: nl8590687
一些常用操作函数的定义
"""

import wave
import difflib
import numpy as np

def open_wav_file(filename):
    wav = None
    try:
        # 尝试打开文件
        wav = wave.open(filename, "rb")
        
        # 检查基础参数
        required_params = {
            "nchannels": 1,     # 期望单声道
            "sampwidth": 2,     # 期望16位采样（2字节）
            "framerate": 16000  # 期望16kHz采样率
        }
        
        # 验证参数
        if wav.getnchannels() != required_params["nchannels"]:
            raise ValueError(f"要求单声道，当前声道数：{wav.getnchannels()}")
            
        if wav.getsampwidth() != required_params["sampwidth"]:
            raise ValueError(f"要求16-bit采样，当前采样位宽：{wav.getsampwidth()*8}-bit")
            
        if wav.getframerate() != required_params["framerate"]:
            raise ValueError(f"要求16kHz采样率，当前采样率：{wav.getframerate()}Hz")
            
        # 检查有效帧数
        if wav.getnframes() == 0:
            raise ValueError("音频文件为空（0帧）")
            
        return wav
        
    except FileNotFoundError:
        print(f"错误：文件不存在 '{filename}'")
    except PermissionError:
        print(f"错误：没有权限读取文件 '{filename}'")
    except wave.Error as e:
        print(f"WAV格式错误：{str(e)}")
        print("可能原因：文件头损坏/非WAV格式/压缩编码")
    except ValueError as ve:
        print(f"参数验证失败：{str(ve)}")
    except Exception as e:
        print(f"未知错误：{str(e)}")
    finally:
        if wav is not None:
            try:
                wav.close()
            except:
                pass
    return None

def read_wav_data(filename: str) -> tuple:
    '''
    读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
    '''
    print("filename==",filename)
    #wav = open_wav_file(filename)
    wav = wave.open(filename,"rb") # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes() # 获取帧数
    print("num_frame",num_frame)
    num_channel=wav.getnchannels() # 获取声道数
    framerate=wav.getframerate() # 获取帧速率
    num_sample_width=wav.getsampwidth() # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame) # 读取全部的帧
    wav.close() # 关闭流
    wave_data = np.fromstring(str_data, dtype = np.short) # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, num_channel # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T # 将矩阵转置
    return wave_data, framerate, num_channel, num_sample_width


def get_edit_distance(str1, str2) -> int:
    '''
    计算两个串的编辑距离，支持str和list类型
    '''
    leven_cost = 0
    sequence_match = difflib.SequenceMatcher(None, str1, str2)
    for tag, index_1, index_2, index_j1, index_j2 in sequence_match.get_opcodes():
        if tag == 'replace':
            leven_cost += max(index_2-index_1, index_j2-index_j1)
        elif tag == 'insert':
            leven_cost += (index_j2-index_j1)
        elif tag == 'delete':
            leven_cost += (index_2-index_1)
    return leven_cost
