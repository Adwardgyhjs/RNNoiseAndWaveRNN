 **首先声明 本系统语音识别模块采用了AI柠檬博主的开源项目 ASRT 进行语音识别
ASRT 项目地址 ：https://gitee.com/ailemon/ASRT_SpeechRecognition?_from=gitee_search
希望大家可以向ASRT作者点star** 

 只需将项目git至本地，运行ASRT文件下的GUI.py即可
 
以下为系统说明书
一、系统设计目标
根据实训安排，设计并实现噪声抑制实验与语音合成系统。需要充分考虑用户体验，要求界面简洁明朗，操作简单快捷，功能实用，内容丰富。其中，噪声抑制系统支持录入使用者的音频。为达到显著效果，会对音频先进性加噪而进行降噪处理。语音合成系统可将用户输入的文字朗诵并转换成音频文件，且提供两段文字的合成转换。
二、问题描述
1、在语音合成技术中，主要分为语言分析部分和声学系统部分，也称为前端部分和后端部分，语言分析部分主要是根据输入的文字信息进行分析，生成对应的语言学规格书，想好该怎么读；声学系统部分主要是根据语音分析部分提供的语音学规格书，生成对应的音频，实现发声的功能。
2、在图像的生成、传输过程中，不可避免会受到噪声的干扰，而且有些图像噪声非常严重，图像中的噪声往往和信号交织在一起，会使图像本身的细节如边界轮廓、线条等变得模糊不清。我们需要对图像进行降噪处理，便于更高层次的图像分析与理解。如何既对图像中噪声进行合理的抑制、衰减以及去除不需要的信息，又能使有用的信息得到加强，从而便于目标或对象解释，是去噪研究的主要任务。
三、需求分析
1、噪声抑制系统：
（1）提供对音频进行转换成频谱图，让用户分析起来更加直观。
（2）可以进行录音，并将录进的音频内容保存到指定的文件夹中。
（3）同时可以对录进的音频进行添加噪音，添加研究内容。
（4）可以对录制的音频进行降噪处理，使录进的音频听起来更加清晰。
（5）可以将处理后的音频播放出来。
2、语音合成系统：
（1）提供对用户想输入的两段文字的编辑功能。
（2）可以将用户输入的两段文字合成为一段文字并将文字转换为音频信息。
（3）可以存取转换后的音频，也可以存取任意一段文字所转换的音频。
（4）可以将存取的音频播放出来
3、语音识别系统：
（1）提供对于用户输入的音频进行录制。
（2）可以将用户录入的音频存储到指定路径中并进行播放。
（3）将存储的音频识别成文字的方式进行输出。 

以下为系统使用说明

一、主界面
	1，当用户进入主界面时，主界面如图所示，我们的题目为噪声抑制实验与语音合成系统，主界面标题为语音识别系统，我们的功能主要围绕语音合成和降噪而展开，主界面有四个按钮，分别为语音识别、FFT降噪、语音降噪和语音合成。点击按钮后，会触发监听事件，并分别进入相应的界面。
 

二、语音识别
当用户点击语音识别按钮后，程序会自动进入语音识别界面，如图所示，此界面共有三个按钮，分别为开始录音、播放录音和开始识别，当用户点击开始录音后，程序会自动将音频文件自动存储在audio文件夹中，存储完成后，会自动跳出存储成功界面。再次点击播放语音，程序会将存储在audio文件夹中的音频文件播放出来。当用户点击开始识别，程序会将存储的音频文件输出为文字，显示在界面中，如图所示。
 

三、FFT降噪
当用户点击FFT降噪按钮后，程序会自动进入FFT降噪界面，如图所示，界面共有六个按钮，分别为开始录音、添加白噪、FFT降噪、播放原音频、播放降噪音频、播放加噪音频。此界面主要添加了两个主要功能，分别为添加白噪和FFT降噪，录音后，点击添加白噪，程序会在原音频中添加幅度分布服从高斯分布、而功率密度又是均匀分布的高斯白噪声。添加高斯白噪后频谱图如图所示，点击播放加噪音频后，会自动播放加噪完成后的音频。点击FFT降噪后，程序将使用傅里叶变换进行图像去噪。FFT降噪频谱图如图所示，点击播放降噪音频后，会自动播放降噪完成后的音频。
  
四、语音降噪
当用户点击语音降噪后，程序会自动进入语音降噪界面，如图所示，当用户点击开始录音后，会将音频文件存储到audio文件夹中，点击播放原音频后，程序会自动将存储在audio文件夹中的音频文件播放出来。点击加噪处理后，程序会将原音频中加入噪声，点击播放加噪音频会将经过加噪处理的音频播放出来。点击降噪处理后，程序会使用维纳滤波器，将信号与噪声信号分离，以达到降噪效果，点击播放降噪音频后会将经过降噪处理的音频播放出来。

 

五、语音合成
当用户在主界面点击语音合成按钮后，程序会自动进入语音合成界面，界面如图所示，语音合成界面由两个文字框、五个按钮组成，五个按钮分别为合成语音、存取合成音频、存取1号音频、存取2号音频、播放合成音频组成。在两个文字框分别输入文字后，点击合成语音按钮，会将两段文字合称为一段文字并转换为音频。点击播放合成音频后，会将音频播放出来，点击存取合成音频后，会将合成的音频存取起来，存取成功后，会自动弹出存取成功界面，如图所示，点击存取1号音频会将左边文字框中文字转换的音频存取起来。点击存取2号音频会将右边文字框中文字框中文字转换的音频存取起来。



