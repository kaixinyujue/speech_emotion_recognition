import os,sys
os.chdir(sys.path[0])

import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tkinter import *
from tkinter.filedialog import *
from PIL import ImageTk
from PIL import Image
from pygame import mixer
from aip import AipSpeech
from creat_utils.features import *
from creat_utils.loader import *


scaler_path = 'ESD/standard.m'
label_list_path = 'ESD/label_list.txt'
device_type = 'cpu'
model_path = 'output/inference/inference.pth'
back0_path = 'sources/back0.jpg'
back_path = 'sources/back.jpg'


# 加载归一化文件
scaler = joblib.load(scaler_path)
# 获取分类标签
with open(label_list_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
class_labels = [l.replace('\n', '') for l in lines]
# 获取模型
device = torch.device(device_type)
model = torch.load(model_path)
model.to(device)
model.eval()

APP_ID = '33091558'
API_KEY = 's3Abr6RMX3krVhWpfoRseMl1'
SECRET_KEY = 'jq6PlXFBk3FjgqpkWvBDrdAECAXFNguF'
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


def inference(audio_path):
    audio = load_audio(audio_path, mode='infer')
    audio = audio[np.newaxis, :]
    audio = scaler.transform(audio)
    audio = torch.tensor(audio, dtype=torch.float32, device=device)
    # 执行预测
    output = model(audio).data.cpu().numpy()[0]
    # 显示图片并输出结果最大的label
    lab0 = np.argsort(output)[-1]
    label = class_labels[lab0]
    score = output[lab0]
    return label, score

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def recognize(audio_file):
    result = client.asr(get_file_content(audio_file), 'wav', 16000, {
        'dev_pid': 1537,
    })
    return result

def normal_figure(wav_path):
    x, sr = librosa.load(wav_path)
    #绘制声波信号
    plt.figure(figsize=(6, 2))
    librosa.display.waveshow(x, sr=sr)
    # plt.show()
    plt.savefig("sources/wav.jpg")

def mfcc_save(wav_path):
    # STFT
    y, sr = librosa.load(wav_path)
    S = librosa.stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='reflect')

    # 取绝对值
    S = np.abs(S)
    print(y.shape)
    print(S.shape)

    #梅尔频率倒谱系数 MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    print(mfccs.shape)

    # MFCC:
    plt.figure(figsize=(6, 2))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.savefig("sources/mfcc.jpg")


bg_weight = 1280
bg_height = 800
root = None
cover = None
app = None

class tkCover(Frame):
    ww = bg_weight
    wh = bg_height
    
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.place(width=bg_weight,height=bg_height)
        self.createWidget()

    def get_img(self, filename, width, height):
        im = Image.open(filename).resize((width, height))
        im = ImageTk.PhotoImage(im)
        return im

    def createWidget(self):
        # 设置背景图片
        self.canvas_root = Canvas(self, width=self.ww, height=self.wh)
        self.im_root0 = self.get_img(back0_path, self.ww, self.wh)
        self.canvas_root.create_image(self.ww/2, self.wh/2, image=self.im_root0)
        self.canvas_root.pack()

        # 画布文字
        self.label1t0 = self.canvas_root.create_text(640+2,260+2,text='语音情感文本识别应用',font=('楷体',40,'bold'),fill='gray')
        self.label1t = self.canvas_root.create_text(640,260,text='语音情感文本识别应用',font=('楷体',40,'bold'),fill='black')

        # 用画布仿制透明按钮
        self.bt1_left = 400 ; self.bt1_top = 550; self.bt1_right = 880; self.bt1_bottom = 600
        self.bt1r1 = self.canvas_root.create_rectangle(self.bt1_left,self.bt1_top,self.bt1_right,self.bt1_bottom,width=2,outline='black')# 按钮外框
        self.bt1r2 = self.canvas_root.create_rectangle(self.bt1_left+3,self.bt1_top+3,self.bt1_right-3,self.bt1_bottom-3,width=2,outline='black')# 按钮内框
        self.bt1t = self.canvas_root.create_text((self.bt1_left+self.bt1_right)/2,(self.bt1_top+self.bt1_bottom)/2,text='开 始',font=('楷体',20,'bold'),fill='black')# 按钮显示文本
        
        self.canvas_root.bind('<Button-1>',lambda event:bind_click(event))# 关联鼠标点击事件
        self.canvas_root.bind('<Motion>',lambda event:motion_all(event))# 关联鼠标经过事件

        def bind_click(event):# 点击响应函数
            if self.bt1_left<=event.x<=self.bt1_right and self.bt1_top<=event.y<=self.bt1_bottom:# 响应的位置
                fluent_change()
            
        def motion_all(event):# 鼠标经过响应函数
            if self.bt1_left<=event.x<=self.bt1_right and self.bt1_top<=event.y<=self.bt1_bottom:# 响应的位置
                self.canvas_root.itemconfigure(self.bt1r1,outline='white')# 重设外框颜色
                self.canvas_root.itemconfigure(self.bt1r2,outline='white')# 重设内框颜色
                self.canvas_root.itemconfigure(self.bt1t,fill='white')# 重设显示文本颜色
            else:
                self.canvas_root.itemconfigure(self.bt1r1,outline='black')# 恢复外框默认颜色
                self.canvas_root.itemconfigure(self.bt1r2,outline='black')# 恢复内框默认颜色
                self.canvas_root.itemconfigure(self.bt1t,fill='black')# 恢复显示文本默认颜色


class tkApplication(Frame):
    ww = bg_weight
    wh = bg_height
    radio_path = None
    play_status = False

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.place(width=bg_weight,height=bg_height,x=bg_weight)
        self.createWidget()

    def get_img(self, filename, width, height):
        im = Image.open(filename).resize((width, height))
        im = ImageTk.PhotoImage(im)
        return im

    def createWidget(self):
        ### app中所有组件必须定义为app的属性
        
        # 设置背景图片
        self.canvas_root = Canvas(self, width=self.ww, height=self.wh)
        self.im_root0 = self.get_img(back_path, self.ww, self.wh)
        self.canvas_root.create_image(self.ww/2, self.wh/2, image=self.im_root0)
        self.canvas_root.pack()

        # 用画布仿制透明按钮
        self.bt1_left = 670 ; self.bt1_top = 120; self.bt1_right = 720; self.bt1_bottom = 150
        self.bt1r1 = self.canvas_root.create_rectangle(self.bt1_left,self.bt1_top,self.bt1_right,self.bt1_bottom,width=1,outline='gray')# 按钮外框
        self.bt1r2 = self.canvas_root.create_rectangle(self.bt1_left+3,self.bt1_top+3,self.bt1_right-3,self.bt1_bottom-3,width=1,outline='gray')# 按钮内框
        self.bt1t = self.canvas_root.create_text((self.bt1_left+self.bt1_right)/2,(self.bt1_top+self.bt1_bottom)/2,text='Load',font=('楷体',10,'bold'),fill='gray')# 按钮显示文本
        
        self.bt2_left = 900 ; self.bt2_top = 240; self.bt2_right = 1060; self.bt2_bottom = 290
        self.bt2r1 = self.canvas_root.create_rectangle(self.bt2_left,self.bt2_top,self.bt2_right,self.bt2_bottom,width=2,outline='white')# 按钮外框
        self.bt2r2 = self.canvas_root.create_rectangle(self.bt2_left+3,self.bt2_top+3,self.bt2_right-3,self.bt2_bottom-3,width=2,outline='white')# 按钮内框
        self.bt2t = self.canvas_root.create_text((self.bt2_left+self.bt2_right)/2,(self.bt2_top+self.bt2_bottom)/2,text='识 别',font=('楷体',20,'bold'),fill='white')# 按钮显示文本

        self.bt3_left = 900 ; self.bt3_top = 120; self.bt3_right = 1060; self.bt3_bottom = 170
        self.bt3r1 = self.canvas_root.create_rectangle(self.bt3_left,self.bt3_top,self.bt3_right,self.bt3_bottom,width=2,outline='white')# 按钮外框
        self.bt3r2 = self.canvas_root.create_rectangle(self.bt3_left+3,self.bt3_top+3,self.bt3_right-3,self.bt3_bottom-3,width=2,outline='white')# 按钮内框
        self.bt3t = self.canvas_root.create_text((self.bt3_left+self.bt3_right)/2,(self.bt3_top+self.bt3_bottom)/2,text='播 放',font=('楷体',20,'bold'),fill='white')# 按钮显示文本
        
        self.canvas_root.bind('<Button-1>',lambda event:bind_click(event))# 关联鼠标点击事件
        self.canvas_root.bind('<Motion>',lambda event:motion_all(event))# 关联鼠标经过事件

        def bind_click(event):# 点击响应函数
            if self.bt1_left<=event.x<=self.bt1_right and self.bt1_top<=event.y<=self.bt1_bottom:# 响应的位置
                self.load_disp()
            if self.bt2_left<=event.x<=self.bt2_right and self.bt2_top<=event.y<=self.bt2_bottom:# 响应的位置
                self.run_out()
            if self.bt3_left<=event.x<=self.bt3_right and self.bt3_top<=event.y<=self.bt3_bottom:# 响应的位置
                self.play_audio()

        def motion_all(event):# 鼠标经过响应函数
            if self.bt1_left<=event.x<=self.bt1_right and self.bt1_top<=event.y<=self.bt1_bottom:# 响应的位置
                self.canvas_root.itemconfigure(self.bt1r1,outline='white')# 重设外框颜色
                self.canvas_root.itemconfigure(self.bt1r2,outline='white')# 重设内框颜色
                self.canvas_root.itemconfigure(self.bt1t,fill='white')# 重设显示文本颜色
            else:
                self.canvas_root.itemconfigure(self.bt1r1,outline='gray')# 恢复外框默认颜色
                self.canvas_root.itemconfigure(self.bt1r2,outline='gray')# 恢复内框默认颜色
                self.canvas_root.itemconfigure(self.bt1t,fill='gray')# 恢复显示文本默认颜色
            if self.bt2_left<=event.x<=self.bt2_right and self.bt2_top<=event.y<=self.bt2_bottom:# 响应的位置
                self.canvas_root.itemconfigure(self.bt2r1,outline='yellow')# 重设外框颜色
                self.canvas_root.itemconfigure(self.bt2r2,outline='yellow')# 重设内框颜色
                self.canvas_root.itemconfigure(self.bt2t,fill='yellow')# 重设显示文本颜色
            else:
                self.canvas_root.itemconfigure(self.bt2r1,outline='white')# 恢复外框默认颜色
                self.canvas_root.itemconfigure(self.bt2r2,outline='white')# 恢复内框默认颜色
                self.canvas_root.itemconfigure(self.bt2t,fill='white')# 恢复显示文本默认颜色
            if self.bt3_left<=event.x<=self.bt3_right and self.bt3_top<=event.y<=self.bt3_bottom:# 响应的位置
                self.canvas_root.itemconfigure(self.bt3r1,outline='yellow')# 重设外框颜色
                self.canvas_root.itemconfigure(self.bt3r2,outline='yellow')# 重设内框颜色
                self.canvas_root.itemconfigure(self.bt3t,fill='yellow')# 重设显示文本颜色
            else:
                self.canvas_root.itemconfigure(self.bt3r1,outline='white')# 恢复外框默认颜色
                self.canvas_root.itemconfigure(self.bt3r2,outline='white')# 恢复内框默认颜色
                self.canvas_root.itemconfigure(self.bt3t,fill='white')# 恢复显示文本默认颜色

        # 画布文字
        self.label1t0 = self.canvas_root.create_text(76+2,100+2,text='音频',font=('楷体',18,'bold'),fill='gray')
        self.label1t = self.canvas_root.create_text(76,100,text='音频',font=('楷体',18,'bold'),fill='white')

        self.label2t = self.canvas_root.create_text(76, 198,text='WAV',font=('楷体',18,'bold'),fill='white')
        self.label3t = self.canvas_root.create_text(82, 485,text='MFCC',font=('楷体',18,'bold'),fill='white')

        self.label4t0 = self.canvas_root.create_text(803+2, 330+2,text='情感：',font=('楷体',18,'bold'),fill='gray')
        self.label4 = self.canvas_root.create_text(803, 330,text='情感：',font=('楷体',18,'bold'),fill='white')

        self.label5t0 = self.canvas_root.create_text(836+2, 480+2,text='文本内容：',font=('楷体',18,'bold'),fill='gray')
        self.label5 = self.canvas_root.create_text(836, 480,text='文本内容：',font=('楷体',18,'bold'),fill='white')

        # 可滚动多行文本text
        t0x = 50; t0y = 120; t0w = 600; t0h = 30
        self.text0 = Text(self, font="song -26", bg="white")
        self.text0.place(x=t0x, y=t0y, width=t0w, height=t0h)
        self.sl0 = Scrollbar(self)
        self.text0['yscrollcommand'] = self.sl0.set
        self.sl0['command'] = self.text0.yview
        self.sl0.place(x=t0x+t0w, y=t0y, height=t0h)

        t1x = 760; t1y = 350; t1w = 400; t1h = 40
        self.text1 = Text(self,  font="song -30", bg="white")
        self.text1.place(x=t1x, y=t1y, width=t1w, height=t1h)
        self.sl1 = Scrollbar(self)
        self.text1['yscrollcommand'] = self.sl1.set
        self.sl1['command'] = self.text1.yview
        self.sl1.place(x=t1x+t1w, y=t1y, height=t1h)

        t2x = 760; t2y = 500; t2w = 400; t2h = 200
        self.text2 = Text(self,  font="song -30", bg="white")
        self.text2.place(x=t2x, y=t2y, width=t2w, height=t2h)
        self.sl2 = Scrollbar(self)
        self.text2['yscrollcommand'] = self.sl2.set
        self.sl2['command'] = self.text2.yview
        self.sl2.place(x=t2x+t2w, y=t2y, height=t2h)

    def run_out(self):
        self.text1.delete(1.0, END)
        self.text2.delete(1.0, END)
        result, score = inference(self.radio_path)
        text = recognize(self.radio_path)
        self.text1.insert(1.0, result)
        self.text2.insert(1.0, text['result'][0])
        
    def play_audio(self):
        mixer.music.set_volume(1)
        mixer.music.load(self.radio_path)
        if(self.play_status):
            self.bt3t = self.canvas_root.create_text((self.bt3_left+self.bt3_right)/2,(self.bt3_top+self.bt3_bottom)/2,text='播 放',font=('楷体',20,'bold'),fill='white')# 按钮显示文本
            mixer.music.stop()
        else:
            # self.bt3t = self.canvas_root.create_text((self.bt3_left+self.bt3_right)/2,(self.bt3_top+self.bt3_bottom)/2,text='停 止',font=('楷体',20,'bold'),fill='white')# 按钮显示文本
            mixer.music.play()

    def load_disp(self):
        self.text0.delete(1.0, END)
        self.radio_path = askopenfilename()
        self.text0.insert(1.0, self.radio_path)
        mfcc_save(self.radio_path)
        normal_figure(self.radio_path)

        self.im00 = Image.open("sources/mfcc.jpg")
        self.im01 = ImageTk.PhotoImage(self.im00)
        self.img0 = Label(self, image=self.im01)
        self.img0.place(x=50, y=500)

        self.im10 = Image.open("sources/wav.jpg")
        self.im11 = ImageTk.PhotoImage(self.im10)
        self.img1 = Label(self, image=self.im11)
        self.img1.place(x=50, y=215)

def fluent_change(i=0):
    global cover, app, root
    fluent_list = [0.01,0.01,0.01,0.01,0.03,0.03,0.05,0.05,0.09,0.09,0.15,0.15,0.09,0.09,0.05,0.04,0.03,0.02,0.01,0.01,0.01,-0.01,-0.01,-0.01]
    imx = len(fluent_list) - 1
    cover.place(x=int(cover.place_info()['x'])-bg_weight*fluent_list[i])
    app.place(x=int(app.place_info()['x'])-bg_weight*fluent_list[i])
    if i<imx:
        root.after(20,fluent_change,i+1)  # 20ms移动一次

def tk_main():
    global root,cover,app
    root = Tk()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    ww = bg_weight
    wh = bg_height
    x = (sw - ww) / 2
    y = (sh - wh) / 2

    root.title("语音情感文本识别应用")
    root.geometry("%dx%d+%d+%d" % (ww, wh, x, y))
    root.resizable(False, False)

    # 初始化音频播放器
    mixer.init()

    cover = tkCover(master=root)
    app = tkApplication(master=root)
    # 窗口循环
    root.mainloop()

if __name__ == '__main__':
    tk_main()
