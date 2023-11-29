import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
file_name=fr"C:\Users\tyasu\Downloads\result_all.csv"

results=pd.read_csv(file_name, skiprows=1,header=None).values



# グラフ描画
plt.figure(figsize=(16, 12))
plt.rcParams["font.size"] = 16
for i in range(4):
  frames = list(range(1, len(results[i,:])+1))
  emotions=results[i,:]
  frames = [frame for i, frame in enumerate(frames) if  type(emotions[i])==str]
  emotions = [emotion for emotion in emotions if type(emotion)==str]
  emotion_nums=[]
  for emotion in emotions:
    emotion_num=0 if emotion=="Happiness" else 1 if emotion=="Fear" else 2 if emotion=="Surprise" else 3 if emotion=="Sadness" else 4 if emotion=="Neutral" else 5 if emotion=="Disgust" else 6 if emotion=="Contempt" else 7 if emotion=="Angry" else -1
    emotion_nums.append(emotion_num)    
    
  plt.subplot(4,1,i+1)
  plt.scatter(frames, emotion_nums, marker='o', color='b', label='感情')
  plt.xlabel('frame')
  plt.ylabel('emotion')
  plt.title('emotion of frames')
  plt.yticks([0,1,2,3,4,5,6,7],["Happiness","Fear","Surprise","Sadness", "Neutral", "Disgust", "Contempt","Angry"])

  plt.grid(True)
plt.tight_layout()
plt.show()
