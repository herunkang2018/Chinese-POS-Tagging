'''
打印模型的平均准确率随着语料库大小的变化关系图
'''
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

mean_precision = [0.8793565015380604, 0.8939492425618768, 0.904964546371579, 0.9074774731346051, 0.9094786030013993, 0.9038174256514628, 0.9043528048126959]
corpus_size = [5, 10, 20, 40, 60, 80, 100] # 语料库大小分别为5%、10%...
x = corpus_size
y = [i*100 for i in mean_precision]
plt.plot(x,y,color="blue",linewidth=2)
plt.xlabel("语料库大小(%)")
plt.ylabel("平均准确率(%)")
plt.title("平均准确率随语料库大小的变化曲线")
plt.hlines(90.38174256514628, 5, 100, colors = "c", linestyles = "dashed")
plt.show()
