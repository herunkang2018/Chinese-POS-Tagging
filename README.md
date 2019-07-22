## 项目文档

项目分为三部分，代码结构如下：

- data目录：包含人民日报语料库原始文件、预处理之后的语料文件
- tag_descr目录：包含本次实验所使用标注集的详细解释，比如r表示代词
- result目录：存放之前的实验结果，每个结果文件中都有详细的说明
- utils.py：放置全局变量，和一些经常使用的函数
- preprocess.py：预处理模块，对原始语料库文件进行预处理，用于后续统计
- viterbi.py：解码函数，viterbi_with_log.py是使用取对数连加方法的解码函数
- tagger.py：主文件，用于训练模型，并测试模型的准确率、标记速度等
- tagger_calc_speed.py：计算训练和标记性能的程序
- tagger_with_diff_corpus_size.py：用于测试模型的平均准确率随着语料库大小的变化情况
- paint_corpus_size_res.py：绘制模型的平均准确率随着语料库大小的变化图
- calc_corpus_vocab_num.py：计算全体语料库的词语总数（含重复）
- calc_all_tag_list.py：统计当前语料库使用的所有标签，用于后续HMM参数计算



执行过程：

1.项目采用python3，依赖模块通过requirements.txt安装：

```
pip install -r requirements.txt
```

2.预处理：

```
python preprocess.py
```

3.统计出所有的tags，用于后续计算：

```
python calc_all_tag_list.py
```

4.训练模型，并利用K折法对模型进行交叉验证(K=10)，计算平均准确率、集外词和兼类词准确率：

```
python tagger.py
```

5.计算训练和标记性能：

```
python tagger_calc_speed.py
```

6.其他实验：测试模型的平均准确率随着语料库大小的变化情况

```
python tagger_with_diff_corpus_size.py
```



其他说明：

1.语料库原始文件中存在一处错误，即有一处为面试/vvn，已改正为vn。

2.解码程序默认使用取对数连加的方法，准确率更高（具体实验结果见result/log_vs_no_log/README.txt)。（在tagger.py中默认import viterbi_with_log）

3.平滑结果在smooth目录中，目前的测试结果比简单平滑要差一些，后续需要再调试。

