这里的结果是探究是否使用概率取对数连加，来代替概率连乘可能导致的浮点数下溢，
而对Viterbi解码造成的影响
结果表明，不使用log的准确率是89.38%，而使用log的准确率是90.18%，提高明显。
说明不使用log是会导致浮点数下溢的。推荐使用log的方法进行解码。
实验代码：
把tagger.py代码中的from viterbi import tagging改为from viterbi_with_log import tagging，重新计算即可