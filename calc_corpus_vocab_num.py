'''
计算总体语料库所包含所有词数 这里把重复的也计算在内 用于后续计算训练和标注速度
'''
from utils import load_new_file

def calc_corpus_vocab_num(data):
    vocab_set = []
    for sentence in data:
        sentence = sentence.strip().split(' ')
        vocab_set += sentence
    print(len(vocab_set))  # 1121119 # 语料库包含112万词（含重复）一共43317个句子 则平均句长为25.88
    print(len(set(vocab_set)))  # 62034 # 把词和标记作为一个整体计数

if __name__ == "__main__":
    sentence_file = "data/sentence_last.txt"
    data = load_new_file(sentence_file)
    calc_corpus_vocab_num(data)