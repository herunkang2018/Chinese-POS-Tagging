'''
预处理模块，对原始语料库文件进行预处理，用于后续统计
'''
import os
import codecs
from convert_to_single_byte import double_byte_to_single_byte

# 转换原始文件的GBK编码到UTF-8
def convert_encoding(input_file, output_file,
                     input_encoding='gbk', output_encoding='utf8'):
    with open(input_file, 'rb') as input_fd, open(output_file, 'wb') as output_fd:
        file_content = input_fd.read()
        unicode_file_content = file_content.decode(input_encoding)
        output_fd.write(unicode_file_content.encode(output_encoding))

# 数据清洗
def data_clean(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    data = input_data.readlines()
    for row in data:
        # 去掉19980101-01-001-001/m 这样的标签 便于后续计算开始概率
        row = row.split("  ")[1:]
        # 针对每一个词 全角转换成半角 尤其是数字
        row = [double_byte_to_single_byte(word) for word in row]
        row = (' ').join(row)
        # 去掉粗粒度的命名实体的词性标注 只使用更细类别的标注
        row = row.replace('[', '')
        row = row.replace(']nt', '')
        row = row.replace(']ns', '')
        row = row.replace(']nz', '')
        row = row.replace(']l', '')
        row = row.replace(']i', '')

        output_data.write(row)

# 句子切分 这里使用"。! ?"作为句子的结束符
# 但这里存在一个小问题 就是有的句子的结束符+引号结束，引号就被分割到下一个句子的开头了
# 这个问题在fix_split_bug中解决
def split_to_sentence(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    data = input_data.readlines()
    for row in data:
        row = row.replace("。/w ", "。/w \n") #"。/w ”/w "类型的使用手工处理
        row = row.replace("?/w ", "?/w\n")
        row = row.replace("!/w ", "!/w\n")
        row = row.strip() # 默认去掉空格和换行
        row += '\n'

        output_data.write(row)

# "。/w ”/w "类型的 单独处理
# eg:
'''
徐/nr 义昭/nr 一板一眼/i 地/u 操/v 着/u 中文/nz 说/v :/w “/w 让/v 我们/r 为/p 中/j 、/w 南/j 两/m 国/n 人民/n 的/u 友谊/n 干杯/v ,/w 让/v 我们/r 两/m 国/n 之间/f 的/u 关系/n 像/p 黄金/n 一样/u 珍贵/a ,/w 像/p 钻石/n 一样/u 坚强/a 。/w
”/w
'''
def fix_split_bug(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    data = input_data.readlines()
    for row_index in range(len(data)):
        if data[row_index][0] == "”":
            # 当前行的数据 去掉多余头部
            data[row_index] = data[row_index][4:]
            # 修改上一行的尾部 保存到文件
            data[row_index-1] = data[row_index-1][:-1]
            data[row_index-1] += "”/w "
            row = data[row_index-1]
            output_data.write(row)
        else:
            if row_index != 0:
                row = data[row_index-1]
                output_data.write(row)
    # 补上最后一行
    row = data[len(data)-1]
    output_data.write(row)


if __name__ == "__main__":
    # 转换原始文件的GBK编码到UTF-8
    print("编码转换")
    convert_encoding('data/ChineseCorpus199801.txt', 'data/raw_data.txt')

    # 数据清洗
    print("数据清洗")
    data_clean("data/raw_data.txt", "data/no_nt.txt")

    # 句子切分
    print("句子切分")
    split_to_sentence("data/no_nt.txt", "data/sentence.txt")
    fix_split_bug("data/sentence.txt", "data/sentence_last.txt")
