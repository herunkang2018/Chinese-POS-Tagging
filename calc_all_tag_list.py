'''统计出所有的tags 用于后续计算'''

from utils import load_new_file

# 统计出所有的tags
def calc_all_tag_list(data):
    # 首先统计出一共有多少类别 下面是统计结果
    '''
    tag_set = {
        'na', '', 'l', 'c', 'Rg', 'm', 'o', 'r', 'nx', 'ns', 'Vg', 'an', 'vd',
        'a', 'Yg', 'nt', 'Dg', 'k', 'h', 'i', 'd', 'Ag', 'q', 'b', 'w', 'u',
        'p', 'f', 'Mg', 'ad', 'Bg', 's', 'z', 'y', 'n', 't', 'vn', 'j', 'Tg',
        'e', 'v', 'Ng', 'nz', 'nr'
    }
    '''
    # 使用二维dict 存储emit计数
    # 这里是从隐含状态到观测状态的dict
    emit_dict = {}
    # using set
    all_tag_list = []
    for sentence in data:
        sentence = sentence.strip().split(' ')
        for word in sentence:
            word = word.split('/')
            # print(word)
            # emit_dict[word[-1]]
            all_tag_list.append(word[-1])
    print(set(all_tag_list))
    tag_number = len(set(all_tag_list))
    print(tag_number)  # 44


if __name__ == "__main__":
    sentence_file = "data/sentence_last.txt"
    data = load_new_file(sentence_file)
    calc_all_tag_list(data)