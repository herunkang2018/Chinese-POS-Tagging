'''
全局变量定义的文件
也放置一些经常使用的函数
'''
import pickle
import codecs


def load_new_file(input_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    data = input_data.readlines()
    return data


def store_dict(dict_obj, filename):
    with open(filename, "wb") as fd:
        pickle.dump(dict_obj, fd)


def load_dict(filename):
    with open(filename, "rb") as fd:
        dict_obj = pickle.load(fd)
        return dict_obj


tag_set = {
    'na', '', 'l', 'c', 'Rg', 'm', 'o', 'r', 'nx', 'ns', 'Vg', 'an', 'vd', 'a',
    'Yg', 'nt', 'Dg', 'k', 'h', 'i', 'd', 'Ag', 'q', 'b', 'w', 'u', 'p', 'f',
    'Mg', 'ad', 'Bg', 's', 'z', 'y', 'n', 't', 'vn', 'j', 'Tg', 'e', 'v', 'Ng',
    'nz', 'nr'
}

# 定义一些全局变量
suffix = "_10"  # 保存模型pkl文件名的后缀
percent = 0.1  # 测试集占比
start_smooth_prob = 0.0001  # 初始状态概率的平滑
emit_smooth_prob = 0.00001  # 发射概率的数据平滑
trans_smooth_prob = 0.0012  # 状态转移概率的平滑
kfold_n_splits = 10
kfold_shuffle = True
