'''用来测试训练和标记速度'''
import sys
import time
from collections import Counter
import numpy as np
from sklearn.model_selection import KFold
from utils import store_dict, suffix, percent, start_smooth_prob, trans_smooth_prob, kfold_n_splits, kfold_shuffle, load_new_file, tag_set
from viterbi_with_log import tagging
# from viterbi import tagging

'''计算初始状态概率'''
def calc_start_prob(data):
    start_tag_list = []
    for sentence in data:
        sentence = sentence.strip()
        start_word = sentence.split(' ')[0]
        start_tag = start_word.split('/')[-1]
        start_tag_list.append(start_tag)
    start_number = len(start_tag_list)

    count = Counter(start_tag_list)
    start_count_dict = dict(count)
    for key in start_count_dict:
        start_count_dict[key] /= start_number

    # 对0概率的tag进行简单平滑
    for tag in tag_set:
        try:
            start_count_dict[tag]
        except:
            start_count_dict[tag] = start_smooth_prob

    store_dict(start_count_dict, "start_prob" + suffix + ".pkl")


# 改进：添加平滑方法 这里的出现次数
# 计算产生概率 这里对每一个词进行统计
def calc_emit_prob_add_one(data):
    # 首先统计出一共有多少类别 下面是统计结果
    tag_set = {
        'na', '', 'l', 'c', 'Rg', 'm', 'o', 'r', 'nx', 'ns', 'Vg', 'an', 'vd',
        'a', 'Yg', 'nt', 'Dg', 'k', 'h', 'i', 'd', 'Ag', 'q', 'b', 'w', 'u',
        'p', 'f', 'Mg', 'ad', 'Bg', 's', 'z', 'y', 'n', 't', 'vn', 'j', 'Tg',
        'e', 'v', 'Ng', 'nz', 'nr'
    }

    # 使用二维dict 存储emit计数
    # 这里是从隐含状态到观测状态的dict
    emit_dict = {}
    # init emit_dict
    for tag in tag_set:
        emit_dict[tag] = {}
    for sentence in data:
        sentence = sentence.strip().split(' ')
        for word in sentence:
            # 注：使用/切分 是有问题的 比如 吨/日/q 需要合并
            word = word.split('/')
            # print(word)
            combine_word = ('/').join(word[:-1])
            # print(combine_word)
            # print(word)
            try:
                emit_dict[word[-1]][combine_word] += 1
            except:
                emit_dict[word[-1]][combine_word] = 1
    # print(emit_dict)

    # 计算概率
    # emit_prob deepcopy from emit_dict
    for tag in emit_dict:
        values = emit_dict[tag].values()
        # print(values)
        # print("tag all_emit_words_seen: ", tag, sum(values))
        # 每一个tag的计数
        one_tag_stats_num = sum(values)
        # 添加对tag中所有不重复单词的计数
        one_tag_word_cnt = len(emit_dict[tag]) + 1
        # 加1平滑的分母
        smooth_num = one_tag_stats_num + one_tag_word_cnt
        # 对所有集外词/未登陆词 平滑后的概率
        smooth_prob_for_oov = 1 / smooth_num
        emit_dict[tag]['oov'] = smooth_prob_for_oov

        for key in emit_dict[tag]:
            # print(key)
            emit_dict[tag][key] += 1
            emit_dict[tag][key] /= smooth_num
    print(emit_dict)
    store_dict(emit_dict, "emit_prob" + suffix + ".pkl")


# 改进：把事情看成是一个2-gram 因为emit->word就2-gram很类似
# 这个时候 在语料库 也即训练集中 就有一个V的概念
# 计算产生概率 这里对每一个词进行统计
def calc_corpus_vocab_num(data):
    vocab_set = []
    for sentence in data:
        sentence = sentence.strip().split(' ')
        vocab_set += sentence
    print(len(vocab_set))  # 112262 # 语料库包含11万词（含重复）所有的
    print(len(set(vocab_set)))  # 16645 # 作为一个emit的整个词汇集使用 也即没有转移的word组成N0 有错误


# 未登录词的标记准确率
def get_iv_set(train_data):
    vocab_set = []
    for sentence in train_data:
        sentence = sentence.strip().split(' ')
        for word in sentence:
            # 注：使用/切分 是有问题的 比如 吨/日/q 需要合并
            word = word.split('/')
            combine_word = ('/').join(word[:-1])
            vocab_set.append(combine_word)
    iv_set = set(vocab_set)
    print("语料库集内词数量: ",
          len(iv_set))  # 16645 # 作为一个emit的整个词汇集使用 也即没有转移的word组成N0
    return iv_set


# 统计兼类词
def get_multi_category_word_set(train_data):
    vocab_set = []
    word_tag_dict = {}
    for sentence in train_data:
        sentence = sentence.strip().split(' ')
        for word in sentence:
            # 注：使用/切分 是有问题的 比如 吨/日/q 需要合并
            word = word.split('/')
            combine_word = ('/').join(word[:-1])
            try:
                word_tag_dict[combine_word]
            except:
                word_tag_dict[combine_word] = []
            word_tag_dict[combine_word].append(word[-1])
    for word in word_tag_dict:
        if len(set(word_tag_dict[word])) > 1:
            vocab_set.append(word)
    print("兼类词数量：", len(vocab_set))
    return vocab_set


def calc_emit_prob_gt_smoothing(data):
    # 首先统计出一共有多少类别 下面是统计结果
    tag_set = {
        'na', '', 'l', 'c', 'Rg', 'm', 'o', 'r', 'nx', 'ns', 'Vg', 'an', 'vd',
        'a', 'Yg', 'nt', 'Dg', 'k', 'h', 'i', 'd', 'Ag', 'q', 'b', 'w', 'u',
        'p', 'f', 'Mg', 'ad', 'Bg', 's', 'z', 'y', 'n', 't', 'vn', 'j', 'Tg',
        'e', 'v', 'Ng', 'nz', 'nr'
    }

    # 使用二维dict 存储emit计数
    # 这里是从隐含状态到观测状态的dict
    emit_dict = {}
    # init emit_dict
    for tag in tag_set:
        emit_dict[tag] = {}
    for sentence in data:
        sentence = sentence.strip().split(' ')
        for word in sentence:
            # 注：使用/切分 是有问题的 比如 吨/日/q 需要合并
            word = word.split('/')
            # print(word)
            combine_word = ('/').join(word[:-1])
            # print(combine_word)
            # print(word)
            try:
                emit_dict[word[-1]][combine_word] += 1
            except:
                emit_dict[word[-1]][combine_word] = 1
    # print(emit_dict)

    # 计算概率
    # emit_prob deepcopy from emit_dict
    vocab_set_size = 16645  # 改变percent参数 这个值必须相应地改变
    for tag in emit_dict:
        values = emit_dict[tag].values()
        one_tag_stats_num = sum(values)

        # 计算最大转移的最大计数
        max_cnt = max(values)
        gt_number_dict = {i: [] for i in range(max_cnt + 1)}
        # 计算转移一次的set
        for key in emit_dict[tag]:
            gt_number_dict[emit_dict[tag][key]].append(key)

        # print(gt_number_dict)
        gt_nr_list = [0 for i in range(max_cnt + 1)]

        nr_cnt = 0
        for index in gt_number_dict:
            gt_nr_list[index] = len(gt_number_dict[index])  # len of list
            nr_cnt += gt_nr_list[index]

        # paint to explore
        print("example tag: ", tag)
        print(gt_nr_list)
        # exit()
        '''
        # using calc_corpus_vocab_num to calc the n0
        gt_nr_list[0] = vocab_set_size - nr_cnt

        # using gt smoothing
        dr = [0 for i in range(max_cnt+1)]
        for i in range(max_cnt):
            dr[i] = i * (gt_nr_list[i+1]/gt_nr_list[i])

        # print(values)
        # print("tag all_emit_words_seen: ", tag, sum(values))
        # 每一个tag的计数
        # 添加对tag中所有不重复单词的计数
        # one_tag_word_cnt = len(emit_dict[tag])
        for key in emit_dict[tag]:
            # print(key)
            emit_dict[tag][key] /= one_tag_stats_num
    print(emit_dict)
    store_dict(emit_dict, "emit_prob" + suffix + ".pkl")
    '''


# 计算产生概率 这里对每一个词进行统计
# Note: 课程网站上面下载的人民日报语料库有一处错误
# 面试/vvn  全县/n  第一/m  的/u  成绩/n 把vvn编程vn即可 不然会出错
def calc_emit_prob(data):
    # 使用二维dict 存储emit计数
    # 这里是从隐含状态到观测状态的dict
    emit_dict = {}
    # init emit_dict
    for tag in tag_set:
        emit_dict[tag] = {}
    for sentence in data:
        sentence = sentence.strip().split(' ')
        for word in sentence:
            # 注：使用/切分 是有问题的 比如 吨/日/q 需要合并
            word = word.split('/')
            # print(word)
            combine_word = ('/').join(word[:-1])
            # print(combine_word)
            # print(word)
            try:
                emit_dict[word[-1]][combine_word] += 1
            except:
                emit_dict[word[-1]][combine_word] = 1
    # print(emit_dict)

    # 计算概率
    # emit_prob deepcopy from emit_dict
    for tag in emit_dict:
        values = emit_dict[tag].values()
        # print(values)
        # print("tag all_emit_words_seen: ", tag, sum(values))
        # 每一个tag的计数
        one_tag_stats_num = sum(values)
        # 添加对tag中所有不重复单词的计数
        # one_tag_word_cnt = len(emit_dict[tag])
        for key in emit_dict[tag]:
            # print(key)
            emit_dict[tag][key] /= one_tag_stats_num
    # print(emit_dict)
    store_dict(emit_dict, "emit_prob" + suffix + ".pkl")


# 计算转移概率
def calc_trans_prob(data):
    trans_dict = {}
    # init trans_dict
    for tag in tag_set:
        trans_dict[tag] = {}

    for sentence in data:
        sentence = sentence.strip().split(' ')
        for index in range(len(sentence)):
            word = sentence[index]
            # 注：使用/切分 是有问题的 比如 吨/日/q 需要合并
            word = word.split('/')
            word_tag = word[-1]
            # print(word_tag)
            if index != len(sentence) - 1:
                next_tag = sentence[index + 1].split('/')[-1]
                try:
                    trans_dict[word_tag][next_tag] += 1
                except:
                    trans_dict[word_tag][next_tag] = 1

    print(trans_dict)
    print()
    print()

    # 计算概率
    # emit_prob deepcopy from emit_dict
    for tag in trans_dict:
        values = trans_dict[tag].values()
        # print(values)
        print("tag all_emit_words_seen: ", tag, sum(values))
        one_tag_trans_num = sum(values)
        for next_tag in trans_dict[tag]:
            # print(next_tag)
            trans_dict[tag][next_tag] /= one_tag_trans_num
    print(trans_dict)
    store_dict(trans_dict, "trans_prob.pkl")


def calc_trans_prob_new(data):
    trans_dict = {}
    # init trans_dict
    for tag in tag_set:
        trans_dict[tag] = {}
        for next_tag in tag_set:
            trans_dict[tag][next_tag] = 0

    for sentence in data:
        # strip 去掉收尾空格和换行
        sentence = sentence.strip().split(' ')
        # print(sentence)
        for index in range(len(sentence)):
            word = sentence[index]
            # 注：使用/切分 是有问题的 比如 吨/日/q 需要合并
            word = word.split('/')
            word_tag = word[-1]
            # print(word_tag)
            if index != len(sentence) - 1:
                next_tag = sentence[index + 1].split('/')[-1]
                try:
                    trans_dict[word_tag][next_tag] += 1
                except:
                    trans_dict[word_tag][next_tag] = 1

    # print(trans_dict)
    # print()
    # print()

    # 计算概率
    # emit_prob deepcopy from emit_dict
    for tag in trans_dict:
        values = trans_dict[tag].values()
        # print(values)
        # print("tag all_emit_words_seen: ", tag, sum(values))
        one_tag_trans_num = sum(values)
        for next_tag in trans_dict[tag]:
            # print(next_tag)
            trans_dict[tag][next_tag] /= one_tag_trans_num
    print(trans_dict)
    store_dict(trans_dict, "trans_prob_new" + suffix + ".pkl")


# just using fixed value to smooth the 0 value
def calc_trans_prob_simple_smooth(data):

    trans_dict = {}
    # init trans_dict
    for tag in tag_set:
        trans_dict[tag] = {}
        for next_tag in tag_set:
            trans_dict[tag][next_tag] = 0

    for sentence in data:
        # strip 去掉收尾空格和换行
        sentence = sentence.strip().split(' ')
        # print(sentence)
        for index in range(len(sentence)):
            word = sentence[index]
            # 注：使用/切分 是有问题的 比如 吨/日/q 需要合并
            word = word.split('/')
            word_tag = word[-1]
            # print(word_tag)
            if index != len(sentence) - 1:
                next_tag = sentence[index + 1].split('/')[-1]
                try:
                    trans_dict[word_tag][next_tag] += 1
                except:
                    trans_dict[word_tag][next_tag] = 1

    # print(trans_dict)
    # print()
    # print()

    # 计算概率
    # emit_prob deepcopy from emit_dict
    for tag in trans_dict:
        # print("tag: ", tag) # 遇到问题 拟声词o在划分数据集时 经常划分不到训练集中 导致这里报错

        values = trans_dict[tag].values()
        # print(values)
        # print("tag all_emit_words_seen: ", tag, sum(values))
        one_tag_trans_num = sum(values)
        # for rag o
        if one_tag_trans_num == 0:
            for next_tag in trans_dict[tag]:
                trans_dict[tag][next_tag] = trans_smooth_prob
        else:
            for next_tag in trans_dict[tag]:
                # print(next_tag)
                trans_dict[tag][next_tag] /= one_tag_trans_num
                if trans_dict[tag][next_tag] == 0:
                    trans_dict[tag][next_tag] = trans_smooth_prob

    # print(trans_dict)
    store_dict(trans_dict, "trans_prob_new" + suffix + ".pkl")


# test for perfermance
def calc_trans_prob_gt_smoothing(data):

    tag_set_size = len(tag_set)

    trans_dict = {}
    # init trans_dict
    for tag in tag_set:
        trans_dict[tag] = {}
        for next_tag in tag_set:
            trans_dict[tag][next_tag] = 0

    for sentence in data:
        # strip 去掉收尾空格和换行
        sentence = sentence.strip().split(' ')
        # print(sentence)
        for index in range(len(sentence)):
            word = sentence[index]
            # 注：使用/切分 是有问题的 比如 吨/日/q 需要合并
            word = word.split('/')
            word_tag = word[-1]
            # print(word_tag)
            if index != len(sentence) - 1:
                next_tag = sentence[index + 1].split('/')[-1]
                try:
                    trans_dict[word_tag][next_tag] += 1
                except:
                    trans_dict[word_tag][next_tag] = 1

    # debug
    all_nr_cnt = 0
    '''
    for tag in trans_dict:
        values = trans_dict[tag].values()
        one_tag_stats_num = sum(values)

        # 计算最大转移的最大计数
        max_cnt = max(values)
        gt_number_dict = {i:[] for i in range(max_cnt+1)}
        # 计算转移一次的list
        for key in trans_dict[tag]:
            gt_number_dict[trans_dict[tag][key]].append(key)

        # print(gt_number_dict)
        gt_nr_list = [0 for i in range(max_cnt+1)]

        nr_cnt = 0
        for index in gt_number_dict:
            gt_nr_list[index] = len(gt_number_dict[index]) # len of list
            nr_cnt += gt_nr_list[index] * index

        # nr_cnt = tag_set_size - gt_nr_list[0]

        # paint to explore
        print("example tag: ", tag)
        print(gt_nr_list)
        print(nr_cnt)
        all_nr_cnt += nr_cnt

        # 完全平滑 左0 都不平滑
        # r or nr?
        # r是出现次数 nr是对出现次数为r进行计数 所以频率应该以r/N
        # 平滑之后r变为dr 频率就变为dr/N
        # using gt smoothing
        dr = [0 for i in range(max_cnt+1)]

        dr[max_cnt] = max_cnt / nr_cnt
        for i in range(1, max_cnt):
            if gt_nr_list[i] != 0 and gt_nr_list[i+1] != 0:
                temp_dr = (i+1) * gt_nr_list[i+1] / gt_nr_list[i]
                dr[i] = temp_dr / nr_cnt
            else:
                gt_nr_list[i] = i / nr_cnt
        remain_prob = 1 - sum(dr[1:])
        n0 = gt_nr_list[0]
        dr[0]= remain_prob /n0
        if dr[0] < 0:
            print(dr)
            print("panic")
            exit()


    print(all_nr_cnt)
    print(dr)

        # 归一化
    exit()
    '''
    ###
    # 计算概率
    # emit_prob deepcopy from emit_dict
    # debug:
    sum_prob_all = []
    using_gt_tag_num = 0
    for tag in trans_dict:
        values = trans_dict[tag].values()
        # print(values)
        # print("tag all_emit_words_seen: ", tag, sum(values))
        one_tag_trans_num = sum(values)
        if one_tag_trans_num < 100000:
            using_gt_tag_num += 1
            for next_tag in trans_dict[tag]:
                # print(next_tag)
                trans_dict[tag][next_tag] /= one_tag_trans_num
                # add simple smooth for 0 value
                if trans_dict[tag][next_tag] == 0:
                    trans_dict[tag][next_tag] = trans_smooth_prob

            # print(trans_dict)
            # store_dict(trans_dict, "trans_prob_new" + suffix + ".pkl")
        else:
            # gt smooth
            # 计算最大转移的最大计数
            max_cnt = max(values)
            gt_number_dict = {i: [] for i in range(max_cnt + 1)}
            # 计算转移一次的list
            for key in trans_dict[tag]:
                gt_number_dict[trans_dict[tag][key]].append(key)

            # print("gt_number_dict: ", gt_number_dict)
            gt_nr_list = [0 for i in range(max_cnt + 1)]

            nr_cnt = 0
            for index in gt_number_dict:
                gt_nr_list[index] = len(gt_number_dict[index])  # len of list
                nr_cnt += gt_nr_list[index] * index

            # nr_cnt = tag_set_size - gt_nr_list[0]

            # paint to explore
            print("example tag: ", tag)
            print(gt_nr_list)
            print(nr_cnt)
            all_nr_cnt += nr_cnt

            # 完全平滑 左0 都不平滑
            # r or nr?
            # r是出现次数 nr是对出现次数为r进行计数 所以频率应该以r/N
            # 平滑之后r变为dr 频率就变为dr/N
            # using gt smoothing
            dr = [0 for i in range(max_cnt + 1)]

            dr[max_cnt] = max_cnt / nr_cnt
            for i in range(1, max_cnt):
                if gt_nr_list[i] != 0 and gt_nr_list[i + 1] != 0:
                    temp_dr = (i + 1) * gt_nr_list[i + 1] / gt_nr_list[i]
                    dr[i] = temp_dr / nr_cnt
                else:
                    gt_nr_list[i] = i / nr_cnt
            remain_prob = 1 - sum(dr[1:])
            n0 = gt_nr_list[0]
            dr[0] = remain_prob / n0
            if dr[0] < 0:
                print(dr)
                print("panic")
                exit()
            print(all_nr_cnt)
            print(dr)

            # to trans_dict
            # index as key
            for index in gt_number_dict:
                for next_tag in gt_number_dict[index]:
                    trans_dict[tag][next_tag] = dr[index]

            # 归一化
            sum_prob = 0
            for i in range(max_cnt + 1):
                sum_prob += gt_nr_list[i] * dr[i]
            print("sum_prob: ", sum_prob)

            for index in gt_number_dict:
                for next_tag in gt_number_dict[index]:
                    trans_dict[tag][next_tag] = dr[index] / sum_prob

            sum_prob_all.append(sum_prob)

        # end if else
    # end for
    print("using_gt_tag_num: ", using_gt_tag_num)
    # exit()
    print(trans_dict)
    print("sum_prob_all: ", sum_prob_all)
    # exit()
    store_dict(trans_dict, "trans_prob_new" + suffix + ".pkl")


# 初始计算词频 看是否存在连续乘法溢出的问题
# 如果不行 就直接使用log

# suffix 使用后缀 来进行train和test时 模型的区别保存
if __name__ == "__main__":
    # 4469 sentences
    sentence_file = "data/sentence_last.txt"
    data = load_new_file(sentence_file)

    # train
    #   calc three prob matrix
    # calc_start_prob(data)
    # calc_emit_prob(data)
    # calc_trans_prob_new(data)

    # test
    # total sentence number: 4469
    # test set 10%: last 447 sentences
    # train_data = data[:-447]
    # test_data = data[-447:]
    # train_data = data[447:]
    # test_data = data[:447]
    # define result
    precision_list = []
    oov_precision_list = []
    multi_cat_precision_list = []
    oov_rate_list = []
    multi_cat_rate_list = []
    kf = KFold(n_splits=kfold_n_splits, shuffle=kfold_shuffle, random_state=3)
    kf_cnt = 0
    data = np.array(data)
    train_consume_time_list = []
    test_consume_time_list = []
    for train_index, test_index in kf.split(data):
        train_start_time = time.time()
        kf_cnt += 1
        print("using CV-KFold {0}: ".format(kf_cnt))
        # print("train index: ", list(train_index))
        # print("test index: ", list(test_index))

        # data = list(data)
        # Note: 使用np.array会出现MemoryError
        # train_data = data[train_index]
        # test_data = data[test_index]

        train_data = [data[index] for index in train_index]
        test_data = [data[index] for index in test_index]

        # 用于计算未登录词的标记准确率
        iv_set = get_iv_set(train_data)
        # 用于计算兼类词标记准确率
        multi_cat_set = get_multi_category_word_set(train_data)

        # train
        calc_start_prob(train_data)
        calc_emit_prob(train_data)
        # calc_emit_prob_add_one(train_data)
        # calc_emit_prob_gt_smoothing(train_data)
        # calc_trans_prob_new(train_data)
        # calc_trans_prob_gt_smoothing(train_data)
        calc_trans_prob_simple_smooth(train_data)

        # calc_corpus_vocab_num(data)

        # 对train计时
        train_end_time = time.time()
        train_consume_time = train_end_time - train_start_time
        train_consume_time_list.append(train_consume_time)

        # test
        test_start_time = time.time()
        # total count
        total_cnt = 0
        total_size = 0
        for test_sentence in test_data:
            test_sentence = test_sentence.strip()
            # print(test_sentence)
            test_list = test_sentence.split(" ")
            # print(test_list)
            observations = []
            test_tag_list = []
            for word in test_list:
                # 注：使用/切分 是有问题的 比如 吨/日/q 需要合并
                word = word.split('/')
                test_tag_list.append(word[-1])
                combine_word = ('/').join(word[:-1])
                observations.append(combine_word)
            # print(observations)
            observations = tuple(observations)
            res = tagging(observations)

            # test time end

            total_size += len(test_list)
        test_time_end = time.time()
        test_consume_time = test_time_end - test_start_time
        test_consume_time_list.append(test_consume_time)
        print("训练耗时：{0}s 测试耗时：{1}s".format(train_consume_time, test_consume_time))

    print(
        "Using CV-KFold K等分数(n_splits)={0} \n训练速度：{1}个/s\n标记速度(仅标记)：{2}个/s"
        .format(kfold_n_splits,
                1121126 * (kfold_n_splits - 1) / sum(train_consume_time_list),
                1121126 / sum(test_consume_time_list)))
