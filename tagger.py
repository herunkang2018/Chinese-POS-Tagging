'''
主文件，用于训练模型，并测试模型的准确率、标记速度等
'''
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

'''
使用加一平滑方法
测试后效果不好 当前未用
'''
def calc_emit_prob_add_one(data):

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
            combine_word = ('/').join(word[:-1])
            try:
                emit_dict[word[-1]][combine_word] += 1
            except:
                emit_dict[word[-1]][combine_word] = 1

    # 计算概率
    for tag in emit_dict:
        values = emit_dict[tag].values()
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
            emit_dict[tag][key] += 1
            emit_dict[tag][key] /= smooth_num
    print(emit_dict)
    store_dict(emit_dict, "emit_prob" + suffix + ".pkl")

''' 统计集内词集合 用于后续计算未登录词的标记准确率 '''
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


''' 统计兼类词集合 用于后续计算兼类词准确率 '''
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

''' 使用good-turing平滑发射概率矩阵 当前未使用'''
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


''' 计算发射概率矩阵 '''
def calc_emit_prob(data):
    # 使用二维dict 存储emit计数
    # 这里是从隐含状态到观测状态的dict
    emit_dict = {}
    # init emit dict
    for tag in tag_set:
        emit_dict[tag] = {}
    for sentence in data:
        sentence = sentence.strip().split(' ')
        for word in sentence:
            # 注：使用/切分 是有问题的 比如 吨/日/q 需要合并
            word = word.split('/')
            combine_word = ('/').join(word[:-1])
            try:
                emit_dict[word[-1]][combine_word] += 1
            except:
                emit_dict[word[-1]][combine_word] = 1

    # 计算概率
    for tag in emit_dict:
        values = emit_dict[tag].values()
        # 每一个tag的计数
        one_tag_stats_num = sum(values)
        # 添加对tag中所有不重复单词的计数
        for key in emit_dict[tag]:
            emit_dict[tag][key] /= one_tag_stats_num
    store_dict(emit_dict, "emit_prob" + suffix + ".pkl")


''' 计算转移概率矩阵 未加任何平滑'''
def calc_trans_prob(data):
    trans_dict = {}
    # init trans dict
    for tag in tag_set:
        trans_dict[tag] = {}

    for sentence in data:
        sentence = sentence.strip().split(' ')
        for index in range(len(sentence)):
            word = sentence[index]
            # 注：使用/切分 是有问题的 比如 吨/日/q 需要合并
            word = word.split('/')
            word_tag = word[-1]
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
    for tag in trans_dict:
        values = trans_dict[tag].values()
        print("tag all_emit_words_seen: ", tag, sum(values))
        one_tag_trans_num = sum(values)
        for next_tag in trans_dict[tag]:
            trans_dict[tag][next_tag] /= one_tag_trans_num
    print(trans_dict)
    store_dict(trans_dict, "trans_prob.pkl")

''' 使用简单平滑方法(也即一个很小的固定值) 对状态转移矩阵进行平滑 '''
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
        for index in range(len(sentence)):
            word = sentence[index]
            # 注：使用/切分 是有问题的 比如 吨/日/q 需要合并
            word = word.split('/')
            word_tag = word[-1]
            if index != len(sentence) - 1:
                next_tag = sentence[index + 1].split('/')[-1]
                try:
                    trans_dict[word_tag][next_tag] += 1
                except:
                    trans_dict[word_tag][next_tag] = 1

    # 计算概率
    for tag in trans_dict:
        values = trans_dict[tag].values()
        one_tag_trans_num = sum(values)
        if one_tag_trans_num == 0:
            for next_tag in trans_dict[tag]:
                trans_dict[tag][next_tag] = trans_smooth_prob
        else:
            for next_tag in trans_dict[tag]:
                trans_dict[tag][next_tag] /= one_tag_trans_num
                if trans_dict[tag][next_tag] == 0:
                    trans_dict[tag][next_tag] = trans_smooth_prob

    store_dict(trans_dict, "trans_prob_new" + suffix + ".pkl")


''' 使用gt平滑状态转移矩阵 测试性能是否提升 未使用 '''
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
        for index in range(len(sentence)):
            word = sentence[index]
            # 注：使用/切分 是有问题的 比如 吨/日/q 需要合并
            word = word.split('/')
            word_tag = word[-1]
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


# suffix 使用后缀 来进行train和test时 模型的区别保存
if __name__ == "__main__":

    sentence_file = "data/sentence_last.txt"
    data = load_new_file(sentence_file)

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

        # Note: 使用np.array会出现MemoryError 所以替换为下面list的方式
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
        calc_trans_prob_simple_smooth(train_data)
        # calc_emit_prob_add_one(train_data)
        # calc_emit_prob_gt_smoothing(train_data)
        # calc_trans_prob_new(train_data)
        # calc_trans_prob_gt_smoothing(train_data)

        # 对train计时
        train_end_time = time.time()
        train_consume_time_list.append(train_end_time - train_start_time)

        # test
        test_start_time = time.time()
        # total count
        total_cnt = 0
        total_size = 0
        total_oov_cnt = 0
        total_oov = 0
        total_multi_cat = 0
        total_multi_cat_cnt = 0
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
            pred_tag_list = res[1]
            # print(res[1])

            # 打印标记结果
            pred_res = ""
            word_tag_pair = zip(observations, res[1])
            for word, tag in word_tag_pair:
                # print("word, tag: ", word, tag)
                pred_res += word + "/" + tag + " "

            # print(pred_res)
            # print("final res: ", pred_res)
            # print("original: ", test_list)

            # 对比结果 进行计数统计
            sentence_size = len(test_list)
            # count
            cnt = 0
            oov = 0
            oov_cnt = 0
            multi_cat = 0
            multi_cat_cnt = 0
            for index in range(sentence_size):
                if observations[index] not in iv_set:
                    oov += 1
                # print(observations[index])
                if observations[index] in multi_cat_set:
                    multi_cat += 1
                if test_tag_list[index] == pred_tag_list[index]:
                    cnt += 1
                    if observations[index] not in iv_set:
                        oov_cnt += 1
                    if observations[index] in multi_cat_set:
                        multi_cat_cnt += 1
            # print("count/size: ", cnt, sentence_size)
            total_cnt += cnt
            total_size += sentence_size
            total_oov += oov
            total_oov_cnt += oov_cnt
            total_multi_cat += multi_cat
            total_multi_cat_cnt += multi_cat_cnt
        oov_precision = total_oov_cnt / total_oov
        oov_precision_list.append(oov_precision)
        multi_cat_precision = total_multi_cat_cnt / total_multi_cat
        multi_cat_precision_list.append(multi_cat_precision)
        precision = total_cnt / total_size
        precision_list.append(precision)
        # 计算集外词占比
        oov_rate = total_oov / total_size
        oov_rate_list.append(oov_rate)
        # 计算兼类词占比
        multi_cat_rate = total_multi_cat / total_size
        multi_cat_rate_list.append(multi_cat_rate)
        print("标记正确计数：{0} 总标记计数：{1} 准确率：{2} ".format(total_cnt, total_size,
                                                     precision))
        print("集外词标记正确计数：{0} 集外词总标记计数：{1} 集外词标记准确率：{2} 集外词占比：{3}".format(
            total_oov_cnt, total_oov, oov_precision, oov_rate))
        print("兼类词标记正确计数：{0} 兼类词总标记计数：{1} 兼类词标记准确率：{2} 兼类词占比：{3}".format(
            total_multi_cat_cnt, total_multi_cat, multi_cat_precision,
            multi_cat_rate))

        test_end_time = time.time()
        test_consume_time = test_end_time - test_start_time
        test_consume_time_list.append(test_consume_time)
        print("消耗时间：{0}s 标记速度(标记+评测)：{1}个/s".format(test_consume_time, total_size/test_consume_time))
    # end for kf.split(data)
    print(
        "Using CV-KFold K等分数(n_splits)={0} \n平均准确率(Mean precision): {1}\n集外词平均准确率(OOV Mean precision): {2}\n兼类词平均准确率(Multi Category Mean precision): {3}\n集外词占比(OOV rate): {4}\n兼类词占比(Multi Category rate): {5}\n训练速度：{6}个/s\n标记速度(标记+评测)：{7}个/s"
        .format(kfold_n_splits,
                sum(precision_list) / kfold_n_splits,
                sum(oov_precision_list) / kfold_n_splits,
                sum(multi_cat_precision_list) / kfold_n_splits,
                sum(oov_rate_list) / kfold_n_splits,
                sum(multi_cat_rate_list) / kfold_n_splits,
                1121126 * (kfold_n_splits - 1) / sum(train_consume_time_list),
                1121126 / sum(test_consume_time_list)))
