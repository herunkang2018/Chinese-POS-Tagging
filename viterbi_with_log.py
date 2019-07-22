'''
维特比解码算法 采用取对数连加的方式替换概率连乘 防止浮点数下溢
'''
from utils import load_dict, suffix, percent, emit_smooth_prob, tag_set
from math import log

states = tag_set
observations = ('江苏', '是', '一', '个', '省份') # ('ns', 'v', 'm', 'q', 'n')
observations = ('汇率')

# 注意 start_p是一个很好玩的东西 需要记录所有的states 0就0
def viterbi(obs, states, start_p, trans_p, emit_p):
    """
    维特比解码算法
    :param obs:观测序列
    :param states:隐状态
    :param start_p:初始概率（隐状态）
    :param trans_p:转移概率（隐状态）
    :param emit_p: 发射概率 （隐状态表现为显状态的概率）
    """

    # 路径概率表 V[时间][隐状态] = 概率
    V = [{}]
    # 一个中间变量，代表当前状态是哪个隐状态
    path = {}

    # 初始化初始状态 (t == 0)
    for y in start_p:
        # print(y)
        try:
            V[0][y] = log(start_p[y]) + log(emit_p[y][obs[0]])
        except:
            # 简单平滑
            emit_p[y][obs[0]] = emit_smooth_prob
            # emit_p[y][obs[0]] = emit_p[y]['oov']
            V[0][y] = log(start_p[y]) + log(emit_p[y][obs[0]])
        path[y] = [y]

    # 对时刻 t > 0 跑一遍维特比算法
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        # 遍历隐状态
        for y in states:
            # 发射概率简单平滑
            try:
                emit_p[y][obs[t]]
            except:
                emit_p[y][obs[t]] = emit_smooth_prob
            # 对时刻t的任一状态 都遍历时刻t-1所有的states 对概率进行连乘后 求最大概率值后保存在V中
            (prob, state) = max(
                [(V[t - 1][y0] + log(trans_p[y0][y]) + log(emit_p[y][obs[t]]), y0)
                 for y0 in states])
            # 记录最大概率
            V[t][y] = prob
            # 记录路径
            newpath[y] = path[state] + [y]

        # 不需要保留旧路径
        path = newpath
        # print(path)
    # 最后求概率表中的记录的最大概率对应的state值
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
    # print("total path: ", path) # debug
    # 用这个state回溯path取路径
    return (prob, path[state])


def tagging(observations):
    # 加载HMM模型参数
    start_probability = load_dict('start_prob' + suffix + '.pkl')
    transition_probability = load_dict('trans_prob_new' + suffix + '.pkl')
    emission_probability = load_dict('emit_prob' + suffix + '.pkl')

    return viterbi(observations, states, start_probability,
                   transition_probability, emission_probability)


if __name__ == "__main__":
    print(tagging(observations))