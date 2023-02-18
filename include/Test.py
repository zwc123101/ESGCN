import numpy as np
import scipy.spatial
import pickle as pkl
from utils_nmn import *
from include.Model import *
import gc
import tensorflow as tf


def get_hits(vec, test_pair, c, top_k=(1, 5, 10, 50, 100)):
    L = np.array([e1 for e1, e2 in test_pair])  # 测试集中源实体
    R = np.array([e2 for e1, e2 in test_pair])  # 测试集中目的实体
    Lvec = vec[L]  # shape(3500, 50)
    Rvec = vec[R]  # shape(3500, 50)

    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')  # 计算两个输入集合中每对之间的距离 曼哈顿距离
    top_lr = [0] * len(top_k)
    candidate = []  # 候选对
    t = time.time()
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
        candidate.append(R[rank[0:c]])

    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
        candidate.append(L[rank[0:c]])

    # mean = valid(ent_embeddings, references_s, references_t_list)
    print('For each left (KG structure embedding):')
    acc_list = []
    for i in range(len(top_lr)):
        acc = top_lr[i] / len(test_pair)
        acc_list.append(round(acc, 4))
        # print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    # print('acc of top {} = {}, mean = {:.3f}, time = {:.3f}'.format(top_k, acc_list, mean, time.time() - t))
    print('acc of top {} = {}, time = {:.3f}'.format(top_k, acc_list, time.time() - t))

    return np.array(candidate).reshape((2, -1, c))


def get_hits_new(vec, candidate, test_pair, c,
                 top_k=(1, 5, 10, 50, 100)):
    t = len(test_pair)
    L = np.array([e1 for e1, e2 in test_pair])
    R = np.array([e2 for e1, e2 in test_pair])

    vec = np.reshape(vec, (2, t, 2, c, -1))
    Lvec = vec[0, :, 0]
    Rvec = vec[1, :, 0]
    sim = np.sum(np.abs(Lvec - Rvec), -1)
    candidate_L = np.reshape(candidate, (2, t, -1))[0]

    top_lr = [0] * len(top_k)
    ti = time.time()
    for i in range(t):
        x = -1
        for j in range(len(candidate_L[i])):
            if R[i] == candidate_L[i][j]:
                x = j
        if x >= 0:
            rank = sim[i].argsort()
            rank_index = np.where(rank == x)[0][0]
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    top_lr[j] += 1
    # mean = valid(ent_embeddings, references_s, references_t_list)
    print('For each left (NMN):')
    acc_list = []
    for i in range(len(top_lr)):
        # print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
        acc = top_lr[i] / len(test_pair)
        acc_list.append(round(acc, 4))
        # print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    # print('acc of top {} = {}, mean = {:.3f}, time = {:.3f}'.format(top_k, acc_list, mean, time.time() - t))
    print('acc of top {} = {}, time = {:.3f}'.format(top_k, acc_list, time.time() - t))


# def valid(ent_embeddings, references_s, references_t):
#     mean = 0
#     s_len = int(references_s.shape[0])
#     s_embeddings = tf.nn.embedding_lookup(ent_embeddings, references_s)
#     t_embeddings = tf.nn.embedding_lookup(ent_embeddings, references_t)
#
#     similarity_mat = tf.matmul(s_embeddings, t_embeddings, transpose_b=True)
#     sim = similarity_mat.eval()
#     for i in range(s_len):
#         ref = i
#         rank = (-sim[i, :]).argsort()
#         assert ref in rank
#         rank_index = np.where(rank == ref)[0][0]
#         mean += (rank_index + 1)
#     mean /= s_len
#
#     return mean