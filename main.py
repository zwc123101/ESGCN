import tensorflow as tf
import argparse
from include.Config import *
from include.Model import build, training
from include.weighted import *
from include.Model import *
from include.Load import *
import random
import torch
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


seed = 12306
random.seed(seed)   # 加与不加有何区别
tf.set_random_seed(seed)

'''
Followed the code style of HGCN-JE-JR:
https://github.com/StephanieWyt/HGCN-JE-JR
'''


if __name__ == '__main__':
    print('loading input files...')
    print('-----The files is as follows-----')

    e1, e2, ILL, index2rel, rel2index, KG1, KG2, refs, reft, refs_list, reft_list = read_data()
    e = len(e1 | e2)

    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * 3], dtype=np.int32)
    test = np.array(ILL[illL // 10 * 3:], dtype=np.int32)

    print('loading raw data...')
    print('-----dataset summary-----')
    print("dataset:\t", DATA_PATH)
    print("triple_1 num:\t", len(KG1), "\ttriple_2 num:\t", len(KG2))
    print("entity num:\t", e)
    print("relation num:\t", len(rel2index))
    print("train ill num:\t", len(train), "\ttest ill num:\t", len(test))
    print("-------------------------")

    # ent_embeddings = tf.Variable(tf.truncated_normal([e, DIM], stddev=1.0 / math.sqrt(DIM)))
    # rel_embeddings = tf.Variable(tf.truncated_normal([len(rel2index), DIM], stddev=1.0 / math.sqrt(DIM)))
    # ent_embeddings = tf.nn.l2_normalize(ent_embeddings, 1)
    # rel_embeddings = tf.nn.l2_normalize(rel_embeddings, 1)
    # references_s = tf.constant(refs, dtype=tf.int32)
    # references_t_list = tf.constant(reft_list, dtype=tf.int32)
    # references_t = tf.constant(reft, dtype=tf.int32)
    # references_s_list = tf.constant(refs_list, dtype=tf.int32)

    output_h, output_h_match, loss_all, sample_w, loss_w, M0, nbr_all, mask_all = \
        build(DIM, DIM_G, ACT_FUNS, GAMMA,
              K, VEC, e, ALL_NBR_NUM, SAMPLED_NBR_NUM, BETA, KG1 + KG2, train)

    se_vec, J = training(output_h, output_h_match, loss_all, sample_w, loss_w, LEARNING_RATE,
                         EPOCHS, PRE_EPOCHS, train, e,
                         K, SAMPLED_NBR_NUM, SAVE_SUFFIX, DIM, DIM_G,
                         C, train_batchnum, test_batchnum,
                         test, M0, e1, e2, nbr_all, mask_all)
    print('loss:', J)
