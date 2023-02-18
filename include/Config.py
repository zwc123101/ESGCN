import tensorflow as tf


LANG = 'fr' # language fr/zh/ja
DATA_PATH = r"data/DBP15K/{}_en/".format(LANG)  # data path
DIM = 300
DIM_G = 50
ACT_FUNS = tf.nn.relu
GAMMA = 1.0  # margin based loss
K = 125  # number of negative samples for each positive one
ALL_NBR_NUM = 30
SAMPLED_NBR_NUM = 5  # number of sampled neighbors 5
ALPHA = 0.1
BETA = 0.9  # weight of the matching vector 原 0.1
LEARNING_RATE = 0.005  # learning rate  1e-05 0.001
EPOCHS = 600
PRE_EPOCHS = 10
SAVE_SUFFIX = r"../data/DBK15K/"
C = 20  # size of the candidate set
train_batchnum = 1
test_batchnum = 5
VEC = r"data/DBP15K/{}_en/{}".format(LANG, 'vectorList.json')
ill = r"data/DBP15K/{}_en/{}".format(LANG, 'ref_ent_ids')
SEED = 3