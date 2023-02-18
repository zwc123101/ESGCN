import numpy as np
from include.Config import *


# load a file and return a list of tuple containing $num integers in each line
# 修改处
def read_data(data_path = DATA_PATH):
	def read_idtuple_file(file_path):
		print('loading a idtuple file...   ' + file_path)
		ret = []
		with open(file_path, 'r', encoding='utf-8') as f:
			for line in f:
				th = line.strip('\n').split('\t')
				x = []
				for i in range(len(th)):
					x.append(int(th[i]))
				ret.append(tuple(x))
		return ret

	def read_id2object(file_paths):
		id2object = {}
		for file_path in file_paths:
			with open(file_path, 'r', encoding='utf-8') as f:
				print('loading a id2object file...  ' + file_path)
				for line in f:
					th = line.strip('\n').split('\t')
					id2object[int(th[0])] = th[1]
		return id2object

	def loadfile(file_path, fm=1):
		# print('loading a file...' + fn)
		ret = []
		with open(file_path, encoding='utf-8') as f:
			for line in f:
				th = line[:-1].split('\t')
				x = []
				for i in range(fm):
					x.append(int(th[i]))
				ret.append(tuple(x))
		return ret

	print("load data from... :", data_path)
	# ent_index(ent_id)2entity / relation_index(rel_id)2relation
	index2entity = read_id2object([data_path + "ent_ids_1", data_path + "ent_ids_2"])
	e1 = loadfile(data_path + 'ent_ids_1')
	e2 = loadfile(data_path + 'ent_ids_2')
	index2rel = read_id2object([data_path + "rel_ids_1", data_path + "rel_ids_2"])
	entity2index = {e: idx for idx, e in index2entity.items()}
	rel2index = {r: idx for idx, r in index2rel.items()}
	ILL = loadfile(ill, 2)
	# triples
	rel_triples_1 = read_idtuple_file(data_path + 'triples_1')
	rel_triples_2 = read_idtuple_file(data_path + 'triples_2')

	ref_source,  ref_target = [], []
	for i in ILL:
		ref_source.append(i[0])
		ref_target.append(i[1])
	import copy
	ref1_list = copy.deepcopy(ref_source)
	ref2_list = copy.deepcopy(ref_target)

	return set(e1), set(e2), ILL, index2rel, rel2index, rel_triples_1, rel_triples_2, ref_source, ref_target, ref1_list, ref2_list

# The most frequent attributes are selected to save space 选择使用频率最高的属性以节省空间
def loadattr(fns, e, ent2id):
	cnt = {}
	for fn in fns:
		with open(fn, 'r', encoding='utf-8') as f:
			for line in f:
				th = line[:-1].split('\t')
				if th[0] not in ent2id:
					continue
				for i in range(1, len(th)):
					if th[i] not in cnt:
						cnt[th[i]] = 1
					else:
						cnt[th[i]] += 1
	fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
	attr2id = {}
	for i in range(1000):
		attr2id[fre[i][0]] = i
	attr = np.zeros((e, 1000), dtype=np.float32)
	for fn in fns:
		with open(fn, 'r', encoding='utf-8') as f:
			for line in f:
				th = line[:-1].split('\t')
				if th[0] in ent2id:
					for i in range(1, len(th)):
						if th[i] in attr2id:
							attr[ent2id[th[0]]][attr2id[th[i]]] = 1.0
	return attr
