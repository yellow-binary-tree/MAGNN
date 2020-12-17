import os
import json

import dgl
import networkx as nx
import numpy as np
import scipy
import pickle
import torch

from utils.logger import logger


def load_IMDB_data(prefix='data/preprocessed/IMDB_processed'):
    G00 = nx.read_adjlist(prefix + '/0/0-1-0.adjlist', create_using=nx.MultiDiGraph)
    G01 = nx.read_adjlist(prefix + '/0/0-2-0.adjlist', create_using=nx.MultiDiGraph)
    G10 = nx.read_adjlist(prefix + '/1/1-0-1.adjlist', create_using=nx.MultiDiGraph)
    G11 = nx.read_adjlist(prefix + '/1/1-0-2-0-1.adjlist', create_using=nx.MultiDiGraph)
    G20 = nx.read_adjlist(prefix + '/2/2-0-2.adjlist', create_using=nx.MultiDiGraph)
    G21 = nx.read_adjlist(prefix + '/2/2-0-1-0-2.adjlist', create_using=nx.MultiDiGraph)        # node types - 0: M, 1: D, 2: A
    idx00 = np.load(prefix + '/0/0-1-0_idx.npy')        # each row is a set of node indexes on a metapath
    idx01 = np.load(prefix + '/0/0-2-0_idx.npy')
    idx10 = np.load(prefix + '/1/1-0-1_idx.npy')
    idx11 = np.load(prefix + '/1/1-0-2-0-1_idx.npy')
    idx20 = np.load(prefix + '/2/2-0-2_idx.npy')
    idx21 = np.load(prefix + '/2/2-0-1-0-2_idx.npy')
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz')      # 3066 dim sparse feature matrix. can have weight.
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz')
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz')
    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')                  # 11616*11616 adj matrix. can have weight.
    type_mask = np.load(prefix + '/node_types.npy')                     # (0 0 ... 1 1 ... 2 2 ...) with dim 11616
    labels = np.load(prefix + '/labels.npy')                            # labels of M nodes witn dim 4278, 0, 1 or 2
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')    # dict, keys = ['train_idx', 'val_idx', 'test_idx']
    return [[G00, G01], [G10, G11], [G20, G21]], \
           [[idx00, idx01], [idx10, idx11], [idx20, idx21]], \
           [features_0, features_1, features_2],\
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx


def load_DBLP_data(prefix='data/preprocessed/DBLP_processed'):
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02[3:]
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = np.load(prefix + '/features_2.npy')
    features_3 = np.eye(20, dtype=np.float32)

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [adjlist00, adjlist01, adjlist02], \
           [idx00, idx01, idx02], \
           [features_0, features_1, features_2, features_3],\
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx


def load_LastFM_data(prefix='data/preprocessed/LastFM_processed'):
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()
    in_file = open(prefix + '/0/0-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02
    in_file.close()
    in_file = open(prefix + '/1/1-0-1.adjlist', 'r')
    adjlist10 = [line.strip() for line in in_file]
    adjlist10 = adjlist10
    in_file.close()
    in_file = open(prefix + '/1/1-2-1.adjlist', 'r')
    adjlist11 = [line.strip() for line in in_file]
    adjlist11 = adjlist11
    in_file.close()
    in_file = open(prefix + '/1/1-0-0-1.adjlist', 'r')
    adjlist12 = [line.strip() for line in in_file]
    adjlist12 = adjlist12
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-1_idx.pickle', 'rb')
    idx10 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-2-1_idx.pickle', 'rb')
    idx11 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-0-1_idx.pickle', 'rb')
    idx12 = pickle.load(in_file)
    in_file.close()

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    train_val_test_pos_user_artist = np.load(prefix + '/train_val_test_pos_user_artist.npz')
    train_val_test_neg_user_artist = np.load(prefix + '/train_val_test_neg_user_artist.npz')

    return [[adjlist00, adjlist01, adjlist02],[adjlist10, adjlist11, adjlist12]],\
           [[idx00, idx01, idx02], [idx10, idx11, idx12]],\
           adjM, type_mask, train_val_test_pos_user_artist, train_val_test_neg_user_artist


# load skipgram-format embeddings, treat missing node embeddings as zero vectors
def load_skipgram_embedding(path, num_embeddings):
    count = 0
    with open(path, 'r') as infile:
        _, dim = list(map(int, infile.readline().strip().split(' ')))
        embeddings = np.zeros((num_embeddings, dim))
        for line in infile.readlines():
            count += 1
            line = line.strip().split(' ')
            embeddings[int(line[0])] = np.array(list(map(float, line[1:])))
    print('{} out of {} nodes have non-zero embeddings'.format(count, num_embeddings))
    return embeddings


# load metapath2vec embeddings
def load_metapath2vec_embedding(path, type_list, num_embeddings_list, offset_list):
    count = 0
    with open(path, 'r') as infile:
        _, dim = list(map(int, infile.readline().strip().split(' ')))
        embeddings_dict = {type: np.zeros((num_embeddings, dim)) for type, num_embeddings in zip(type_list, num_embeddings_list)}
        offset_dict = {type: offset for type, offset in zip(type_list, offset_list)}
        for line in infile.readlines():
            line = line.strip().split(' ')
            # drop </s> token
            if line[0] == '</s>':
                continue
            count += 1
            embeddings_dict[line[0][0]][int(line[0][1:]) - offset_dict[line[0][0]]] = np.array(list(map(float, line[1:])))
    print('{} node embeddings loaded'.format(count))
    return embeddings_dict


def load_glove_vectors(dim=50):
    print('Loading GloVe pretrained word vectors')
    file_paths = {
        50: 'data/wordvec/GloVe/glove.6B.50d.txt',
        100: 'data/wordvec/GloVe/glove.6B.100d.txt',
        200: 'data/wordvec/GloVe/glove.6B.200d.txt',
        300: 'data/wordvec/GloVe/glove.6B.300d.txt'
    }
    f = open(file_paths[dim], 'r', encoding='utf-8')
    wordvecs = {}
    for line in f.readlines():
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        wordvecs[word] = embedding
    print('Done.', len(wordvecs), 'words loaded!')
    return wordvecs


class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, hps, embed):
        self.hps = hps
        self.first_iter = True
        self.embed = embed
        self.worker_id = 0
        self.num_worker = 1

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = worker_info.num_workers, worker_info.id
        graph_dir = os.path.join('data/preprocessed', self.hps.dataset)
        if self.first_iter:
            self.first_iter = False
            logger.info("init the train dataset for the first time")
            # need to fast forward training data if the training procesjust begins
            return ExampleSet(num_workers, worker_id, graph_dir, self.embed, self.hps, fast_woward=True)
        else:
            logger.info("init the train dataset not for the first time")
            # else, no need to ff data, start from the first data
            return ExampleSet(num_workers, worker_id, graph_dir, self.embed, self.hps)


class ExampleSet():
    def __init__(self, num_workers, worker_id, graph_dir, embed, hps, fast_woward=False):
        self.num_workers = num_workers
        self.worker_id = worker_id
        self.graph_dir = graph_dir
        self.graph_data_folder_num = len([f for f in os.listdir(graph_dir) if 'train' in f])
        self.folder = None
        self.embed = embed
        self.hps = hps
        self.folder_i = -1
        self.data_no = 0
        self.folder_records = -1
        start_index = 0
        if fast_woward:
            start_index = hps.start_iteration * hps.batch_size
            logger.info("[INFO] fast-fowarding train dataset to %d" % start_index)
            while start_index > 0:
                self.folder_i += 1
                if self.folder_i >= self.graph_data_folder_num:
                    self.folder_i = 0
                self.folder = os.path.join(self.graph_dir, 'train'+str(self.folder_i))
                self.folder_records = len(os.listdir(self.folder))
                logger.info("[INFO] fast-forwarding data, folder %d has %d files" % (self.folder_i, self.folder_records))
                if start_index >= self.folder_records:
                    start_index -= self.folder_records
                    self.data_no += self.folder_records
                else:
                    break
        self.graph_i = start_index - 1
        self.data_no += self.graph_i
        if self.graph_i > 0:
            self.data_list = readJson(os.path.join('data/raw', hps.dataset, 'data/train'+str(self.folder_i)+'.json'))
            for i in range(self.graph_i + 1):
                self.data_fd.readline()
        logger.info("[INFO] starting at: data_no=%d, folder_i=%d, graph_i=%d" % (self.data_no, self.folder_i, self.graph_i))

    def __next__(self):
        while True:
            self.data_no += 1
            self.graph_i += 1
            while self.graph_i >= self.folder_records:
                self.folder_i += 1
                self.graph_i = 0
                if self.folder_i >= self.graph_data_folder_num:
                    raise StopIteration
                self.folder = os.path.join(self.graph_dir, 'train'+str(self.folder_i))
                self.folder_records = len(os.listdir(self.folder))
                self.data_list = readJson(os.path.join('data/raw', self.hps.dataset, 'data/train'+str(self.folder_i)+'.json'))
            if self.data_no % self.num_workers == self.worker_id:
                break

        # print('dataloader %d yielded datano %d, folder %d, graph %d' % (self.worker_id, self.data_no, self.folder_i, self.graph_i))
        data_dict = self.data_list[self.graph_i]
        graph_folder = os.path.join(self.folder, str(self.graph_i))
        get_data_result = get_data(data_dict, graph_folder, self.embed, self.hps.expected_metapaths)
        get_data_result.append(self.data_no)
        return get_data_result


class MapDataset(torch.utils.data.Dataset):
    """ Constructor: Dataset of example(object) for single document summarization"""

    def __init__(self, hps, embed, mode='val'):
        self.hps = hps
        self.mode = mode
        self.embed = embed
        self.graph_dir = os.path.join('data/preprocessed', self.hps.dataset)
        self.size = len(os.listdir(os.path.join(self.graph_dir, mode)))
        self.example_list = readJson(os.path.join('data/raw', hps.dataset, 'data', mode+'.json'))

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        data_dict = self.example_list[index]
        graph_folder = os.path.join(self.graph_dir, self.mode, str(index))
        get_data_result = get_data(data_dict, graph_folder, self.embed, self.hps.expected_metapaths)
        get_data_result.append(index)
        return get_data_result

    def get_example(self, index):
        text_dict = self.example_list[index]
        ret_dict = dict()
        ret_dict['labels'] = text_dict['label']
        ret_dict['original_abstract'] = '\n'.join(text_dict['summary'])
        if isinstance(text_dict['ori_text'], list) and isinstance(text_dict['ori_text'][0], list):
            ret_dict['original_article_sents'] = text_dict['ori_text'][len(text_dict['ori_text'])//2]        # only the center chapter can be extracted
        else:
            ret_dict['original_article_sents'] = text_dict['ori_text']
        return ret_dict

    def __len__(self):
        return self.size


def get_data(data_dict, graph_folder, embed, expected_metapaths):
    '''
    params:
        data_dict: dict, contains text, label, extractable, etc.
        graph_folder: the base folder of graph-related files.
        embed: torch.nn.Embedding
        expected_metapaths: list(list(list-formatted-metapath))
    returns:
        see return, all matrices about the wanted graph
    '''
    # 1. load g_lists
    g_lists = []        # 一条记录中所有metapath based graph组成的list(list)
    for mps in expected_metapaths:
        g_lists.append(list())
        for mp in mps:
            nx_G = nx.read_adjlist(os.path.join(graph_folder, '-'.join([str(i) for i in mp])+'.adjlist'),
                                   create_using=nx.MultiDiGraph)
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(nx_G.number_of_nodes())
            add_edges = sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges()))
            g.add_edges(u=[i[0] for i in add_edges], v=[i[1] for i in add_edges])
            g_lists[-1].append(g)

    # 2. load features (word embedding) 一条记录中每个node（包括token、sent、chap三种类型）的初始word embedding
    nid2wid = json.load(open(os.path.join(graph_folder, 'nid2wid.json'), encoding='utf-8'))
    wordids = torch.LongTensor([nid2wid[str(i)] for i in range(len(nid2wid))])
    token_embeddings = embed(wordids).numpy()
    # print('wordids', wordids.shape, wordids)
    adj_mat = scipy.sparse.load_npz(os.path.join(graph_folder, 'adjM.npz'))
    adj_mat_12to0 = adj_mat[wordids.shape[0]:, :wordids.shape[0]].todense()
    # print('adj_mat_12to0:', adj_mat_12to0.shape, adj_mat_12to0)
    sum_adj_mat = adj_mat_12to0.sum(axis=1)
    zero_sum_rows = np.where(sum_adj_mat == 0)[0]
    if zero_sum_rows.shape[0]:
        # logger.warning('zero division encountered at index {}'.format(zero_sum_rows))
        sum_adj_mat[zero_sum_rows] = 1      # solve zero division error
    # print('sum_adj_mat', sum_adj_mat.shape, sum_adj_mat)
    sum_adj_mat_invert = 1 / sum_adj_mat
    if zero_sum_rows.shape[0]:
        sum_adj_mat_invert[zero_sum_rows] = 0
    norm_diag_mat = np.diagflat(sum_adj_mat_invert)
    # print('norm_diag_mat', norm_diag_mat.shape, norm_diag_mat)
    adj_mat_12to0_normalized = norm_diag_mat.dot(adj_mat_12to0)
    # print('adj_mat_12to0_normalized: ', adj_mat_12to0_normalized.shape, adj_mat_12to0_normalized)
    sent_chap_embeddings = adj_mat_12to0_normalized.dot(token_embeddings)
    sent_num = sum([len(x) for x in data_dict['text']])
    sent_embeddings, chap_embeddings = np.asarray(sent_chap_embeddings[:sent_num, :]), np.asarray(sent_chap_embeddings[sent_num:, :])

    # 3. load type_mask
    type_mask = np.load(os.path.join(graph_folder, 'node_types.npy'))

    # 4. load edge_metapath_indices_lists 一条记录中，所有metapath路过的node组成的序列，list(list)
    edge_metapath_indices_lists = []
    for mps in expected_metapaths:
        edge_metapath_indices_lists.append(list())
        for mp in mps:
            node_idxs = np.load(os.path.join(graph_folder, '-'.join([str(i) for i in mp])+'_idx.npy'))
            edge_metapath_indices_lists[-1].append(node_idxs)

    # 5. extractable and label
    extractables = data_dict['extractable']
    labels = np.zeros(len(extractables))
    labels[data_dict['label']] = 1

    return [g_lists, edge_metapath_indices_lists, [token_embeddings, sent_embeddings, chap_embeddings], adj_mat, type_mask, extractables, labels]


def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def graph_collate_fn(samples):
    '''
    :param batch: (G, input_pad)
    :return:
    '''
    g_lists, edge_metapath_indices_lists, features_list, adj_mats, type_masks, extractables, labels, indexs = map(list, zip(*samples))
    # print('graph_collate index', indexs)
    combined_g_lists = []
    # merge dgl graph
    for i in range(len(g_lists[0])):
        combined_g_lists.append(list())
        for j in range(len(g_lists[0][i])):
            combined_g = dgl.batch([g_l[i][j] for g_l in g_lists])
            combined_g_lists[-1].append(combined_g)

    # merge type masks, labels, extractables
    graph_node_counts = [mask.shape[0] for mask in type_masks]
    combined_type_masks = np.concatenate(type_masks, axis=-1)
    combined_extractables = np.concatenate(extractables, axis=-1)
    combined_labels = np.concatenate(labels, axis=-1)
    combined_labels = combined_labels[np.where(combined_extractables == 1)[0]]  # label of extractable nodes
    combined_labels = torch.LongTensor(combined_labels)

    # merge edge metapath indices
    combined_metapath_indices_lists = []
    for i in range(len(edge_metapath_indices_lists[0])):
        combined_metapath_indices_lists.append(list())
        for j in range(len(edge_metapath_indices_lists[0][i])):
            processed_node_num = 0
            concat_list = []
            for k, metapaths_indices in enumerate(edge_metapath_indices_lists):
                metapath_indices = metapaths_indices[i][j]
                metapath_indices = metapath_indices + processed_node_num
                concat_list.append(metapath_indices)
                processed_node_num += graph_node_counts[k]
            combined_metapath_indice = np.concatenate(concat_list, axis=0)
            combined_metapath_indices_lists[-1].append(torch.LongTensor(combined_metapath_indice))

    # merge feature lists
    combined_feature_lists = []
    for i in range(len(features_list[0])):
        combined_feature = np.concatenate([feat[i] for feat in features_list])
        combined_feature_lists.append(torch.FloatTensor(combined_feature))

    # extractable nodes and labels
    combined_extractable_nodes = np.where(combined_type_masks == 1)[0]
    combined_extractable_nodes = combined_extractable_nodes[np.where(combined_extractables == 1)[0]]

    # return combined_g_lists, combined_metapath_indices_lists, combined_feature_lists, combined_type_masks, combined_extractable_nodes, combined_one_hot_labels
    return combined_g_lists, combined_metapath_indices_lists, combined_feature_lists, combined_type_masks, combined_extractable_nodes, combined_labels, indexs
