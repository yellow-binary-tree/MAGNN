# coding=utf-8
import os
import json
import pathlib
import argparse
import threading

import scipy
import numpy as np
import networkx as nx

import utils
from utils.vocabulary import Vocab
from utils.preprocess import get_metapath_neighbor_pairs, get_networkx_graph, get_edge_metapath_idx_array


# Public Resources
VOCAB, TFIDF_W, FILTERIDS = None, None, None
expected_metapaths = [
    [(0, 1, 0), (0, 2, 0), (0, 1), (0, 2)],
    [(1, 0, 1), (1, 2, 1), (1, 0), (1, 2)],
    [(2, 0, 2), (2, 1, 2), (2, 0), (2, 1)]
]


# utils
def read_text(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data


def cat_doc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res


# threading
class ProcessThread(threading.Thread):
    def __init__(self, src_data_file, src_w2s_file, src_w2d_file, dest_folder, hps, _id=-1):
        threading.Thread.__init__(self)
        self.src_data_fd = open(src_data_file, encoding='utf-8')
        self.src_w2s_fd = open(src_w2s_file, encoding='utf-8')
        self.src_w2d_fd = open(src_w2d_file, encoding='utf-8')
        self.dest_folder = dest_folder
        self.hps = hps
        self._id = _id

    def run(self):
        print('[graph] start thread: %d' % self._id)

        for i, data_line in enumerate(self.src_data_fd):
            data_dict = json.loads(data_line)
            w2s_dict = json.loads(self.src_w2s_fd.readline())
            w2d_dict = json.loads(self.src_w2d_fd.readline())
            type_mask, adj_mat, G_lists, all_edge_metapath_idx_arrays, nid2wid = self.preprocess_chapter(data_dict, w2s_dict, w2d_dict)
            # TODO: save _ to dest_folder

            save_base_folder = os.path.join(self.dest_folder, str(i))
            pathlib.Path(save_base_folder).mkdir(parents=True, exist_ok=True)
            np.save(os.path.join(save_base_folder, 'node_types.npy'), type_mask)
            scipy.sparse.save_npz(os.path.join(save_base_folder, 'adjM.npz'), scipy.sparse.csr_matrix(adj_mat))
            for i, G_list in enumerate(G_lists):
                for G, metapath in zip(G_list, expected_metapaths[i]):
                    nx.write_adjlist(G, os.path.join(save_base_folder, '-'.join(map(str, metapath)) + '.adjlist'))
            for i, all_edge_metapath_idx_array in enumerate(all_edge_metapath_idx_arrays):
                for edge_metapath_idx_array, metapath in zip(all_edge_metapath_idx_array, expected_metapaths[i]):
                    np.save(os.path.join(save_base_folder, '-'.join(map(str, metapath)) + '_idx.npy'), edge_metapath_idx_array)
            with open(os.path.join(save_base_folder, 'nid2wid.json'), 'w', encoding='utf-8') as f:
                json.dump(nid2wid, f)

        print('[graph] finish thread: %d' % self._id)

    def preprocess_chapter(self, data_dict, w2s_dict, w2d_dict):
        if isinstance(data_dict['text'], list) and isinstance(data_dict['text'][0], list):
            text = cat_doc(data_dict['text'])
        else:
            text = data_dict['text']
        text_token_ids = [[VOCAB.word2id(token) for token in sent.split(' ')] for sent in text]
        wid2nid, nid2wid = self.get_word_node(text_token_ids)
        token_nodes_num = len(list(nid2wid.keys()))
        sent_nodes_num = len(text)
        chap_nodes_num = len(data_dict['text'])
        dim = token_nodes_num + sent_nodes_num + chap_nodes_num

        # 1. type_mask. token->0, sentnece->1, document->2
        type_mask = type_mask = np.zeros((dim), dtype=int)
        type_mask[token_nodes_num: token_nodes_num + sent_nodes_num] = 1
        type_mask[token_nodes_num + sent_nodes_num:] = 2

        # 2. adjacent matrix
        adj_mat = np.zeros((dim, dim), dtype=float)
        # token-sent edge
        for sentid in w2s_dict:
            for token in w2s_dict[sentid]:
                tokenid = VOCAB.word2id(token)
                if tokenid in wid2nid.keys():
                    adj_mat[wid2nid[tokenid], token_nodes_num + int(sentid)] = w2s_dict[sentid][token]
                    adj_mat[token_nodes_num + int(sentid), wid2nid[tokenid]] = w2s_dict[sentid][token]
        # token-chap edge
        for chapid in w2d_dict:
            min_tfidf_weight = min(list(w2d_dict[chapid].values()))
            for token in w2d_dict[chapid]:
                tokenid = VOCAB.word2id(token)
                if tokenid in wid2nid.keys():
                    if w2d_dict[chapid][token] != min_tfidf_weight:
                        # filter edges that token only appear once in the chapter, or there can be too much token-chap edges.
                        adj_mat[wid2nid[tokenid], token_nodes_num + sent_nodes_num + int(chapid)] = w2d_dict[chapid][token]
                        adj_mat[token_nodes_num + sent_nodes_num + int(chapid), wid2nid[tokenid]] = w2d_dict[chapid][token]

        # sent-chap edge
        chap_lens = [len(chap) for chap in data_dict['text']]
        for i, sents in enumerate(data_dict['text']):
            sents_visited = sum(chap_lens[:i])
            for j, sent in enumerate(sents):
                adj_mat[token_nodes_num + sent_nodes_num + i, token_nodes_num + sents_visited + j] = 1
                adj_mat[token_nodes_num + sents_visited + j, token_nodes_num + sent_nodes_num + i] = 1

        # 3. metapath based graph
        G_lists = []
        all_edge_metapath_idx_arrays = []
        for i in range(len(expected_metapaths)):
            neighbor_pairs = get_metapath_neighbor_pairs(adj_mat, type_mask, expected_metapaths[i])
            G_lists.append(get_networkx_graph(neighbor_pairs, type_mask, i))
            all_edge_metapath_idx_arrays.append(get_edge_metapath_idx_array(neighbor_pairs))

        return type_mask, adj_mat, G_lists, all_edge_metapath_idx_arrays, nid2wid

    def get_word_node(self, text_token_ids):
        wid2nid, nid2wid, nid = {}, {}, 0
        for sent in text_token_ids:
            for tokenid in sent:
                if tokenid not in FILTERIDS and tokenid not in wid2nid.keys():
                    wid2nid[tokenid] = nid
                    nid2wid[nid] = tokenid
                    nid += 1
        return wid2nid, nid2wid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='qidian_1118_winsize1', help='dataset name')
    parser.add_argument('--src_folder', type=str, default=None, help='data and cache src folder')
    parser.add_argument('--dest_folder', type=str, default=None, help='MAGNN graph dest folder')
    parser.add_argument('--vocab_size', type=int, default=100000, help='Size of vocabulary. [default: 100000]')
    parser.add_argument('--num_proc', type=int, default=1, help='num of processes.')
    parser.add_argument('--no_proc', type=int, default=1, help='no. of this process.')
    args = parser.parse_args()

    data_folder = os.path.join(args.src_folder, 'data')
    cache_folder = os.path.join(args.src_folder, 'cache')
    dest_folder = os.path.join(args.dest_folder)
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    # Load Public Resources
    VOCAB = Vocab(os.path.join(cache_folder, 'vocab'), args.vocab_size)
    TFIDF_W = read_text(os.path.join(cache_folder, 'filter_word.txt'))
    FILTERWORDS = [line.strip() for line in open('baidu_stopwords.txt', encoding='utf-8').readlines()]
    FILTERIDS = [VOCAB.word2id(w.lower()) for w in FILTERWORDS]
    FILTERIDS.append(VOCAB.word2id("[PAD]"))   # keep "[UNK]" but remove "[PAD]"
    FILTERIDS = set(FILTERIDS)
    print('[graph] VOCAB, TFIDF_W and FILTERIDS loaded.')

    threads = []
    filenames = [f for f in os.listdir(data_folder)]
    filenames.sort()
    for i, filename in enumerate(filenames):
        if i % args.num_proc != args.no_proc - 1:
            continue
        if not os.path.exists(os.path.join(dest_folder, filename.replace('.json', ''))):
            os.mkdir(os.path.join(dest_folder, filename.replace('.json', '')))
        threads.append(ProcessThread(
            src_data_file=os.path.join(data_folder, filename),
            src_w2s_file=os.path.join(cache_folder, 'w2s', filename),
            src_w2d_file=os.path.join(cache_folder, 'w2d', filename),
            dest_folder=os.path.join(dest_folder, filename.replace('.json', '')),
            hps=args, _id=i
        ))
    for i, thread in enumerate(threads):
        thread.start()
    for i, thread in enumerate(threads):
        thread.join()
