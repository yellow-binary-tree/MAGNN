import os
import sys
import time
import json
import datetime
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from rouge import Rouge

from utils.pytorchtools import EarlyStopping
from utils.data import IterDataset, MapDataset, graph_collate_fn
from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from utils.vocabulary import Vocab, result_word2id
from utils.logger import logger

from tester import SLTester
from model import MAGNN_nc
from model.embedding import WordEmbeddingLoader

# exp_uploader
sys.path.append('/share/wangyq/tools/')
import exp_uploader
import rouge_server


def save_model(model, save_file):
    with open(save_file, 'wb') as f:
        torch.save(model.state_dict(), f)
    logger.info('[INFO] Saved model to %s', save_file)


def run_model_qidian(model, train_loader, valid_loader, valid_dataset, hps):
    logger.info("[INFO] Starting run_training")
    optimizer = torch.optim.Adam(model.parameters(), lr=hps.lr, weight_decay=hps.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # exp_uploader
    exp = exp_uploader.Exp(proj_name=hps.proj_name, exp_name=hps.exp_name, command=str(hps))
    exp_uploader.init_exp(exp)

    best_train_loss = None
    best_loss = None
    best_F = None
    non_descent_cnt = 0
    saveNo = 0
    iters_elapsed = hps.start_iteration

    model_save_dir = os.path.join('save', hps.dataset+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    try:
        for epoch in range(1, hps.epoch + 1):
            epoch_start_time = time.time()
            iters_elapsed_in_epoch = 0
            epoch_loss = 0.0
            train_loss = 0.0
            for i, data in enumerate(train_loader):
                iter_start_time = time.time()
                model.train()
                iters_elapsed_in_epoch += 1
                iters_elapsed += 1

                combined_g_lists, combined_metapath_indices_lists, combined_feature_lists, combined_type_masks, combined_extractable_nodes, combined_labels, indexs = data

                # print('inedxs', indexs)
                # print('combined_g_lists', combined_g_lists)
                # for i, arrs in enumerate(combined_metapath_indices_lists):
                #     for j, arr in enumerate(arrs):
                #         print('combined_metapath_indices_lists', i, j, arr.shape, arr)
                # for i, arr in enumerate(combined_feature_lists):
                #     print('combined_feature_lists', i, arr.shape)
                # print('combined_type_masks', combined_type_masks.shape, combined_type_masks.sum())
                # print('combined_extractable_nodes', combined_extractable_nodes.shape, combined_extractable_nodes)
                # print('combined_labels', combined_labels.shape, combined_labels)

                combined_feature_lists = [cfl.to(device) for cfl in combined_feature_lists]
                combined_metapath_indices_lists = [[indices.to(device) for indices in indices_list] for indices_list in combined_metapath_indices_lists]
                combined_labels = combined_labels.to(device)
                inputs = combined_g_lists, combined_feature_lists, combined_type_masks, combined_metapath_indices_lists
                logits, embeddings = model(inputs, combined_extractable_nodes)
                # print('logits', logits.shape, logits)
                # print('combined_labels', combined_labels.shape, combined_labels)

                loss = criterion(logits, combined_labels).mean()
                train_loss += float(loss.data)
                epoch_loss += float(loss.data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iters_elapsed % hps.report_every == 0:
                    logger.info('       | end of iter {:3d} | time: {:5.2f}s | train loss {:5.4f} | '
                                .format(iters_elapsed, (time.time() - iter_start_time), float(train_loss)))
                    train_loss = 0.0

                if iters_elapsed % hps.eval_after_iterations == 0:
                    save_model(model, os.path.join(model_save_dir, 'train_iter_'+str(iters_elapsed)))
                    best_loss, best_F, non_descent_cnt, saveNo = run_eval(model, valid_loader, valid_dataset, hps, best_loss, best_F, non_descent_cnt, saveNo, iters_elapsed, model_save_dir, exp)
                    if non_descent_cnt >= 3:
                        logger.error("[Error] val loss does not descent for three times. Stopping supervisor...")
                        save_model(model, os.path.join(model_save_dir, "earlystop"))
                        return
                time6 = time.time()
                logger.debug('[DEBUG] iter %d, total time %.5f' % (iters_elapsed, (time6-iter_start_time)))

            epoch_avg_loss = epoch_loss / (iters_elapsed_in_epoch * hps.batch_size)
            logger.info('   | end of epoch {:3d} | time: {:5.2f}s | epoch loss: {:5.2f}'
                        .format(epoch, (time.time() - epoch_start_time), epoch_avg_loss))

            if not best_train_loss or epoch_avg_loss < best_train_loss:
                save_file = os.path.join(model_save_dir, "bestmodel")
                logger.info('[INFO] Found new best model with %.3f running_train_loss. Saving to %s', float(epoch_avg_loss),
                            save_file)
                save_model(model, save_file)
                best_train_loss = epoch_avg_loss
            elif epoch_avg_loss >= best_train_loss:
                logger.error("[Error] training loss does not descent. Stopping supervisor...")
                save_model(model, os.path.join(model_save_dir, "earlystop"))
                sys.exit(1)
    except Exception as e:
        save_model(model, os.path.join(model_save_dir, 'exception'))
        raise e


def run_eval(model, valid_loader, valid_dataset, hps, best_loss, best_F, non_descent_cnt, saveNo, iters_elapsed, model_save_dir, exp):
    logger.info("[INFO] Starting eval for this model ...")

    if hps.use_exp_rouge:
        test_vocab = Vocab(os.path.join('data/raw', hps.dataset, 'cache/test_vocab'), max_size=-1)

    model.eval()
    iter_start_time = time.time()

    with torch.no_grad():
        tester = SLTester(model, hps, exp)
        for i, data in enumerate(valid_loader):
            combined_g_lists, combined_metapath_indices_lists, combined_feature_lists, combined_type_masks, combined_extractable_nodes, combined_labels, indexs = data
            combined_feature_lists = [cfl.to(device) for cfl in combined_feature_lists]
            combined_metapath_indices_lists = [[indices.to(device) for indices in indices_list] for indices_list in combined_metapath_indices_lists]
            combined_labels = combined_labels.to(device)
            inputs = combined_g_lists, combined_feature_lists, combined_type_masks, combined_metapath_indices_lists
            tester.evaluation(inputs, combined_extractable_nodes, combined_labels, indexs, valid_dataset)

            if i % 20 == 0:
                exp_uploader.async_heart_beat(exp)

    running_avg_loss = tester.running_avg_loss
    if len(tester.hyps) == 0 or len(tester.refer) == 0:
        logger.error("During testing, no hyps is selected!")
        return

    if hps.use_exp_rouge:
        exp_server_hyps, exp_server_refer = result_word2id(
            test_vocab, [chap.split('\n') for chap in tester.hyps], [chap.split('\n') for chap in tester.refer])
        rouge_server.eval_rouge(hps.proj_name, hps.exp_name, 'decode_test_ckpt-{}'.format(iters_elapsed),
                                exp_server_hyps, exp_server_refer)

    rouge = Rouge()
    scores_all = rouge.get_scores(tester.hyps, tester.refer, avg=True)
    logger.info('[INFO] End of valid | time: {:5.2f}s | valid loss {:5.4f} | ' .format((time.time() - iter_start_time), float(running_avg_loss)))

    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
        scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
        + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
        scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
        + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
        scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    logger.info('\n' + res)

    tester.getMetric()
    F_metric = tester.labelMetric

    if best_loss is None or running_avg_loss < best_loss:
        bestmodel_save_path = os.path.join(model_save_dir, 'bestmodel_%d' % (saveNo % 3))  # this is where checkpoints of best models are saved
        if best_loss is not None:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is %.6f, Saving to %s',
                float(running_avg_loss), float(best_loss), bestmodel_save_path)
        else:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is None, Saving to %s',
                float(running_avg_loss), bestmodel_save_path)
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        best_loss = running_avg_loss
        non_descent_cnt = 0
        saveNo += 1
    else:
        non_descent_cnt += 1

    if best_F is None or best_F < F_metric:
        bestmodel_save_path = os.path.join(model_save_dir, 'bestFmodel')  # this is where checkpoints of best models are saved
        if best_F is not None:
            logger.info('[INFO] Found new best model with %.6f F. The original F is %.6f, Saving to %s', float(F_metric),
                        float(best_F), bestmodel_save_path)
        else:
            logger.info('[INFO] Found new best model with %.6f F. The original F is None, Saving to %s', float(F_metric),
                        bestmodel_save_path)
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        best_F = F_metric

    return best_loss, best_F, non_descent_cnt, saveNo


def run_test_qidian(model, test_loader, test_dataset, hps):
    test_dir = os.path.join('save', hps.save_root, "test")      # make a subdir of the root dir for eval data
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    log_fname = os.path.join(test_dir, hps.dataset + '_' + hps.restore_model)
    log_fd = open(log_fname, 'w', encoding='utf-8')

    model.load_state_dict(torch.load(os.path.join('save', hps.save_root, hps.restore_model)))
    model.eval()

    # exp_uploader
    exp = exp_uploader.Exp(proj_name=hps.proj_name, exp_name=hps.exp_name, command=str(hps))
    exp_uploader.init_exp(exp)
    if hps.use_exp_rouge:
        test_vocab = Vocab(os.path.join('data/raw', hps.dataset, 'cache/test_vocab'), max_size=-1)

    start_time = time.time()
    with torch.no_grad():
        logger.info("[Model] Sequence Labeling!")
        tester = SLTester(model, hps, exp, test_dir=test_dir)
        for j, data in enumerate(test_loader):
            combined_g_lists, combined_metapath_indices_lists, combined_feature_lists, combined_type_masks, combined_extractable_nodes, combined_labels, indexs = data
            combined_feature_lists = [cfl.to(device) for cfl in combined_feature_lists]
            combined_metapath_indices_lists = [[indices.to(device) for indices in indices_list] for indices_list in combined_metapath_indices_lists]
            combined_labels = combined_labels.to(device)
            inputs = combined_g_lists, combined_feature_lists, combined_type_masks, combined_metapath_indices_lists
            pred_idxs, hypss, refers = tester.evaluation(inputs, combined_extractable_nodes, combined_labels, indexs, valid_dataset)
            for i, pred_idx, hyps, refer in zip(indexs, pred_idxs, hypss, refers):
                log_fd.write(json.dumps({'index': i, 'pred_idx': pred_idx, 'hyps': hyps, 'refer': refer}, ensure_ascii=False) + '\n')
            if j % 20 == 0:
                exp_uploader.async_heart_beat(exp)

    running_avg_loss = tester.running_avg_loss
    logger.info("The number of pairs is %d", tester.rougePairNum)
    if not tester.rougePairNum:
        logger.error("During testing, no hyps is selected!")
        sys.exit(1)

    if hps.use_exp_rouge:
        exp_server_hyps, exp_server_refer = result_word2id(
            test_vocab, [chap.split('\n') for chap in tester.hyps], [chap.split('\n') for chap in tester.refer])
        rouge_server.eval_rouge(hps.proj_name, hps.exp_name, 'decode_test_ckpt-{}'.format(hps.restore_model.split('_')[-1]),
                                exp_server_hyps, exp_server_refer)

    rouge = Rouge()
    scores_all = rouge.get_scores(tester.hyps, tester.refer, avg=True)

    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
        + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
        + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    logger.info(res)
    logger.info('[INFO] End of test | time: {:5.2f}s | test loss {:5.4f} | '.format((time.time() - start_time), float(running_avg_loss)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')

    # run mode
    parser.add_argument('--mode', type=str, default='train', help='running mode (train | test)')
    parser.add_argument('--restore_model', type=str, default=None, help='model to restore (only works in test mode)')
    parser.add_argument('--save_root', type=str, default=None, help='save root (only works in test mode)')

    # model
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of the GNN layers. Default is 2.')
    parser.add_argument('--expected_metapaths', default='010,020,101,121,202,212', help='metapaths to use')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    parser.add_argument('--attn_vec_dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    parser.add_argument('--rnn_type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    parser.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    parser.add_argument('--save_postfix', default='qidian', help='Postfix for the saved model and result.')

    # resources
    parser.add_argument('--embedding_path', type=str, help='word embedding file path')
    parser.add_argument('--vocab_size', type=int, default=100000, help='vocab size')
    parser.add_argument('--word_emb_dim', type=int, default=200, help='Word embedding size')

    # data
    parser.add_argument('--dataset', type=str, default='winsize1', help='dataset_name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')

    # training
    parser.add_argument('-m', type=int, default=5, help='decode summary length')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--start_iteration', type=int, default=0, help='From which iteration to restart training')
    parser.add_argument('--report_every', type=int, default=100, help='report every itereation')
    parser.add_argument('--eval_after_iterations', type=int, default=4000, help='eval after itereations')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. [default: 0]')
    parser.add_argument('--train_num_workers', type=int, default=0, help='num of workers of DataLoader. [default: 4]')
    parser.add_argument('--eval_num_workers', type=int, default=0, help='num of workers of DataLoader. [default: 4]')

    # log
    parser.add_argument('--proj_name', type=str, default='wyq_structural_summ', help='Project Name')
    parser.add_argument('--exp_name', type=str, default='MAGNN', help='Experiment Name')
    parser.add_argument('--use_exp_rouge', type=bool, default=True, help='whether send decoded summ to the exp_server to get rouge')

    args = parser.parse_args()

    expected_metapaths = args.expected_metapaths.split(',')
    start_types = sorted(list(set([int(mp[0]) for mp in expected_metapaths])))
    expected_metapaths_list = [list() for i in range(max(start_types) + 1)]
    for metapath in expected_metapaths:
        expected_metapaths_list[int(metapath[0])].append([int(i) for i in metapath])
    args.expected_metapaths = expected_metapaths_list
    hps = args
    logger.info(hps)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.info("Pytorch %s", torch.__version__)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # vocab and embedding
    VOCAL_FILE = os.path.join('data', 'raw', args.dataset, 'cache', "vocab")
    logger.info("[INFO] Create Vocab, vocab path is %s", VOCAL_FILE)
    vocab = Vocab(VOCAL_FILE, args.vocab_size)

    embed = torch.nn.Embedding(vocab.size(), args.word_emb_dim, padding_idx=0)
    embed_loader = WordEmbeddingLoader(args.embedding_path, vocab)
    vectors = embed_loader.load_my_vecs(args.word_emb_dim)
    pretrained_weight = embed_loader.add_unknown_words_by_avg(vectors, args.word_emb_dim)
    embed.weight.data.copy_(torch.Tensor(pretrained_weight))
    embed.weight.requires_grad = False

    # dataset
    dataset = IterDataset(hps, embed)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=False, num_workers=args.train_num_workers, collate_fn=graph_collate_fn, pin_memory=True)
    del dataset
    if args.mode == 'train':
        valid_dataset = MapDataset(hps, embed, mode='val')
    elif args.mode == 'test':
        valid_dataset = MapDataset(hps, embed, mode='test')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False, collate_fn=graph_collate_fn, num_workers=args.eval_num_workers, pin_memory=True)

    # model
    model = MAGNN_nc(num_layers=args.num_layers, num_metapaths_list=[len(m) for m in args.expected_metapaths], num_edge_type=6,
                     etypes_lists=args.expected_metapaths, feats_dim_list=[args.word_emb_dim]*3, hidden_dim=args.hidden_dim, out_dim=2,
                     num_heads=args.num_heads, attn_vec_dim=args.attn_vec_dim, rnn_type=args.rnn_type, dropout_rate=args.dropout_rate)
    model.to(device)

    if args.mode == 'train':
        run_model_qidian(model, train_loader, valid_loader, valid_dataset, hps)
    elif args.mode == 'test':
        run_test_qidian(model, valid_loader, valid_dataset, hps)
