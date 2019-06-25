import re
import argparse
import numpy as np
from pythonrouge.pythonrouge import Pythonrouge
from pprint import pprint
from numpy import random
from nltk.stem.porter import *


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r, t_trg):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out) / t_trg


def map2digit(reference, summary):
    lexicon = set()
    for line in reference:
        for tokens in line:
            assert isinstance(tokens, list) and len(tokens) == 1
            ref = tokens[0]
            for t in ref:
                if re.search(u'[\u4e00-\u9fff]', t):
                    lexicon.add(t)
    for s in summary:
        for line in s:
            assert isinstance(line, list) and len(line) == 1
            summ = line[0]
            for t in summ:
                if re.search(u'[\u4e00-\u9fff]', t):
                    lexicon.add(t)
    print('The total character size: %d' % len(lexicon))
    c2d = {}
    d2c = {}
    for i, value in enumerate(lexicon):
        c2d[value] = str(i)
        d2c[i] = value

    def map_string(text, c2d):
        '''
        "nihao你好" -> "nihao 3 5"  (assume c2d['你']=3, c2d['好']=5)
        :param text: string
        :param c2d: dict
        :return: string
        '''

        def spliteKeyWord(str):
            regex = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
            matches = re.findall(regex, str, re.UNICODE)
            return matches
        str_list = spliteKeyWord(text)
        return ' '.join([c2d[t] if re.search(u'[\u4e00-\u9fff]', t) else t for t in str_list])

    # map to digit
    res_ref = []
    res_summ = []
    for line in reference:
        tmp_s = []
        for tokens in line:
            assert isinstance(tokens, list) and len(tokens) == 1
            ref = tokens[0]  # string
            tmp = map_string(ref, c2d)
            tmp_s.append([tmp])
        res_ref.append(tmp_s)

    for s in summary:
        tmp_s = []
        for line in s:
            assert isinstance(line, list) and len(line) == 1
            summ = line[0]
            tmp = map_string(summ, c2d)
            tmp_s.append([tmp])
        res_summ.append(tmp_s)
    return res_ref, res_summ


def evaluate_func(opts):
    """
    calculate the macro-averaged precesion, recall and F1 score
    """

    target_file = open(opts.tgt, encoding='utf-8')
    target_lines = target_file.readlines()

    if opts.random:
        train_tgt_file = open(opts.train_tgt, encoding='utf-8')
        train_tgt_lines = train_tgt_file.readlines()
        train_tgts = []
        for line in train_tgt_lines:
            train_tgts.extend(line.strip().split(';'))
        preds_lines = []
        random.seed(opts.random)
        for i in range(len(target_lines)):
            sample_tgts = random.choice(train_tgts, 15)
            preds_lines.append(';'.join(sample_tgts))
    else:
        preds_file = open(opts.pred, encoding='utf-8')
        preds_lines = preds_file.readlines()

    # the number of examples should be the same
    assert len(target_lines) == len(preds_lines), \
        'tgt# %d should be equal to pred# %d' % (len(target_lines), len(preds_lines))

    file_len = len(target_lines)
    print('Total files number: %d' % file_len)

    correct_cnt = {}
    gold_cnt = 0
    for top_k in [1, 5, 10, 15]:
        correct_cnt[top_k] = 0

    ap_cnt = []
    trg_cnt = []
    stemmer = PorterStemmer()
    for pred, tgt in zip(preds_lines, target_lines):
        preds = pred.split(';')
        tgts = tgt.split(';')
        preds = [stemmer.stem(t.strip()) for t in preds if t.strip()]
        tgts = [stemmer.stem(t.strip()) for t in tgts if t.strip()]
        gold_cnt += len(tgts)
        trg_cnt.append(len(tgts))
        for top_k in [1, 5, 10, 15]:
            top_preds = preds[:top_k]
            for tag in tgts:
                if tag in top_preds:
                    correct_cnt[top_k] += 1

        tmp_cnt = [1 if t in tgts else 0 for t in preds]
        ap_cnt.append(tmp_cnt)

    for top_k in [1, 5, 10, 15]:
        ap_topk = []
        for item, t_cnt in zip(ap_cnt, trg_cnt):
            ap_topk.append(average_precision(item[:top_k], t_cnt))

        acc = correct_cnt[top_k] / float(top_k * file_len)
        recall = correct_cnt[top_k] / float(gold_cnt)
        if acc == 0 or recall == 0:
            f1 = 0
        else:
            f1 = 2 * acc * recall / (acc + recall)

        print('The top_k %d: %.6f acc, %.6f recall, %.6f f measure,  %.6f map' %
              (top_k, acc, recall, f1, np.mean(ap_topk)))

    return

    # begin to compute rouge
    reference = [[[stemmer.stem(tokens.strip())] for tokens in line.split(';')] for line in target_lines]
    summary = [[[stemmer.stem(line.split(';')[top_k].strip())] for line in preds_lines] for top_k in range(0, 1)]

    # map chinese characters into digits for weibo dataset
    if opts.filter_chinese:
        reference, summary = map2digit(reference, summary)

    for top_k in range(0, 1):
        rouge = Pythonrouge(summary_file_exist=False,
                            summary=summary[top_k], reference=reference,
                            n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
                            recall_only=False, stemming=True, stopwords=False,
                            word_level=False, length_limit=True, length=50,
                            use_cf=True, cf=95, scoring_formula='average',
                            resampling=True, samples=1000, favor=True, p=0.5)
        score = rouge.calc_score()
        print('\nROUGE-top %d' % (top_k + 1))
        pprint(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('-filter_chinese', type=int, default=1)
    parser.add_argument('-tgt', type=str, required=True)
    parser.add_argument('-train_tgt', type=str)
    parser.add_argument('-pred', type=str, required=True)
    parser.add_argument('-random', type=int, default=0)

    opts = parser.parse_args()

    evaluate_func(opts=opts)


