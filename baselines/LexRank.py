import sys
import os
import argparse

from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from lexrank.utils.text import tokenize
from path import Path

# change to your DialogueSum folder
sys.path.insert(1, "/home/rickywchen/code/DialogueSum")
UTR_SPLITTER = "|"
DIAL_TOKEN_NUM_TH = 15


def lexrank_setup(opts):
    corpus_file = open(opts.corpus_file, encoding='utf-8')
    corpus_lines = corpus_file.readlines()
    # filter the corpus lines with less than 15 tokens
    corpus_lines = [line.strip() for line in corpus_lines if len(line.strip().split()) >= DIAL_TOKEN_NUM_TH]

    corpus_documents = []
    for line in corpus_lines:
        utr_list = [utr.strip() for utr in line.strip().split(UTR_SPLITTER) if len(utr.strip()) > 0]
        corpus_documents.append(utr_list)

    lxr = LexRank(corpus_documents, stopwords=STOPWORDS['en'])
    return lxr


def lexrank_func(opts, lxr):
    dial_file = open(opts.dial_file, encoding='utf-8')
    saved_sum_file = open(opts.saved_sum_file, 'w', encoding='utf-8')

    for line in dial_file:
        utr_list = [utr.strip() for utr in line.split(UTR_SPLITTER) if len(utr.strip()) > 0]
        summary = lxr.get_summary(utr_list, summary_size=opts.selected_num, threshold=opts.threshold)
        summary = '{}'.format(UTR_SPLITTER).join(summary)
        saved_sum_file.write(summary.strip() + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LexRank.py")
    parser.add_argument("--corpus_file", "-corpus_file", type=str, required=True,
                        help="The corpus dialogue file.")
    parser.add_argument("--dial_file", "-dial_file", type=str, required=True,
                        help="The dialogue file.")
    parser.add_argument("--selected_num", "-selected_num", type=int, default=2,
                        help="The number of selected utterances.")
    parser.add_argument("--threshold", "-threshold", type=int, default=None,
                        help="The threshold of LexRank for selecting utterances.")
    parser.add_argument("--saved_sum_file", "-saved_sum_file", type=str,
                        help="The file name for saving the extracted summary by the LexRank baseline.")
    opts = parser.parse_args()

    lxr = lexrank_setup(opts)

    lexrank_func(opts, lxr)