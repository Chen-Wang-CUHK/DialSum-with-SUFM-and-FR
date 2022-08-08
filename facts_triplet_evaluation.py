import os
import json
import argparse
from functools import partial
from nltk.stem.porter import *
from dataset.data_preprocess_utils import SpacyNLP, fact_extractor
from dataset.data_preprocess_utils import init_logger, logger

NLP_TOOL = SpacyNLP(whitespace_tokenizer_for_tokenizer=True)
my_fact_extractor = partial(fact_extractor, nlp_tool=NLP_TOOL)
my_stemmer = PorterStemmer()
SUM_SENT_SPLITTER = ['<q>']


def prepare_factTriplets(file_name, do_stem=True):
    total_factTriplets = []
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            line = ' '.join(line.strip().split())
            for w in SUM_SENT_SPLITTER:
                line = line.replace(w, ' ')
            facts_idx_str = my_fact_extractor(line)
            factIdxTriplets = json.loads(facts_idx_str)
            if len(factIdxTriplets) == 0:
                total_factTriplets.append([])
            else:
                word_list = NLP_TOOL.word_tokenize(line)
                facts_str_list = []
                for idx_triplet in factIdxTriplets:
                    fact_str = [word_list[idx_triplet['h']], word_list[idx_triplet['r']], word_list[idx_triplet['t']]]
                    if do_stem:
                        fact_str = [my_stemmer.stem(item) for item in fact_str]
                    fact_str = ' '.join(fact_str)
                    facts_str_list.append(fact_str)
                total_factTriplets.append(facts_str_list)
    return total_factTriplets


def prepare_factTriplets_from_splitFiles(path, suffix, do_stem=True):
    assert os.path.isdir(path), 'The path is invalid: {},'.format(path)
    file_num = len(os.listdir(path))
    logger.info('{} files are loaded for extracting fact triplets'.format(file_num))

    total_factTriplets = []
    for i in range(file_num):
        file_name = '{}.{}'.format(i, suffix)
        file_name = os.path.join(path, file_name)
        with open(file_name, encoding='utf-8') as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]
            line = ' '.join(lines).strip()
            line = ' '.join(line.strip().split())
            for w in SUM_SENT_SPLITTER:
                line = line.replace(w, ' ')
            facts_idx_str = my_fact_extractor(line)
            factIdxTriplets = json.loads(facts_idx_str)
            if len(factIdxTriplets) == 0:
                total_factTriplets.append([])
            else:
                word_list = NLP_TOOL.word_tokenize(line)
                facts_str_list = []
                for idx_triplet in factIdxTriplets:
                    fact_str = [word_list[idx_triplet['h']], word_list[idx_triplet['r']], word_list[idx_triplet['t']]]
                    if do_stem:
                        fact_str = [my_stemmer.stem(item) for item in fact_str]
                    fact_str = ' '.join(fact_str)
                    facts_str_list.append(fact_str)
                total_factTriplets.append(facts_str_list)
    return total_factTriplets


def exact_match_NPR(predict_facts, gold_facts):
    # remove duplicated predict facts and gold facts
    predict_facts = set(predict_facts)
    gold_facts = set(gold_facts)

    intersec = predict_facts & gold_facts

    precision = len(intersec) * 1.0 / len(predict_facts) if len(predict_facts) != 0 else 0.0
    recall = len(intersec) * 1.0 / len(gold_facts) if len(gold_facts) != 0 else 0.0

    # uniqe predict, gold, interset facts number
    num = (len(predict_facts), len(gold_facts), len(intersec))

    return {'predict_gold_interset_num': num, 'precision': precision, 'recall': recall}


def patial_match_NPR(predict_facts, gold_facts):
    # remove duplicated predict facts and gold facts
    predict_facts = set(predict_facts)
    predict_facts = [pf.strip().split() for pf in predict_facts]

    gold_facts = set(gold_facts)
    gold_facts = [gf.strip().split() for gf in gold_facts]

    partial_matched = [0] * len(predict_facts)
    gold_matched = [0] * len(gold_facts)
    for i, pf in enumerate(predict_facts):
        for j, gf in enumerate(gold_facts):
            pm_1 = pf[0] == gf[0] and pf[1] == gf[1]
            pm_2 = pf[1] == gf[1] and pf[2] == gf[2]
            pm_3 = pf[0] == gf[0] and pf[2] == gf[2]
            if (pm_1 or pm_2 or pm_3) and gold_matched[j] == 0:
                partial_matched[i] = 1
                gold_matched[j] = 1
                break

    precision = sum(partial_matched) * 1.0 / len(predict_facts) if len(predict_facts) != 0 else 0.0
    recall = sum(gold_matched) * 1.0 / len(gold_facts) if len(gold_facts) != 0 else 0.0

    # uniqe predict, gold, interset facts number
    num = (len(predict_facts), len(gold_facts), sum(partial_matched))

    return {'predict_gold_interset_num': num, 'precision': precision, 'recall': recall}


def ave_PRF(predict_facts_list, gold_facts_list, match_method='patial'):
    assert len(predict_facts_list) == len(gold_facts_list), 'The dialogue number of the predicted and the gold should be the same.'
    assert match_method in ['exact', 'patial']
    predict_num_list = []
    gold_num_list = []
    intersec_num_list = []

    precision_list = []
    recall_list = []
    for predict_facts, gold_facts in zip(predict_facts_list, gold_facts_list):
        if match_method == 'exact':
            result = exact_match_NPR(predict_facts=predict_facts, gold_facts=gold_facts)
        elif match_method == 'patial':
            result = patial_match_NPR(predict_facts=predict_facts, gold_facts=gold_facts)
        else:
            raise NotImplementedError
        predict_num_list.append(result['predict_gold_interset_num'][0])
        gold_num_list.append(result['predict_gold_interset_num'][1])
        intersec_num_list.append(result['predict_gold_interset_num'][2])

        precision_list.append(result['precision'])
        recall_list.append(result['recall'])

    logger.info('Num. of matched facts: {}'.format(sum(intersec_num_list)))
    logger.info('Num. of predicted facts: {}'.format(sum(predict_num_list)))
    logger.info('Num. of gold facts: {}'.format(sum(gold_num_list)))

    # macro averaged scores
    round_num = 3
    macro_ave_p = round(sum(precision_list) * 1.0 / len(precision_list), round_num)
    macro_ave_r = round(sum(recall_list) * 1.0 / len(recall_list), round_num)
    macro_ave_f = 0.000
    if (macro_ave_p + macro_ave_r) != 0:
        macro_ave_f = round(2 * macro_ave_p * macro_ave_r / (macro_ave_p + macro_ave_r), round_num)

    logger.info('macro_ave_p: {}, macro_ave_r: {}, macro_ave_f: {}'.format(macro_ave_p, macro_ave_r, macro_ave_f))

    # micro averged scores
    micro_ave_p = round(sum(intersec_num_list) * 1.0 / sum(predict_num_list), round_num)
    micro_ave_r = round(sum(intersec_num_list) * 1.0 / sum(gold_num_list), round_num)
    micro_ave_f = 0.000
    if (micro_ave_p + micro_ave_r) != 0:
        micro_ave_f = round(2 * micro_ave_p * micro_ave_r / (micro_ave_p + micro_ave_r), round_num)
    logger.info('micro_ave_p: {}, micro_ave_r: {}, micro_ave_f: {}'.format(micro_ave_p, micro_ave_r, micro_ave_f))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_summary_file", "-predict_summary_file", type=str, required=True)
    parser.add_argument("--gold_summary_file", "-gold_summary_file", type=str, required=True)
    parser.add_argument("--log_file", "-log_file", type=str, required=True)
    parser.add_argument("--dataset", "-dataset", type=str, required=True)
    parser.add_argument("--match_method", "-match_method", type=str, choices=['patial', 'exact'], default='patial')
    parser.add_argument("--no_stem", "-no_stem", action='store_true')
    parser.add_argument("--fast_abs_rl", "-fast_abs_rl", action='store_true')

    args = parser.parse_args()
    init_logger(args.log_file)
    stem = 'no_stem' if args.no_stem else 'do_stem'
    logger.info('Match method: {}, {}'.format(args.match_method, stem))

    do_stem = not args.no_stem
    if not args.fast_abs_rl:
        logger.info("Extracting predicted fact triplets from {}".format(args.predict_summary_file))
        predict_facts_list = prepare_factTriplets(args.predict_summary_file, do_stem=do_stem)
        logger.info("Extracting gold fact triplets from {}".format(args.gold_summary_file))
        gold_facts_list = prepare_factTriplets(args.gold_summary_file, do_stem=do_stem)
    else:
        logger.info("Extracting predicted fact triplets from {}".format(args.predict_summary_file))
        predict_facts_list = prepare_factTriplets_from_splitFiles(args.predict_summary_file, suffix='dec', do_stem=do_stem)
        logger.info("Extracting gold fact triplets from {}".format(args.gold_summary_file))
        gold_facts_list = prepare_factTriplets_from_splitFiles(args.gold_summary_file, suffix='ref', do_stem=do_stem)

    ave_PRF(predict_facts_list=predict_facts_list,
            gold_facts_list=gold_facts_list)
