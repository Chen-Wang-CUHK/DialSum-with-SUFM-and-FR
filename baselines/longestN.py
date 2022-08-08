import sys
import os
import argparse

# change to your DialogueSum folder
sys.path.insert(1, "/home/rickywchen/code/DialogueSum")
UTR_SPLITTER = "|"


def longest_n(dial_line, n=3):
    """
    Extract the longest n utterances from the dialogue as the summary
    :param dial_line: str, the dialogue string
    :param n: int,
    :return: the longest n utterances
    """

    dial_line = dial_line.strip()
    utr_list = [utr for utr in dial_line.split(UTR_SPLITTER) if len(utr.strip()) > 0]
    utr_len_tuple_list = [(utr.strip(), len(utr.strip().split())) for utr in utr_list]
    sorted_utr_len_tuple_list = sorted(utr_len_tuple_list, key=lambda x: x[1], reverse=True)

    extracted_sum = [utr[0].strip() for utr in sorted_utr_len_tuple_list[:n]]
    extracted_sum = '{}'.format(UTR_SPLITTER).join(extracted_sum)

    return extracted_sum


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="longestN.py")
    parser.add_argument("--dial_file", "-dial_file", type=str, required=True,
                        help="The dialogue file.")
    parser.add_argument("--selected_num", "-selected_num", type=int, default=3,
                        help="The number of selected longest utterances.")
    parser.add_argument("--saved_sum_file", "-saved_sum_file", type=str,
                        help="The file name for saving the extracted summary by the LONGEST-N baseline.")
    opts = parser.parse_args()

    print("Dialogue file: {}".format(opts.dial_file))
    dial_file = open(opts.dial_file, encoding='utf-8')
    print("Save the extracted longest-{} summaries to {}".format(opts.selected_num, opts.saved_sum_file))
    saved_sum_file = open(opts.saved_sum_file, 'w', encoding='utf-8')

    for dial_line in dial_file:
        extracted_sum = longest_n(dial_line, opts.selected_num)
        saved_sum_file.write(extracted_sum.strip() + '\n')