import json
import torch
import torch.nn as nn

from onmt.utils.misc import aeq
from onmt.modules import GlobalAttentionPlus
from onmt.utils.misc import aeq


class CorefMergeLayer(nn.Module):
    def __init__(self, hidden_size, merge_type='attn', fnn_all=False):
        super(CorefMergeLayer, self).__init__()
        self.fnn_all = fnn_all
        self.merge_type = merge_type

        if merge_type in ['attn', 'attn_noself']:
            self.attn = GlobalAttentionPlus(hidden_size, hidden_size, attn_type="mlp")
        elif merge_type in ['attn_weighted', 'attn_noself_weighted']:
            self.attn = GlobalAttentionPlus(hidden_size, hidden_size, attn_type="mlp", score_feat_dim=1)
        elif merge_type == 'attn_no_extra_params':
            assert not fnn_all
            self.attn = GlobalAttentionPlus(hidden_size, hidden_size, attn_type="dot")
        else:
            self.attn = None

        if merge_type != 'attn_no_extra_params':
            self.mrg_linear = nn.Linear(2 * hidden_size, hidden_size) if merge_type != 'none' else None
            self.mrg_acti_func = nn.Tanh() if merge_type != 'none' else None

    def forward(self, m_bank, corefs):
        if self.merge_type == 'none':
            return m_bank
        src_len, b_size, h_size = m_bank.size()
        mrgd_m_bank = []
        b_coref_posi, b_coref_score = self.preprocess(corefs)
        for ex_idx, (ex_coref_posi, ex_coref_score) in enumerate(zip(b_coref_posi, b_coref_score)):
            # [src_len, h_size]
            ex_m_bank = m_bank[:, ex_idx, :]
            mrgd_coref_m_bank = {}
            for clustr_coref_posi, clustr_coref_score in zip(ex_coref_posi, ex_coref_score):
                if len(clustr_coref_posi) > 0:
                    # [clustr_size, h_size]
                    clustr_m_bank = ex_m_bank.index_select(0, torch.LongTensor(clustr_coref_posi).to(ex_m_bank.device))
                    clustr_size = clustr_m_bank.size(0)
                    # [cluster_size, clustr_size]
                    clustr_scores = torch.Tensor(clustr_coref_score).to(ex_m_bank.device)
                    if self.merge_type == 'max_pool':
                        # [1, h_size]
                        clustr_representation, _ = clustr_m_bank.max(0, keepdim=True)
                        # [clustr_size, h_size]
                        clustr_representation = clustr_representation.expand(len(clustr_coref_posi), -1)
                    elif self.merge_type in ['attn', 'attn_no_extra_params']:
                        # [clustr_size, 1, h_size], [clustr_size, 1, clustr_size]
                        clustr_representation, attn_scores = self.attn(source=clustr_m_bank.unsqueeze(0),
                                                                       memory_bank=clustr_m_bank.unsqueeze(0))
                        # [clustr_size, h_size]
                        clustr_representation = clustr_representation.squeeze(1)
                    elif self.merge_type == 'attn_weighted':
                        # [clustr_size, 1, h_size]
                        query = clustr_m_bank.unsqueeze(1)
                        # [clustr_size, clustr_size, h_size]
                        attn_memory_bank = clustr_m_bank.unsqueeze(0).expand(clustr_size, -1, -1)
                        # [clustr_size, clustr_size, 1]
                        attn_clustr_scores = clustr_scores.unsqueeze(2)
                        # [1, clustr_size, h_size],  [1, clustr_size, clustr_size]
                        clustr_representation, attn_scores = self.attn(source=query,
                                                                       memory_bank=attn_memory_bank,
                                                                       score_feats=attn_clustr_scores)
                        # [clustr_size, h_size]
                        clustr_representation = clustr_representation.squeeze(0)
                    elif self.merge_type == 'attn_noself':
                        noself_clustr_m_bank = [torch.cat([clustr_m_bank[:self_idx], clustr_m_bank[self_idx + 1:]], dim=0)
                                                for self_idx in range(clustr_size)]
                        # [ clustr_size, clustr_size-1, h_size]
                        noself_clustr_m_bank = torch.stack(noself_clustr_m_bank, dim=0)
                        assert noself_clustr_m_bank.size(1) == clustr_size - 1
                        # [1, clustr_size, h_size], [1, clustr_size, clustr_size - 1]
                        clustr_representation, attn_scores = self.attn(source=clustr_m_bank.unsqueeze(1),
                                                                       memory_bank=noself_clustr_m_bank)
                        # [clustr_size, h_size]
                        clustr_representation = clustr_representation.squeeze(0)
                    elif self.merge_type == 'attn_noself_weighted':
                        # [clustr_size, 1, h_size]
                        query = clustr_m_bank.unsqueeze(1)

                        noself_clustr_m_bank = [
                            torch.cat([clustr_m_bank[:self_idx], clustr_m_bank[self_idx + 1:]], dim=0)
                            for self_idx in range(clustr_size)]
                        # [ clustr_size, clustr_size-1, h_size]
                        noself_clustr_m_bank = torch.stack(noself_clustr_m_bank, dim=0)

                        noself_clustr_scores = [torch.cat([clustr_scores[self_idx][:self_idx], clustr_scores[self_idx][self_idx+1:]], dim=0)
                                                for self_idx in range(clustr_size)]
                        # [ clustr_size, clustr_size-1, 1]
                        noself_clustr_scores = torch.stack(noself_clustr_scores, dim=0).unsqueeze(2)

                        assert noself_clustr_m_bank.size(1) == clustr_size - 1
                        assert noself_clustr_scores.size(1) == clustr_size - 1

                        # [1, clustr_size, h_size],  [1, clustr_size, clustr_size]
                        clustr_representation, attn_scores = self.attn(source=query,
                                                                       memory_bank=noself_clustr_m_bank,
                                                                       score_feats=noself_clustr_scores)
                        # [clustr_size, h_size]
                        clustr_representation = clustr_representation.squeeze(0)
                    else:
                        raise NotImplementedError
                    # check the size of the clustr_represenation
                    aeq(clustr_m_bank.size(), clustr_representation.size())
                    if self.merge_type != 'attn_no_extra_params':
                        # [clustr_size, 2 * h_size]
                        mrg_input = torch.cat([clustr_m_bank, clustr_representation], dim=1)
                        # [clustr_size, h_size]
                        mrg_out = self.mrg_acti_func(self.mrg_linear(mrg_input))
                    else:
                        # [clustr_size, h_size]
                        mrg_out = clustr_m_bank + clustr_representation

                    for coref_idx, coref_posi in enumerate(clustr_coref_posi):
                        assert coref_posi not in mrgd_coref_m_bank
                        mrgd_coref_m_bank[coref_posi] = mrg_out[coref_idx]

            if self.fnn_all:
                # [seq_len, 2 * h_size]
                fnn_all_in = torch.cat([ex_m_bank, torch.zeros_like(ex_m_bank)], dim=1)
                fnn_ex_m_bank = self.mrg_acti_func(self.mrg_linear(fnn_all_in))

            mrgd_ex_m_bank = []
            for src_posi in range(src_len):
                if src_posi not in mrgd_coref_m_bank:
                    if self.fnn_all:
                        append_vec = fnn_ex_m_bank[src_posi]
                    else:
                        append_vec = ex_m_bank[src_posi]
                    mrgd_ex_m_bank.append(append_vec)
                else:
                    mrgd_ex_m_bank.append(mrgd_coref_m_bank[src_posi])
            # [src_len, h_size]
            mrgd_ex_m_bank = torch.stack(mrgd_ex_m_bank, dim=0)
            mrgd_m_bank.append(mrgd_ex_m_bank)
        # [src_len, b_size, h_size]
        mrgd_m_bank = torch.stack(mrgd_m_bank, dim=1)
        # check whether the size is correct
        aeq(m_bank.size(), mrgd_m_bank.size())
        return mrgd_m_bank

    def preprocess(self, corefs):
        """
        convert from a list of string to a list of list of position Tensors and Score Tensors
        :param corefs: a list of coref string
        :return:
            b_coref_posi: a list of list of position 1-D list
            b_coref_score: a list of list of score 2-D list
        """
        b_coref_posi = []
        b_coref_score = []
        for coref in corefs:
            coref = json.loads(coref)
            ex_coref_posi = []
            ex_coref_score = []
            for clustr in coref:
                m_posi_list = clustr['m_posi_list']
                m_coref_scores_list = clustr['m_coref_scores_list']
                assert len(m_posi_list) == len(m_coref_scores_list)
                last_token_posi_list = [int(span[1]) - 1 for span in m_posi_list]
                m_coref_scores_list = [[float(sc) for sc in one_row] for one_row in m_coref_scores_list]

                ex_coref_posi.append(last_token_posi_list)
                ex_coref_score.append(m_coref_scores_list)

            b_coref_posi.append(ex_coref_posi)
            b_coref_score.append(ex_coref_score)

        return b_coref_posi, b_coref_score