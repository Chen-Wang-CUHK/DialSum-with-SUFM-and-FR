"""Define RNN-based encoders."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase, aeq
from onmt.encoders.coref_merge_layer import CorefMergeLayer
from onmt.utils.rnn_factory import rnn_factory


class RNNEncoderCoref(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False, coref_layer_type='none', coref_fnn_all=False):
        super(RNNEncoderCoref, self).__init__()
        assert embeddings is not None
        assert num_layers == 1

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.first_rnn, self.first_no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # self.second_rnn, self.second_no_pack_padded_seq = \
        #     rnn_factory(rnn_type,
        #                 input_size=hidden_size * num_directions,
        #                 hidden_size=hidden_size,
        #                 num_layers=num_layers,
        #                 dropout=dropout,
        #                 bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

        # coreference merge layer
        self.coref_merge_layer = CorefMergeLayer(hidden_size * num_directions, coref_layer_type, coref_fnn_all)

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.bridge,
            opt.coref_layer_type,
            opt.coref_fnn_all)

    def forward(self, src, lengths=None, corefs=None, utr_position_tuple=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        # The first BRNN layer
        packed_emb = emb
        if lengths is not None and not self.first_no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)
        memory_bank, encoder_final = self.first_rnn(packed_emb)
        if lengths is not None and not self.first_no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        # for coreference merge layer
        memory_bank = self.coref_merge_layer(memory_bank, corefs)

        # # The second BRNN layer
        # packed_emb = memory_bank
        # if lengths is not None and not self.second_no_pack_padded_seq:
        #     # Lengths data is wrapped inside a Tensor.
        #     lengths_list = lengths.view(-1).tolist()
        #     packed_emb = pack(memory_bank, lengths_list)
        # memory_bank, encoder_final = self.second_rnn(packed_emb)
        # if lengths is not None and not self.second_no_pack_padded_seq:
        #     memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return encoder_final, memory_bank, lengths

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """Forward hidden state through bridge."""
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout


class SeqHRECoref(EncoderBase):
    """ A hierarchical recurrent neural network encoder with a coreference merge layer.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False, coref_layer_type='none', coref_fnn_all=False,
                 utr_mrg_type='ave_pool', enc_final_state_type='word', rm_middle_dp=False):
        super(SeqHRECoref, self).__init__()
        assert embeddings is not None
        assert num_layers == 1
        assert enc_final_state_type in ['word', 'utr']

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.hidden_size = hidden_size

        self.utr_mrg_type = utr_mrg_type
        self.enc_final_state_type = enc_final_state_type
        self.rm_middle_dp = rm_middle_dp
        self.embeddings = embeddings

        self.first_rnn, self.first_no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        self.second_rnn, self.second_no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=hidden_size * num_directions,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        if not rm_middle_dp:
            self.dropout = nn.Dropout(dropout)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

        # coreference merge layer
        self.coref_layer_type = coref_layer_type
        self.coref_merge_layer = CorefMergeLayer(hidden_size * num_directions, coref_layer_type, coref_fnn_all)

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.bridge,
            opt.coref_layer_type,
            opt.coref_fnn_all,
            opt.utr_mrg_type,
            opt.enc_final_state_type,
            opt.rm_middle_dp)

    def forward(self, src, lengths=None, corefs=None, utr_position_tuple=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        src_len, batch, f_num = src.size()
        batch_1 = lengths.size(0)

        assert isinstance(utr_position_tuple, tuple), "The utr_position for seqHREncoder should be a tuple."
        # utr_p: [batch_size, s_num, 2], utr_nums: [batch_size]
        utr_p, utr_nums = utr_position_tuple

        # we do args check here
        batch_2, s_num, _ = utr_p.size()
        batch_3 = utr_nums.size(0)
        aeq(batch, batch_1, batch_2, batch_3)

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        # The first BRNN layer
        packed_emb = emb
        if lengths is not None and not self.first_no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)
        word_memory_bank, word_encoder_final = self.first_rnn(packed_emb)
        if lengths is not None and not self.first_no_pack_padded_seq:
            word_memory_bank = unpack(word_memory_bank)[0]

        # for coreference merge layer
        word_memory_bank = self.coref_merge_layer(word_memory_bank, corefs)

        utr_memory_bank = None
        assert utr_nums is not None
        assert not self.second_no_pack_padded_seq
        if self.utr_mrg_type == 'fw_bw_cat':
            assert self.coref_layer_type == 'none', "Currently, fw_bw_cat style utr_merge_type is not reasonable for" \
                                                    "the not-none type coref_merge_layer"
            # [s_num, batch_size, 2]
            utr_p = utr_p.transpose(0, 1)
            # [s_num, batch_size, h_size]
            f_index = utr_p[:, :, 0].unsqueeze(-1).expand(-1, -1, self.hidden_size)
            b_index = utr_p[:, :, 1].unsqueeze(-1).expand(-1, -1, self.hidden_size)
            # [s_num, batch_size, h_size]
            gather_index = torch.cat([f_index, b_index], dim=-1)
            utr_vector = word_memory_bank.gather(dim=0, index=gather_index)
        else:
            # TODO: more efficient code is needed
            utr_vector = []
            # [2 * h_size]
            zero_pad = torch.zeros_like(word_memory_bank[0][0])
            for b in range(batch):
                # [src_len, 2 * h_size]
                ex_word_memory_bank = word_memory_bank[:, b, :]
                # [s_num, 2]
                ex_utr_p = utr_p[b]
                ex_utr_vector = []
                utr_num_check = 0
                for utr_idx in range(s_num):
                    ex_utr_p_i = ex_utr_p[utr_idx]
                    start_p = ex_utr_p_i[1]
                    end_p = ex_utr_p_i[0]
                    assert end_p >= start_p
                    if utr_idx < utr_nums[b]:
                        utr_num_check += 1
                        if self.utr_mrg_type == 'ave_pool':
                            # [2 * h_size]
                            utr_represent = torch.mean(ex_word_memory_bank[start_p: end_p + 1], dim=0)
                        elif self.utr_mrg_type == 'max_pool':
                            # [2 * h_size]
                            utr_represent, _ = torch.max(ex_word_memory_bank[start_p: end_p + 1], dim=0)
                    else:
                        utr_represent = zero_pad
                    ex_utr_vector.append(utr_represent)
                assert utr_num_check == utr_nums[b]
                # [s_num, 2 * h_size]
                ex_utr_vector = torch.stack(ex_utr_vector, dim=0)
                utr_vector.append(ex_utr_vector)
            # [s_num, b_size, 2 * h_size]
            utr_vector = torch.stack(utr_vector, dim=1)

        if not self.rm_middle_dp:
            # dropout the utterance vector
            utr_vector = self.dropout(utr_vector)
        # 2. use utr_rnn to encode the utterance representations
        sorted_utr_nums, idx_sort = torch.sort(utr_nums, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        # use the sorted order
        utr_vector = utr_vector.index_select(1, idx_sort)
        sorted_utr_nums_list = sorted_utr_nums.view(-1).tolist()
        packed_emb = pack(utr_vector, sorted_utr_nums_list)
        utr_memory_bank, utr_encoder_final = self.second_rnn(packed_emb)
        utr_memory_bank = unpack(utr_memory_bank)[0]
        # recover the original order
        utr_memory_bank = utr_memory_bank.index_select(1, idx_unsort)
        utr_encoder_final = utr_encoder_final.index_select(1, idx_unsort)

        if self.enc_final_state_type == 'word':
            encoder_final = word_encoder_final
        else:
            encoder_final = utr_encoder_final

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return encoder_final, (word_memory_bank, utr_memory_bank), (lengths, utr_nums)

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """Forward hidden state through bridge."""
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout