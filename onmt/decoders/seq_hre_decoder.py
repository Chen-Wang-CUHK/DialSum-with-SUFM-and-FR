import torch
import torch.nn as nn

from onmt.models.stacked_rnn import StackedLSTM, StackedGRU
from onmt.modules import context_gate_factory, GlobalAttention, GlobalAttentionPlus, SeqHREWordGlobalAttention
from onmt.utils.rnn_factory import rnn_factory

from onmt.utils.misc import aeq

class SeqHREInputFeedRNNDecoder(nn.Module):
    """Input feeding based decoder.

    See :class:`~onmt.decoders.decoder.RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[memory_bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :class:`~onmt.modules.GlobalAttention`
       attn_func (str) : see :class:`~onmt.modules.GlobalAttention`
       coverage_attn (str): see :class:`~onmt.modules.GlobalAttention`
       context_gate (str): see :class:`~onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
       reuse_copy_attn (bool): reuse the attention for copying
       copy_attn_type (str): The copy attention style. See
        :class:`~onmt.modules.GlobalAttention`.
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general", attn_func="softmax",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False, copy_attn_type="general",
                 hr_attn_type='utr_word_both', seqHRE_attn_rescale=False,
                 tgt_transE_state_type='out_h_after_dp'):
        super(SeqHREInputFeedRNNDecoder, self).__init__()

        self.attentional = attn_type != "none" and attn_type is not None
        self.attn_type = attn_type
        self.hr_attn_type = hr_attn_type
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        self.tgt_transE_state_type = tgt_transE_state_type
        # Decoder state
        self.state = {}

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        if not self.attentional:
            if self._coverage:
                raise ValueError("Cannot use coverage term with no attention.")
            self.attn = None
        else:
            self.utr_attn = GlobalAttentionPlus(
                hidden_size, hidden_size, coverage=coverage_attn,
                attn_type=attn_type, attn_func=attn_func)
            self.word_attn = None
            if hr_attn_type in ['word_only', 'utr_word_both']:
                self.word_attn = SeqHREWordGlobalAttention(
                    hidden_size, coverage=coverage_attn,
                    attn_type=attn_type, attn_func=attn_func,
                    seqHRE_attn_rescale=seqHRE_attn_rescale)
            # add one more mlp layer to merge two outputs from utr-level and word-level attention
            # mlp attention wants it with bias
            out_bias = attn_type == "mlp"
            if self.hr_attn_type == 'utr_word_both':
                self.attn_merge_linear_out = nn.Linear(hidden_size * 3, hidden_size, bias=out_bias)
            else:
                # 'utr_only', 'word_only'
                self.attn_merge_linear_out = nn.Linear(hidden_size * 2, hidden_size, bias=out_bias)

        if copy_attn and not reuse_copy_attn:
            if copy_attn_type == "none" or copy_attn_type is None:
                raise ValueError(
                    "Cannot use copy_attn with copy_attn_type none")
            self.copy_attn = GlobalAttention(
                hidden_size, attn_type=copy_attn_type, attn_func=attn_func
            )
        else:
            self.copy_attn = None

        self._reuse_copy_attn = reuse_copy_attn and copy_attn
        if self._reuse_copy_attn and not self.attentional:
            raise ValueError("Cannot reuse copy attention with no attention.")

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.global_attention,
            opt.global_attention_function,
            opt.coverage_attn,
            opt.context_gate,
            opt.copy_attn,
            opt.dropout[0] if type(opt.dropout) is list
            else opt.dropout,
            embeddings,
            opt.reuse_copy_attn,
            opt.copy_attn_type,
            opt.hr_attn_type,
            opt.seqHRE_attn_rescale,
            opt.tgt_transE_state_type)

    def init_state(self, src, memory_bank, encoder_final):
        """Initialize decoder state with last state of the encoder."""
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = tuple(_fix_enc_hidden(enc_hid)
                                         for enc_hid in encoder_final)
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(encoder_final), )

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = \
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        self.state["coverage"] = None

    def map_state(self, fn):
        self.state["hidden"] = tuple(fn(h, 1) for h in self.state["hidden"])
        self.state["input_feed"] = fn(self.state["input_feed"], 1)
        if self._coverage and self.state["coverage"] is not None:
            self.state["coverage"] = fn(self.state["coverage"], 1)

    def detach_state(self):
        self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
        self.state["input_feed"] = self.state["input_feed"].detach()

    def forward(self, tgt, memory_bank_tuple, memory_lengths=None, step=None,
                utr_position_tuple=None, src_word_utr_ids=None):
        """
        Args:
            tgt (LongTensor): sequences of padded tokens
                 ``(tgt_len, batch, nfeats)``.
            memory_bank_tuple (`tuple`): (word_memory_bank, utr_memory_bank) from the encoder
                 `([src_len x batch x hidden], [utr_num x batch x hidden])`.
            memory_lengths (LongTensor):  (word_lengths, utr_nums), the source lengths and utr numbers
                ``([batch,], [batch,])``.
            utr_position_tuple (`tuple`): the sentence position infomation. (utr_position_info, utr_nums)
                `([batch, utr_num, 2], [batch])`
            src_word_utr_ids (:obj: `tuple'): (word_utr_ids, src_lengths) with size `([batch, src_lengths], [batch])'
        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * dec_outs: output from the decoder (after attn)
              ``(tgt_len, batch, hidden)``.
            * attns: distribution over src at each tgt
              ``(tgt_len, batch, src_len)``.
        """

        dec_state, dec_outs, attns = self._run_forward_pass(
            tgt, memory_bank_tuple, lengths_tuple=memory_lengths,
            utr_position_tuple=utr_position_tuple,
            src_word_utr_ids=src_word_utr_ids)

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
        self.state["coverage"] = None
        if "coverage" in attns:
            self.state["coverage"] = attns["coverage"][-1].unsqueeze(0)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
        return dec_outs, attns

    def _run_forward_pass(self, tgt, memory_bank_tuple, lengths_tuple=None,
                          utr_position_tuple=None,
                          src_word_utr_ids=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        assert isinstance(memory_bank_tuple, tuple)
        word_memory_bank, utr_memory_bank = memory_bank_tuple
        word_lengths, utr_lengths = lengths_tuple
        utr_position, utr_lengths_ = utr_position_tuple
        assert (utr_lengths == utr_lengths_).all(), "The utr_lengths should be consistent."

        dec_outs = []
        attns = {"utr_std": [], "transE_re_states": []}
        if self.word_attn is not None:
            attns["std"] = []
            attns["word_std"] = []
        if self.copy_attn is not None or self._reuse_copy_attn:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        dec_state = self.state["hidden"]
        coverage = self.state["coverage"].squeeze(0) \
            if self.state["coverage"] is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for emb_t in emb.split(1):
            decoder_input = torch.cat([emb_t.squeeze(0), input_feed], 1)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)
            if self.attentional:
                # utterance level attention
                # [batch, h_size], [batch, utr_num]
                utr_attn_c, utr_p_attn = self.utr_attn(
                    rnn_output,
                    utr_memory_bank.transpose(0, 1),
                    memory_lengths=utr_lengths)

                if self.hr_attn_type in ['word_only', 'utr_word_both']:
                    # word level attention
                    # [batch, h_size], [batch, src_len]
                    word_attn_c, word_attn_h, word_p_attn = self.word_attn(
                        rnn_output,
                        word_memory_bank.transpose(0, 1),
                        memory_lengths=word_lengths,
                        utr_align_vectors=utr_p_attn,
                        utr_position_tuple=utr_position_tuple,
                        src_word_utr_ids=src_word_utr_ids)

                if self.hr_attn_type == 'utr_word_both':
                    # [batch, 3 * hidden_size]
                    decoder_output = torch.cat([rnn_output, utr_attn_c, word_attn_c], dim=1)
                elif self.hr_attn_type == 'utr_only':
                    # [batch, 2 * hidden_size]
                    decoder_output = torch.cat([rnn_output, utr_attn_c], dim=1)
                else:
                    # 'word_only'
                    # [batch, 2 * hidden_size]
                    decoder_output = torch.cat([rnn_output, word_attn_c], dim=1)

                # [batch, hidden_size]
                decoder_output = self.attn_merge_linear_out(decoder_output)
                if self.attn_type in ["general", "dot"]:
                    decoder_output = torch.tanh(decoder_output)

                attns["utr_std"].append(utr_p_attn)
                attns["word_std"].append(word_p_attn)
                attns["std"].append(word_p_attn)
            else:
                decoder_output = rnn_output
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )

            decoder_output_bf_dp = decoder_output
            decoder_output = self.dropout(decoder_output_bf_dp)
            input_feed = decoder_output

            if self.tgt_transE_state_type == "out_h_before_dp":
                attns["transE_re_states"] += [decoder_output_bf_dp]
            elif self.tgt_transE_state_type == "out_h_after_dp":
                attns["transE_re_states"] += [decoder_output]
            elif self.tgt_transE_state_type == "state_h":
                attns["transE_re_states"] += [rnn_output]

            dec_outs += [decoder_output]

            # TODO: implement the multi-level coverage mechanism
            # # Update the coverage attention.
            # if self._coverage:
            #     coverage = p_attn if coverage is None else p_attn + coverage
            #     attns["coverage"] += [coverage]

            if self.copy_attn is not None:
                _, copy_attn = self.copy_attn(
                    decoder_output, word_memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._reuse_copy_attn:
                attns["copy"] = attns["std"]

        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert rnn_type != "SRU", "SRU doesn't support input feed! " \
            "Please set -input_feed 0!"
        stacked_cell = StackedLSTM if rnn_type == "LSTM" else StackedGRU
        return stacked_cell(num_layers, input_size, hidden_size, dropout)

    @property
    def _input_size(self):
        """Using input feed by concatenating input with attention vectors."""
        return self.embeddings.embedding_size + self.hidden_size

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.rnn.dropout.p = dropout
        self.embeddings.update_dropout(dropout)