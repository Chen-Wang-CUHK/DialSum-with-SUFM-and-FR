"""Global attention modules (Luong / Bahdanau)"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.modules.sparse_activations import sparsemax
from onmt.utils.misc import aeq, sequence_mask

# This class is mainly used by decoder.py for RNNs but also
# by the CNN / transformer decoder when copy attention is used
# CNN has its own attention mechanism ConvMultiStepAttention
# Transformer has its own MultiHeadedAttention


class GlobalAttentionPlus(nn.Module):
    r"""
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = \sum_{j=1}^{\text{SeqLength}} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`\text{score}(H_j,q) = H_j^T q`
       * general: :math:`\text{score}(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`\text{score}(H_j, q) = v_a^T \text{tanh}(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]
       attn_func (str): attention function to use, options [softmax,sparsemax]

    """

    def __init__(self, q_dim, ctx_dim, coverage=False, attn_type="dot",
                 attn_func="softmax", score_feat_dim=0):
        super(GlobalAttentionPlus, self).__init__()

        self.q_dim = q_dim
        self.ctx_dim = ctx_dim
        self.score_feat_dim = score_feat_dim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type (got {:s}).".format(
                attn_type))
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        if self.attn_type == "general":
            self.linear_in = nn.Linear(q_dim, ctx_dim + score_feat_dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(ctx_dim + score_feat_dim, ctx_dim, bias=False)
            self.linear_query = nn.Linear(q_dim, ctx_dim, bias=True)
            self.v = nn.Linear(ctx_dim, 1, bias=False)

        if coverage:
            self.linear_cover = nn.Linear(1, ctx_dim, bias=False)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (FloatTensor): sequence of queries ``(batch, tgt_len, q_dim)``
          h_s (FloatTensor): sequence of sources ``(batch, src_len, ctx_dim + score_feat_dim)``

        Returns:
          FloatTensor: raw attention scores (unnormalized) for each src index
            ``(batch, tgt_len, src_len)``
        """

        # Check input sizes
        src_batch, src_len, mrgd_dim = h_s.size()
        tgt_batch, tgt_len, q_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(self.q_dim, q_dim)
        aeq(self.ctx_dim + self.score_feat_dim, mrgd_dim)
        ctx_dim = mrgd_dim - self.score_feat_dim

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, q_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, ctx_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            wq = self.linear_query(h_t.view(-1, q_dim))
            wq = wq.view(tgt_batch, tgt_len, 1, ctx_dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, ctx_dim)

            uh = self.linear_context(h_s.contiguous().view(-1, mrgd_dim))
            uh = uh.view(src_batch, 1, src_len, ctx_dim)
            uh = uh.expand(src_batch, tgt_len, src_len, ctx_dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, ctx_dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, memory_lengths=None, coverage=None, score_feats=None):
        """

        Args:
          source (FloatTensor): query vectors ``(batch, tgt_len, q_dim)``
          memory_bank (FloatTensor): source vectors ``(batch, src_len, ctx_dim)``
          memory_lengths (LongTensor): the source context lengths ``(batch,)``
          coverage (FloatTensor): None (not supported yet)
          score_feats (FloatTensor): score feature vectors ``(batch, src_len, score_feat_dim)``

        Returns:
          (FloatTensor, FloatTensor):

          * Computed vector ``(tgt_len, batch, ctx_dim)``
          * Attention distribtutions for each query
            ``(tgt_len, batch, src_len)``
        """

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, ctx_dim = memory_bank.size()
        batch_, target_l, q_dim = source.size()
        aeq(batch, batch_)
        aeq(self.q_dim, q_dim)
        aeq(self.ctx_dim, ctx_dim)
        if coverage is not None:
            batch_, source_l_ = coverage.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank += self.linear_cover(cover).view_as(memory_bank)
            memory_bank = torch.tanh(memory_bank)

        if score_feats is not None:
            batch_, source_l_, score_feat_dim = score_feats.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)
            aeq(self.score_feat_dim, score_feat_dim)
            # [batch, src_len, ctx_dim + score_feat_dim]
            mrgd_memory_bank = torch.cat([memory_bank, score_feats], dim=2)
        else:
            mrgd_memory_bank = memory_bank

        # compute attention scores, as in Luong et al.
        align = self.score(source, mrgd_memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(~mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch*target_l, source_l), -1)
        else:
            align_vectors = sparsemax(align.view(batch*target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        # (batch, tgt_len, ctx_dim)
        c = torch.bmm(align_vectors, memory_bank)

        if one_step:
            c = c.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
            # Check output sizes
            batch_, ctx_dim = c.size()
            aeq(batch, batch_)
            aeq(self.ctx_dim, ctx_dim)
            batch_, source_l_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        else:
            c = c.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
            # Check output sizes
            target_l_, batch_, ctx_dim = c.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(self.ctx_dim, ctx_dim)
            target_l_, batch_, source_l_ = align_vectors.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        return c, align_vectors
