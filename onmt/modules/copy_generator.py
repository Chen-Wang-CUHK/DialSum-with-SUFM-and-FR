import torch
import torch.nn as nn
import json

from onmt.utils.misc import aeq
from onmt.utils.loss import NMTLossCompute


def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs=None,
                         batch_dim=1, batch_offset=None):
    """
    Given scores from an expanded dictionary
    corresponeding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.
    """
    offset = len(tgt_vocab)
    for b in range(scores.size(batch_dim)):
        blank = []
        fill = []

        if src_vocabs is None:
            src_vocab = batch.src_ex_vocab[b]
        else:
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = batch.indices.data[batch_id]
            src_vocab = src_vocabs[index]

        for i in range(1, len(src_vocab)):
            sw = src_vocab.itos[i]
            ti = tgt_vocab.stoi[sw]
            if ti != 0:
                blank.append(offset + i)
                fill.append(ti)
        if blank:
            blank = torch.Tensor(blank).type_as(batch.indices.data)
            fill = torch.Tensor(fill).type_as(batch.indices.data)
            score = scores[:, b] if batch_dim == 1 else scores[b]
            score.index_add_(1, fill, score.index_select(1, blank))
            score.index_fill_(1, blank, 1e-10)
    return scores


class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks
    :cite:`DBLP:journals/corr/SeeLM17`.

    These networks consider copying words
    directly from the source sequence.

    The copy generator is an extended version of the standard
    generator that computes three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`


    .. mermaid::

       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O


    Args:
       input_size (int): size of input representation
       output_size (int): size of output vocabulary
       pad_idx (int)
    """

    def __init__(self, input_size, output_size, pad_idx):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear_copy = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx

    def forward(self, hidden, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.

        Args:
           hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
           attn (FloatTensor): attn for each ``(batch x tlen, input_size)``
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab containing.
               ``(src_len, batch, extra_words)``
        """

        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.pad_idx] = -float('inf')
        prob = torch.softmax(logits, 1)

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(hidden))
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy)
        mul_attn = torch.mul(attn, p_copy)
        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, slen).transpose(0, 1),
            src_map.transpose(0, 1)
        ).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1)


class CopyGeneratorLoss(nn.Module):
    """Copy generator criterion."""
    def __init__(self, vocab_size, force_copy, unk_index=0,
                 ignore_index=-100, eps=1e-20):
        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index

    def forward(self, scores, align, target):
        """
        Args:
            scores (FloatTensor): ``(batch_size*tgt_len)`` x dynamic vocab size
                whose sum along dim 1 is less than or equal to 1, i.e. cols
                softmaxed.
            align (LongTensor): ``(batch_size x tgt_len)``
            target (LongTensor): ``(batch_size x tgt_len)``
        """
        # probabilities assigned by the model to the gold targets
        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)

        # probability of tokens copied from source
        copy_ix = align.unsqueeze(1) + self.vocab_size
        copy_tok_probs = scores.gather(1, copy_ix).squeeze(1)
        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == self.unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = align == self.unk_index
        if not self.force_copy:
            non_copy = non_copy | (target != self.unk_index)

        probs = torch.where(
            non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        )

        loss = -probs.log()  # just NLLLoss; can the module be incorporated?
        # Drop padding.
        loss[target == self.ignore_index] = 0
        return loss


class CopyGeneratorLossCompute(NMTLossCompute):
    """Copy Generator Loss Computation."""
    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length,
                 lambda_coverage=0.0, lambda_support_utr=0.0, lambda_previous_utr=0.0, lambda_tgt_fact_re=0.0):
        super(CopyGeneratorLossCompute, self).__init__(
            criterion, generator, lambda_coverage=lambda_coverage)
        self.tgt_vocab = tgt_vocab
        self.normalize_by_length = normalize_by_length
        self.lambda_support_utr = lambda_support_utr
        self.lambda_previous_utr = lambda_previous_utr
        self.lambda_tgt_fact_re = lambda_tgt_fact_re


    def _make_shard_state(self, batch, output, range_, attns):
        """See base class for args description."""
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        shard_state = super(CopyGeneratorLossCompute, self)._make_shard_state(
            batch, output, range_, attns)

        shard_state.update({
            "copy_attn": attns.get("copy"),
            "align": batch.alignment[range_[0] + 1: range_[1]],
            "transE_re_states": attns.get("transE_re_states")
        })
        return shard_state

    def _compute_support_utr_loss(self, tgt_support_utrs, word_std_attns):
        # [tgt_len, b_size, src_len]
        tgt_len, b_size, src_len = word_std_attns.size()
        aeq(len(tgt_support_utrs), b_size)
        batch_loss = []
        for b in range(b_size):
            support_utrs = tgt_support_utrs[b]
            support_utrs = json.loads(support_utrs)
            support_utr_loss = []
            for tgt_sent_idx, support_utr_info in enumerate(support_utrs):
                tgt_sent_posi = support_utr_info['tgt_sent_posi']
                utr_idxs = support_utr_info["utr_idxs"]
                utr_jaccard_scores = support_utr_info["utr_jaccard_scores"]
                utr_posis = support_utr_info["utr_posis"]

                # calculate the averaged src word attentions
                sent_start = tgt_sent_posi[1]
                sent_end = tgt_sent_posi[0] + 1
                # [sent_len, src_len]
                sent_word_attns = word_std_attns[sent_start: sent_end, b, :]
                # [src_len]
                ave_sent_word_attns = torch.mean(sent_word_attns, dim=0, keepdim=False)
                support_utr_attn_sum = None
                for utr_posi in utr_posis:
                    utr_start = utr_posi[1]
                    utr_end = utr_posi[0] + 1
                    attn_sum = torch.sum(ave_sent_word_attns[utr_start: utr_end])
                    support_utr_attn_sum = support_utr_attn_sum + attn_sum if support_utr_attn_sum is not None else attn_sum
                # calculate the support_utr_loss for current tgt sent in b-th example
                if support_utr_attn_sum is None:
                    support_utr_attn_sum = torch.sum(ave_sent_word_attns)

                sent_support_utr_loss = support_utr_attn_sum / torch.sum(ave_sent_word_attns)
                sent_support_utr_loss = -sent_support_utr_loss.log()
                support_utr_loss.append(sent_support_utr_loss)
            loss = sum(support_utr_loss)
            batch_loss.append(loss)
        return torch.stack(batch_loss, dim=0)

    def _compute_previous_utr_loss(self, tgt_support_utrs, word_std_attns):
        # [tgt_len, b_size, src_len]
        tgt_len, b_size, src_len = word_std_attns.size()
        aeq(len(tgt_support_utrs), b_size)
        batch_loss = []
        for b in range(b_size):
            support_utrs = tgt_support_utrs[b]
            support_utrs = json.loads(support_utrs)
            previous_utr_loss = []
            previous_utrs = {}
            for tgt_sent_idx, support_utr_info in enumerate(support_utrs):
                tgt_sent_posi = support_utr_info['tgt_sent_posi']
                utr_idxs = support_utr_info["utr_idxs"]
                utr_jaccard_scores = support_utr_info["utr_jaccard_scores"]
                utr_posis = support_utr_info["utr_posis"]
                aeq(len(utr_idxs), len(utr_jaccard_scores), len(utr_posis))

                # the first tgt sentence has no previous_utr_loss
                if tgt_sent_idx == 0:
                    sent_previous_utr_loss = torch.zeros(1).to(word_std_attns.device).sum()
                else:
                    # calculate the averaged src word attentions
                    sent_start = tgt_sent_posi[1]
                    sent_end = tgt_sent_posi[0] + 1
                    # [sent_len, src_len]
                    sent_word_attns = word_std_attns[sent_start: sent_end, b, :]
                    # [src_len]
                    ave_sent_word_attns = torch.mean(sent_word_attns, dim=0, keepdim=False)
                    previous_utr_attn_sum = None
                    for utr_idx in previous_utrs:
                        utr_posi = previous_utrs[utr_idx]
                        utr_start = utr_posi[1]
                        utr_end = utr_posi[0] + 1
                        attn_sum = torch.sum(ave_sent_word_attns[utr_start: utr_end])
                        previous_utr_attn_sum = previous_utr_attn_sum + attn_sum if previous_utr_attn_sum is not None else attn_sum
                    # calculate the support_utr_loss for current tgt sent in b-th example
                    if previous_utr_attn_sum is None:
                        previous_utr_attn_sum = torch.zeros(1).to(word_std_attns.device).sum()

                    sent_previous_utr_loss = previous_utr_attn_sum / torch.sum(ave_sent_word_attns)
                    sent_previous_utr_loss = -(1 - sent_previous_utr_loss).log()
                previous_utr_loss.append(sent_previous_utr_loss)

                # we update the previous supporting utterances when finish computing the previous_utr_loss
                # fix the bug when preparing the previous supporting utterances
                for utr_idx, utr_posi in zip(utr_idxs, utr_posis):
                    previous_utrs[utr_idx] = utr_posi

            loss = sum(previous_utr_loss)
            batch_loss.append(loss)
        return torch.stack(batch_loss, dim=0)


    def _compute_tgt_fact_re_loss(self, tgt_fact_triplets, hiddens):
        tgt_len, b_size, h_size = hiddens.size()
        aeq(b_size, len(tgt_fact_triplets))
        batch_loss = []
        for b in range(b_size):
            ex_tgt_fact_triplets = json.loads(tgt_fact_triplets[b])
            ex_loss = torch.zeros(1).to(hiddens.device).sum()
            if len(ex_tgt_fact_triplets) != 0:
                ex_hiddens = hiddens[:, b, :]
                for triplet in ex_tgt_fact_triplets:
                    h_hidden = ex_hiddens[triplet['h']]
                    r_hidden = ex_hiddens[triplet['r']]
                    t_hidden = ex_hiddens[triplet['t']]
                    ex_loss = ex_loss + (1 - torch.nn.functional.cosine_similarity(h_hidden + r_hidden, t_hidden, dim=0))
            batch_loss.append(ex_loss)
        return torch.stack(batch_loss, dim=0)


    def _compute_loss(self, batch, output, target, copy_attn, align, transE_re_states,
                      std_attn=None, coverage_attn=None):
        """Compute the loss.

        The args must match :func:`self._make_shard_state()`.

        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
            transE_re_states: the states from the decoder to compute the fact transE loss
        """
        target = target.view(-1)
        align = align.view(-1)
        scores = self.generator(
            self._bottle(output), self._bottle(copy_attn), batch.src_map
        )
        # [tgt_len * b_size]
        gen_loss = self.criterion(scores, align, target)

        if self.lambda_coverage != 0.0:
            # already multiply self.lambda_coverage in the function
            coverage_loss = self._compute_coverage_loss(std_attn,
                                                        coverage_attn)
            gen_loss += coverage_loss

        # [b_size]
        support_utr_loss = torch.zeros(batch.batch_size).to(gen_loss.device)
        if self.lambda_support_utr != 0.0:
            tgt_support_utrs = batch.tgt_support_utrs
            word_std_attns = copy_attn
            support_utr_loss = self._compute_support_utr_loss(tgt_support_utrs, word_std_attns)

        # [b_size]
        previous_utr_loss = torch.zeros(batch.batch_size).to(gen_loss.device)
        if self.lambda_previous_utr != 0.0:
            tgt_support_utrs = batch.tgt_support_utrs
            word_std_attns = copy_attn
            previous_utr_loss = self._compute_previous_utr_loss(tgt_support_utrs, word_std_attns)

        # [b_size]
        tgt_fact_re_loss = torch.zeros(batch.batch_size).to(gen_loss.device)
        if self.lambda_tgt_fact_re != 0.0:
            tgt_fact_triplets = batch.tgt_fact_triplets
            hiddens = transE_re_states
            tgt_fact_re_loss = self._compute_tgt_fact_re_loss(tgt_fact_triplets, hiddens)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        scores_data = collapse_copy_scores(
            self._unbottle(scores.clone(), batch.batch_size),
            batch, self.tgt_vocab, None)
        scores_data = self._bottle(scores_data)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.clone()
        unk = self.criterion.unk_index
        correct_mask = (target_data == unk) & (align != unk)
        offset_align = align[correct_mask] + len(self.tgt_vocab)
        target_data[correct_mask] += offset_align

        # Compute sum of perplexities for stats
        stats = self._stats(gen_loss.sum().clone(), scores_data, target_data,
                            support_utr_loss=support_utr_loss.sum().clone(),
                            previous_utr_loss=previous_utr_loss.sum().clone(),
                            tgt_fact_re_loss=tgt_fact_re_loss.sum().clone(),
                            b_size=batch.batch_size)

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            tgt_lens = batch.tgt[:, :, 0].ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            gen_loss = gen_loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            gen_loss = torch.div(gen_loss, tgt_lens).sum()
        else:
            gen_loss = gen_loss.sum()

        support_utr_loss = support_utr_loss.sum()
        previous_utr_loss = previous_utr_loss.sum()
        tgt_fact_re_loss = tgt_fact_re_loss.sum()

        loss = gen_loss + self.lambda_support_utr * support_utr_loss +\
               self.lambda_previous_utr * previous_utr_loss + self.lambda_tgt_fact_re * tgt_fact_re_loss

        return loss, stats
