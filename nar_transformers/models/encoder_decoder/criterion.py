from typing import Union, List
import logging
import torch
import torch.nn.functional as F
import numpy as np

from scipy.optimize import linear_sum_assignment as lsa

logger = logging.getLogger(__name__)


def xe_loss(
        logits: torch.FloatTensor,
        targets: torch.LongTensor,
        length_logits: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

    pad_idx = torch.tensor(0).long()

    lprobs = logits.log_softmax(-1)
    lprobs = lprobs.view(-1, lprobs.size(-1))

    targets = targets.view(-1, 1)
    non_pad_mask = targets.ne(pad_idx)

    nll_loss = -lprobs.gather(dim=-1, index=targets)[non_pad_mask]
    nll_loss = nll_loss.mean()

    if length_logits is not None:
        length_lprobs = length_logits.log_softmax(-1)
        length_target = non_pad_mask.long().sum(1)
        length_loss = -length_lprobs.gather(dim=-1, index=length_target)
        nll_loss += length_loss

    return nll_loss

def oaxe_loss(
        logits: torch.FloatTensor,
        logit_lengths: torch.Tensor,
        targets: torch.LongTensor,
        target_lengths: torch.Tensor,
    ) -> torch.FloatTensor:

    pad_idx = torch.tensor(0).long()
    #margin = 0.15

    #lprobs = logits.log_softmax(-1)
    #lprobs = lprobs.view(-1, lprobs.size(-1))

    bs, seq_len = targets.size()
    targets = targets.repeat(1, seq_len).view(bs, seq_len, seq_len)
    bipart_no_pad = targets.ne(pad_idx)
    bipart_lprobs = logits.log_softmax(-1)

    nll_loss = -bipart_lprobs.gather(dim=-1, index=targets) # bs seq seq
    nll_loss = nll_loss * bipart_no_pad

    best_match = np.repeat(np.arange(seq_len).reshape(1, -1, 1), bs, axis=0) # np.zeros((bs, seq_len, 1))
    nll_loss_numpy = nll_loss.detach().cpu().numpy()

    for batch_id in range(bs):
        no_pad_num = bipart_no_pad[batch_id, 0].sum()
        raw_index, col_index = lsa(nll_loss_numpy[batch_id, :no_pad_num, :no_pad_num])
        best_match[batch_id, :no_pad_num] = col_index.reshape(-1, 1)

    best_match = torch.Tensor(best_match).to(targets).long()
    nll_loss = nll_loss.gather(dim=-1, index=best_match)
    nll_loss = nll_loss.sum()

    return nll_loss

def axe_loss(logits: torch.FloatTensor,
             logit_lengths: torch.Tensor,
             targets: torch.LongTensor,
             target_lengths: torch.Tensor,
             blank_index: torch.LongTensor,
             delta: torch.FloatTensor,
             reduction: str = 'mean',
             label_smoothing: float = None,
             return_a: bool = False
            ) -> Union[torch.FloatTensor, List[torch.Tensor]]:
    """Aligned Cross Entropy
    Marjan Ghazvininejad, Vladimir Karpukhin, Luke Zettlemoyer, Omer Levy, in arXiv 2020
    https://arxiv.org/abs/2004.01655

    Computes the aligned cross entropy loss with parallel scheme.
    Parameters
    ----------
    logits : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    logit_lengths : ``torch.Tensor``, required.
        A ``torch.Tensor`` of size (batch_size,)
        which contains lengths of the logits
    targets : ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    target_lengths : ``torch.Tensor``, required.
        A ``torch.Tensor`` of size (batch_size,)
        which contains lengths of the targets
    blank_index : ``torch.LongTensor``, required.
        A ``torch.LongTensor``, An index of special blank token.
    delta : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` for penalizing skip target operators.
    reduction : ``str``, optional.
        Specifies the reduction to apply to the output.
        Default "mean".
    label_smoothing : ``float``, optional
        Whether or not to apply label smoothing.
    return_a : ``bool``, optional.
        Whether to return the matrix of conditional axe values. Default is False.
    """

    assert targets.size(0) == logits.size(0), f'Inconsistency of batch size,  {targets.size(0)} of targets and {logits.size(0)} of logits.'

    batch_size, logits_sequence_length, num_class = logits.shape
    _, target_sequence_length = targets.shape

    device = logits.device

    """
    # for torch.gather
    targets = targets.unsqueeze(-1) # batch_size, target_sequence_length, 1

    # (batch_size, target_sequence_length + 1, logits_sequence_length + 1)
    batch_A = torch.zeros(targets.size(0), targets.size(1) + 1, logits.size(1) + 1).to(device)
    batch_blank_index = torch.full((logits.size(0), 1), blank_index, dtype = torch.long).to(device)
    """
    # concat_targets: [null, ref1, ref2, ref3]
    concat_targets = targets.new(targets.size(0), 1).fill_(blank_index)
    concat_targets = torch.cat([concat_targets, targets], dim=1)
    repeat_targets = concat_targets.repeat(1, target_sequence_length).view(batch_size, target_sequence_length, target_sequence_length + 1)

    # prev lprobs: bs x seq-len x vocab size
    # now  lprobs: bs x seq-len x target seq len
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs = log_probs.gather(dim=-1, index=repeat_targets)#bs seq seq

    # new targets : 1, 2, 3, 4, 5, 6
    targets = torch.tensor(np.repeat(np.arange(target_sequence_length).reshape(1, -1, 1) + 1, batch_size, axis=0)).to(device)

    # (batch_size, target_sequence_length + 1, lprobs_sequence_length + 1)
    batch_A = torch.zeros(targets.size(0), targets.size(1) + 1, log_probs.size(1) + 1).to(device)
    batch_blank_index = torch.full((log_probs.size(0), 1), 0, dtype = torch.long).to(device)

    # A_{i,0} = A_{iâ1,0} â delta * log P_1 (Y_i)
    for i in range(1, targets.size(1) + 1):
        # batch_A[:, i, 0] is calculated from targets[:, i-1, :], because batch_A added 0-th row
        batch_A[:, i, 0] = batch_A[:, i-1, 0] - delta * torch.gather(log_probs[:, 0, :], dim=1, index=targets[:, i-1, :]).squeeze(-1)

    # A_{0,j} = A_{0,jâ1} â log P_j ("BLANK")
    for j in range(1, logits.size(1) + 1):
        # batch_A[:, 0, j] is calculated from log_probs[:, j-1, :], because batch_A added 0-th column
        batch_A[:, 0, j] = batch_A[:, 0, j-1] - delta * torch.gather(log_probs[:, j-1, :], dim=1, index=batch_blank_index).squeeze(-1)


    # flip logit dim to get anti-diagonal part by using use torch.diag
    batch_A_flip = batch_A.flip(-1) # (batch_size, target_sequence_length + 1, logits_sequence_length + 1)
    log_probs_flip = log_probs.flip(-2) # (batch_size, sequence_length, num_classes)

    # to extract indices for the regions corresponding diag part.
    map_logits = torch.arange(logits.size(1)) - torch.zeros(targets.size(1), 1)
    map_targets = torch.arange(targets.size(1)).unsqueeze(-1) - torch.zeros((1, logits.size(1)))
    # index must be int
    map_logits = map_logits.long().to(device)
    map_targets = map_targets.long().to(device)

    for i in range(logits.size(1) - 1, -targets.size(1), -1):


        # batch_A_flip_sets[:, :, :, 0] : batch_A[:, i  , j-1]
        # batch_A_flip_sets[:, :, :, 1] : batch_A[:, i-1, j  ]
        # batch_A_flip_sets[:, :, :, 2] : batch_A[:, i-1, j-1]
        batch_A_flip_sets = torch.cat((batch_A_flip.roll(shifts=-1, dims=-1).unsqueeze(-1),
                                       batch_A_flip.roll(shifts= 1, dims=-2).unsqueeze(-1),
                                       batch_A_flip.roll(shifts=(1, -1), dims=(-2, -1)).unsqueeze(-1)),
                                       dim = -1)

        # trimming
        # - the last column (A_{0,j} = A_{0,jâ1} â log P_j ("BLANK"))
        # - the first row (A_{i,0} = A_{iâ1,0} â delta * log P_1 (Y_i))
        batch_A_flip_sets_trim = batch_A_flip_sets[:, 1:, :-1, :]

        # extracting anti-diagonal part
        # (batch, 3, num_diag)
        A_diag = batch_A_flip_sets_trim.diagonal(offset=i, dim1 = -3, dim2 = -2)

        # (batch, num_diag, 3)
        A_diag = A_diag.transpose(-1, -2)
        num_diag = A_diag.size(1)

        logit_indices = map_logits.diagonal(offset=i, dim1 = -2, dim2 = -1)
        # log_probs_diag : (batch, num_diag, num_class)
        log_probs_flip_diag = log_probs_flip[:, logit_indices[0]:logit_indices[-1]+1, :]

        target_indices = map_targets.diagonal(offset=i, dim1 = -2, dim2 = -1)
        # targets_diag : (batch, num_diag, num_class)
        targets_diag = targets[:, target_indices[0]:target_indices[-1]+1, :]

        # align, skip_prediction, skip_target
        batch_align = A_diag[:, :, 2] - torch.gather(log_probs_flip_diag, dim=2, index=targets_diag).squeeze(-1)
        batch_skip_prediction = A_diag[:, :, 0] - torch.gather(log_probs_flip_diag, dim=2, index=batch_blank_index.expand(-1, num_diag).unsqueeze(-1)).squeeze(-1)
        batch_skip_target = A_diag[:, :, 1] - delta * torch.gather(log_probs_flip_diag, dim=2, index=targets_diag).squeeze(-1)

        # (batch_size, num_diag, 3)
        operations = torch.cat((batch_align.unsqueeze(-1), batch_skip_prediction.unsqueeze(-1), batch_skip_target.unsqueeze(-1)), dim = -1)

        # (batch_size, num_diag)
        diag_axe = torch.min(operations, dim = -1).values

        #assert logits.size(1) > targets.size(1), "assuming target length < logit length."

        if i > (logits.size(1) - targets.size(1)):
            # (batch_size, logits_length, logits_length)
            # -> (batch_size, targets_length, logits_length)
            axe = torch.diag_embed(diag_axe, offset=i, dim1=-2, dim2=-1)
            batch_A_flip[:, 1:, :-1] += axe[:, :targets.size(1), :]
        elif i > 0:
            # (batch_size, logits_length, logits_length)
            # -> (batch_size, targets_length, logits_length)
            axe = torch.diag_embed(diag_axe, offset=0, dim1=-2, dim2=-1)
            batch_A_flip[:, 1:, i : i + targets.size(1)] += axe
        else:
            axe = torch.diag_embed(diag_axe, offset=i, dim1=-2, dim2=-1)
            batch_A_flip[:, 1:, :targets.size(1)] += axe

    # recover correct order in logit dim
    batch_A = batch_A_flip.flip(-1)

    # rm 0-th row and column
    _batch_A = batch_A[:, 1:, 1:]

    ## Gather A_nm, avoiding masks
    # index_m : (batch_size, target_sequence_length, 1)
    index_m = logit_lengths.unsqueeze(-1).expand(-1, _batch_A.size(1)).unsqueeze(-1).long()

    # gather m-th colmun
    # batch_A_nm : (batch_size, target_sequence_length, 1)
    # index_m occors out of bounds for index
    batch_A_m = torch.gather(_batch_A, dim=2, index=(index_m - 1))
    batch_A_m = batch_A_m.squeeze(-1)

    # index_n : (batch_size, 1)
    index_n = target_lengths.unsqueeze(-1).long()

    # gather n-th row
    # batch_A_nm : (batch_size, 1, 1)
    batch_A_nm = torch.gather(batch_A_m, dim=1, index=(index_n - 1))

    # batch_A_nm : (batch_size)
    batch_A_nm = batch_A_nm.squeeze(-1)

    if reduction == "mean":
        axe_nm = batch_A_nm.mean()
    else:
        raise NotImplementedError

    # Refs fairseq nat_loss.
    # https://github.com/pytorch/fairseq/blob/6f6461b81ac457b381669ebc8ea2d80ea798e53a/fairseq/criterions/nat_loss.py#L70
    # actuary i'm not sure this is reasonable.
    if label_smoothing is not None and label_smoothing > 0.0:
        axe_nm = axe_nm * (1.0-label_smoothing) - log_probs.mean() * label_smoothing

    if return_a:
        return axe_nm, batch_A.detach()

    return axe_nm
