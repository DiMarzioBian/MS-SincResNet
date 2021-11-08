import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, p=2., mining_type='all'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.mining = mining_type

        if mining_type == 'all':
            self.loss_function = all_triplet_loss
        if mining_type == 'hard':
            self.loss_function = hard_triplet_loss

    def forward(self, embeddings, labels):
        return self.loss_function(labels, embeddings, self.margin, self.p)


def hard_triplet_loss(labels, embeddings, margin, p):
    pairwise_distribution = torch.cdist(embeddings, embeddings, p=p)

    mask_anchor_positive = _fetch_anchor_positive_triplet_mask(labels).float()
    anchor_positive_distribution = mask_anchor_positive * pairwise_distribution

    # hardest positive for every anchor
    hardest_positive_distribution, _ = anchor_positive_distribution.max(1, keepdim=True)

    mask_anchor_negative = _fetch_anchor_negative_triplet_mask(labels).float()

    # Add max value in each row to invalid negatives
    max_anchor_negative_distribution, _ = pairwise_distribution.max(1, keepdim=True)
    anchor_negative_distribution = pairwise_distribution + max_anchor_negative_distribution * (1.0 - mask_anchor_negative)

    # hardest negative for every anchor
    hardest_negative_distribution, _ = anchor_negative_distribution.min(1, keepdim=True)

    triplet_loss = hardest_positive_distribution - hardest_negative_distribution + margin
    triplet_loss[triplet_loss < 0] = 0

    triplet_loss = triplet_loss.mean()

    return triplet_loss, -1


def all_triplet_loss(labels, embeddings, margin, p):
    pairwise_distribution = torch.cdist(embeddings, embeddings, p=p)

    anchor_positive_distribution = pairwise_distribution.unsqueeze(2)
    anchor_negative_distribution = pairwise_distribution.unsqueeze(1)

    triplet_loss = anchor_positive_distribution - anchor_negative_distribution + margin

    mask = _fetch_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (easy triplets)
    triplet_loss[triplet_loss < 0] = 0

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def _fetch_triplet_mask(labels):
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels & distinct_indices


def _fetch_anchor_positive_triplet_mask(labels):
    # Check that i and j are distinct
    check_indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    indices_not_equal = ~check_indices_equal

    # Check if labels[i] == labels[j]
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal


def _fetch_anchor_negative_triplet_mask(labels):
    return labels.unsqueeze(0) != labels.unsqueeze(1)