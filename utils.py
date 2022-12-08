import torch
import torch.nn as nn
import scipy
from torch.nn import functional as F


def pairwise_sim(z_mat):
    # compute pairwise cosine similarity between every pair of z's
    # z_mat: all z vectors obtained from projector, with shape batch_dim x 2048
    # batch_dim = 2N, 2 * the original batch size
    # credit to https://stackoverflow.com/questions/60467264/pairwise-similarity-matrix-between-a-set-of-vectors-in-pytorch
    # for computing the pairwise similarities
    batch_dim = z_mat.size()[0]
    sim_mat = F.cosine_similarity(z_mat[None, :, :], z_mat[:, None, :], dim=-1)
    assert sim_mat.shape == torch.Size([batch_dim, batch_dim])
    return sim_mat


def NTXent(zi, zj, sim_mat, temp):
    # compute the loss functions for simCLR
    # zi, zj: idx of z vectors
    # sim_mat: pairwise similarity matrix
    # temp: tempature scaling, hyperparameter
    numerator = torch.exp(sim_mat[zi, zj] / temp)
    denom = torch.sum(torch.exp(sim_mat[zi] / temp)) - torch.exp(torch.tensor(1) / temp)
    return -torch.log(numerator / denom)

