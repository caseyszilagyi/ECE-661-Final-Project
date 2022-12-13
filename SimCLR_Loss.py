import torch
import torch.nn as nn
import scipy
from torch.nn import functional as F


class NTXent_Loss(nn.Module):

    def __init__(self, batch_size, temp):
        super(NTXent_Loss, self).__init__()
        self.batch_size = batch_size
        self.temp = temp

    def sim_all_pairs(self, z1, z2):
        # compute the cosine similarity matrix between every pair of z's
        # z1, z2: z vectors obtained from different augmentations, batch_size x 128
        # credit to https://stackoverflow.com/questions/60467264/pairwise-similarity-matrix-between-a-set-of-vectors-in-pytorch
        # for computing the pairwise similarities
        # could be further optimized given that this is a symmetric matrix with diagonal values=1
        # only need to store the upper/lower triangle matrix if batchsize gets huge
        z_mat = torch.cat((z1, z2), 0)
        # assert z_mat.shape == torch.Size([2 * self.batch_size, 128])
        sim_mat = F.cosine_similarity(z_mat[None, :, :], z_mat[:, None, :], dim=-1)
        # assert sim_mat.shape == torch.Size([2 * self.batch_size, 2 * self.batch_size])
        return sim_mat

    def sim_pos_pairs(self, z1, z2):
        # compute the pairwise similarity between every positive pairs
        # z1, z2: z vectors obtained from different augmentations, batch_size x 128
        pair_sim = F.cosine_similarity(z1, z2, dim=1)
        # assert pair_sim.shape == torch.Size([self.batch_size])
        return pair_sim

    def forward(self, z1, z2):
        # compute the NTXent loss, vectorized
        pos_pairs = torch.exp(self.sim_pos_pairs(z1, z2) / self.temp) # batch_size
        # assert pos_pairs.shape == torch.Size([self.batch_size])
        numerator = torch.stack((pos_pairs, pos_pairs), dim=0).permute(1, 0).reshape(-1) # 2 * batchsize, with ordering [s12, s21, s34, s43, ...]
        # assert numerator.shape == torch.Size([2 * self.batch_size])
        denom = torch.sum(torch.exp(self.sim_all_pairs(z1, z2) / self.temp), dim=-1) - torch.exp(torch.tensor(1. / self.temp)) # 2 * batch_size
        # assert denom.shape == torch.Size([2 * self.batch_size])
        result = torch.sum(-torch.log(numerator / denom)) / 2 * self.batch_size
        return result



class NTXent_Loss_updated(nn.Module):

    def __init__(self, batch_size, temp):
        super(NTXent_Loss, self).__init__()
        self.batch_size = batch_size
        self.temp = temp

    def forward(self, z1, z2):
        z_mat = torch.stack((z1,z2), dim=1).view(z1.size()[0]*2,z1.size()[1])
        sim_mat = F.cosine_similarity(z_mat[None, :, :], z_mat[:, None, :], dim=-1) / temp
        exp_sim_mat = torch.exp(sim_mat)


        sums = exp_sim_mat.sum(dim=1)
        diagonal_entries = exp_sim_mat.diagonal()

        sum = sums - diagonal_entries

        pair_sim = torch.exp(F.cosine_similarity(z1, z2, dim=1)/temp)
        pair_sim = pair_sim.repeat_interleave(2)

        numdom = pair_sim / sum

        loss = -torch.log(numdom)

        total_loss = loss.sum()
        return total_loss

    
