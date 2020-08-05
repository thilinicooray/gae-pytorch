import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def loss_function(preds, labels, mu, logvar, grouped_mu, grouped_logvar, n_nodes, norm, pos_weight):

    cost = norm* F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    '''KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))'''

    KLD = torch.mean(
        - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    )

    '''KLD_g = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp().pow(2), 1))'''

    KLD_g = torch.mean(
        - 0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp())
    )


    #print (cost, KLD, KLD_g)

    return cost + KLD + KLD_g
