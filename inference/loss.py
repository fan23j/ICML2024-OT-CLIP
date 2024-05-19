from torch.nn import functional as F
import torch

def fgwi_softmax(image_features, text_features, a1, a2, reg, n):
    """
    image_features: image features
    text_features: text features
    a1: regularization term for scaling the KL divergence term 1
    a2: regularization term for scaling the KL divergence term 2
    reg: regularization parameter > 0
    n: number of iterations for adjusting the structural cost matrix
    """

    # KL term 1: compute the self-similarity matrix for image features
    sigma_1 = image_features @ image_features.t()
    # KL term 2: compute the self-similarity matrix for text features
    sigma_2 = text_features @ text_features.t()

    # init cost matrix
    C = 1.0 - image_features @ text_features.t()
    # init transport plan
    P = F.softmax(-C / reg)

    for _ in range(n):
        C = C - a1 * sigma_1 @ P @ sigma_2 * a2
        P = F.softmax(-C/ reg)
    return P

def fgw_pgrad(image_features, text_features, a1, a2, reg, n):
    """
    image_features: image features
    text_features: text features
    a1: regularization term for scaling the KL divergence term 1
    a2: regularization term for scaling the KL divergence term 2
    reg: regularization parameter > 0
    n: number of iterations for adjusting the structural cost matrix
    """

    # KL term 1: compute the self-similarity matrix for image features
    sigma_1 = image_features @ image_features.t()
    # KL term 2: compute the self-similarity matrix for text features
    sigma_2 = text_features @ text_features.t()

    # init cost matrix
    C = 1.0 - image_features @ text_features.t()
    # init transport plan
    P = F.softmax(-C / reg)

    for _ in range(n):
        grad = C - a1 * sigma_1 @ P @ sigma_2 * a2
        K = P * torch.exp(-grad / reg)
        P = K / K.sum(dim=1, keepdim=True)

    return P


def partial_ot(image_features, text_features, reg, n, num_class):
    """
    image_features: image features
    text_features: text features
    reg: regularization parameter > 0
    n: number of iterations for adjusting the structural cost matrix
    num_class: number of prediction classes
    """

    P = torch.exp(image_features @ text_features.t()) / reg
    m = torch.full((num_class,), 1/num_class)

    for _ in range(n):
        row_P = P.sum(dim=1)
        scaling = torch.max(row_P, 1)
        P = torch.div(P, scaling)
        P *= m/P.sum(dim=1)
    return P