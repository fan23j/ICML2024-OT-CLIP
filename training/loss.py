import torch.nn as nn
from torch.nn import functional as F
import torch

class Entropic_OT_Loss(nn.Module):
    def __init__(self):
        super(Entropic_OT_Loss, self).__init__()
        
    def get_logits(self, image_features, text_features):
        logits_per_image = image_features @ text_features.T
        logits_per_text = text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def entropic_ot(self, M, reg=0.01, n=5):
        """
        M: metric cost matrix
        reg: regularization parameter > 0
        n: number of iterations
        """

        a = torch.ones(M.shape[0])
        b = torch.ones(M.shape[0]) / M.shape[0]

        # initialize u and v as uniform distributions
        u = torch.ones(a.shape[0]) / a.shape[0]
        v = torch.ones(b.shape[0]) / b.shape[0]

        # compute kernel K using the negative cost matrix scaled by the regularization term
        K = torch.exp(-M / reg)
        # pre-compute the row normalization factor
        Kp = (1./a).reshape(-1,1) * K

        # iteratively update u and v using Sinkhorn algorithm
        for _ in range(n):
            # compute the matrix-vector product of K transposed and u
            KtransposeU = K.t() @ u
            # update v based on the current u
            v = b / KtransposeU
            # update u using the new v and precomputed Kp
            u = 1. / Kp @ v
        
        return u.reshape((-1, 1)) * K * v.reshape((-1,1))
    
    def forward(self, all_image_features, all_text_features, labels):
        logits_per_image, logits_per_text = self.get_logits(all_image_features, all_text_features)
        loss = (
            F.cross_entropy(self.entropic_ot(1.0 - logits_per_image), labels) +
            F.cross_entropy(self.entropic_ot(1.0 - logits_per_text), labels)
        ) / 2
        
        return {"entropic_ot_loss": loss}

class DBOT_Sinkhorn_Loss(nn.Module):
    def __init__(self):
        super(DBOT_Sinkhorn_Loss, self).__init__()
        
    def get_logits(self, image_features, text_features):
        logits_per_image = image_features @ text_features.T
        logits_per_text = text_features @ image_features.T
        
        return logits_per_image, logits_per_text
    
    def dbot_sinkhorn(self, M, n=5, reg=1.0):
        """
        M: metric cost matrix
        reg: regularization parameter > 0
        n: number of iterations
        """
        bs = M.shape[0]
        device = P.device

        # initialize source distribution as uniform
        a = torch.ones((bs,)).to(device)
        # define lower bound for target distribution
        b_d = torch.full((bs,), 0.1 * n).to(device)
        # define upper bound for target distribution
        b_u = torch.full((bs,), 0.9 * n).to(device)

        # init transport plan
        P = torch.exp(-M/reg)

        for _ in range(n):
            # normalize P row-wise to match source distribution
            sum_P = P.sum(dim=1)
            P = torch.diag(a / sum_P) @ P

            # adjust P not to exceed the upper bound of the target distribution
            sum_P_t = P.t().sum(dim=1)
            P = P @ torch.diag(torch.max(b_d / sum_P_t, torch.ones(P.shape[1]).to(P.device)))

            # adjust P to meet at least the lower bound of the target distribution
            sum_P_t = P.t().sum(dim=1)
            P = P @ torch.diag(torch.min(b_u / sum_P_t, torch.ones(P.shape[1]).to(P.device)))
        return P
    
    def forward(self, all_image_features, all_text_features, labels):
        logits_per_image, logits_per_text = self.get_logits(all_image_features, all_text_features)
        loss = (
            F.cross_entropy(self.dbot_sinkhorn(1.0 - logits_per_image), labels) +
            F.cross_entropy(self.dbot_sinkhorn(1.0 - logits_per_text), labels)
        ) / 2
        
        return {"dbot_sinkhorn_loss": loss}

class Fused_Gromov_Loss(nn.Module):
    def __init__(self):
        super(Fused_Gromov_Loss, self).__init__()
    
    def sinkhorn(self, C, num_iters=5):
        """
        C: cost matrix
        num_iters: number of Sinkhorn iterations
        """
        P = torch.exp(-C)
        for _ in range(num_iters):
            P = P / P.sum(dim=1, keepdim=True)
            P = P / P.sum(dim=0, keepdim=True)
        return P
    
    def fused_gromov(self, image_features, text_features, a1=0.01, a2=0.01, n=5, reg=0.01):
        """
        image_features: image features
        text_features: text features
        a1: regularization term for scaling KL divergence term 1
        a2: regularization term for scaling KL divergence term 2
        reg: regularization parameter > 0
        n: number of iterations
        """

        # KL term 1: compute the self-similarity matrix for image features
        sigma_1 = image_features @ image_features.t()
        # KL term 2: compute the self-similarity matrix for text features
        sigma_2 = text_features @ text_features.t()

        # construct initial cost matrix based on the negative dot product between image
        # and text features
        C = 1.0 - image_features @ text_features.t()
        # init transport plan
        P = F.softmax(-C / reg, dim=1)

        for _ in range(n):
            C = C - a1 * sigma_1 @ P @ sigma_2 * a2
            P = self.sinkhorn(1.0 - C, num_iters=5)

        return P
    
    def forward(self, all_image_features, all_text_features, labels):
        loss = F.cross_entropy(self.fused_gromov(all_image_features, all_text_features), labels)
        
        return {"fused_gromov_loss": loss}
    
class UFG_OT_Loss(nn.Module):
    def __init__(self):
        super(UFG_OT_Loss, self).__init__()

    def ufg_ot(self, image_features, text_features, a0=0.01, a1=0.01, a2=0.01, a3=0.01, tau=0.01, num=4, inner=4, eps=1e-5):
        """
        image_features: image features
        text_features: text features
        a0: weight of gw term
        a1: weight of entropic term
        a2: weight of KL term for p0
        a3: weight of KL term for q0
        tau: temperature parameter
        num: number of outer iterations
        inner: number of inner iterations
        eps: small value to avoid numerical instability
        """
        bs = image_features.shape[0]
        # marginal prior of dimensions
        p0 = torch.ones((bs, 0.001)).to(image_features.device)
        # marginal prior of samples
        q0 = torch.ones((bs, 0.001)).to(image_features.device)

        # cosine sim matrix of image and text features
        x = 1.0 - image_features @ text_features.t()

        # cost matrix of image features
        c1 = 1.0 - image_features @ image_features.t()
        # cost matrix of text features
        c2 = 1.0 - text_features @ text_features.t()

        # init transport t
        t = q0 * p0
        log_p0 = torch.log(p0 + eps)
        log_q0 = torch.log(q0 + eps)

        for m in range(num):
            n = min(m, a1.shape[0] -1)
            a11 = a1[n] + tau
            tmp1 = torch.matmul(c2, t)
            tmp2 = torch.matmul(tmp1, c1)
            cost = -x - a0[n] * tmp2 - tau * torch.log(t + eps)
            y = -cost / a11

            for k in range(inner):
                log_p = torch.logsumexp(y - log_p0, dim=2, keepdim=True)
                log_q = torch.logsumexp(y - log_q0, dim=1, keepdim=True)
                a = a2[n] / (a2[n] + a11) * (log_p0 - log_p)
                b = a3[n] / (a3[n] + a11) * (log_q0 - log_q)
                y = -cost / a11 + a + b
            t = torch.exp(y)
        return t
    
    def forward(self, all_image_features, all_text_features, labels):
        loss = F.cross_entropy(self.ufg_ot(all_image_features, all_text_features), labels)
        
        return {"ufg_ot_loss": loss}
