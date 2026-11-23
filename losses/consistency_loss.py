import torch
import torch.nn as nn
import torch.nn.functional as F

class TotalConsistencyLoss(nn.Module):
    def __init__(self, class_weights=None, lambda_cons=0.5, eps=1e-7):
        super().__init__()
        self.lambda_cons = lambda_cons
        self.eps = eps
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def task_loss(self, logits, targets):
        """
        Class-weighted cross-entropy loss over labeled nodes.
        logits: (N, C) unnormalized scores
        targets: (N,) class labels {0, 1, ..., C-1}
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + self.eps)
        
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            loss = F.nll_loss(log_probs, targets, weight=self.class_weights, reduction='mean')
        else:
            loss = F.nll_loss(log_probs, targets, reduction='mean')
        return loss

    def consistency_loss(self, logits_clean, logits_pert):
        """
        Symmetric Kullback-Leibler divergence between clean and perturbed posteriors.
        Gradients flow through both branches.
        logits_clean: (N, C)
        logits_pert: (N, C)
        """
        p = F.softmax(logits_clean, dim=-1)
        q = F.softmax(logits_pert, dim=-1)

        kl_pq = torch.sum(p * (torch.log(p + self.eps) - torch.log(q + self.eps)), dim=-1)
        kl_qp = torch.sum(q * (torch.log(q + self.eps) - torch.log(p + self.eps)), dim=-1)

        sym_kl = 0.5 * torch.mean(kl_pq + kl_qp)
        return sym_kl

    def forward(self, logits_clean, targets, logits_pert=None):
        loss_task = self.task_loss(logits_clean, targets)
        if logits_pert is not None and self.lambda_cons > 0:
            loss_cons = self.consistency_loss(logits_clean, logits_pert)
            total_loss = loss_task + self.lambda_cons * loss_cons
            return total_loss, loss_task, loss_cons
        return loss_task, loss_task, torch.tensor(0.0, device=logits_clean.device)
