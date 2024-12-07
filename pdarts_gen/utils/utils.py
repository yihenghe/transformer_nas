import shutil
import math
import os
from copy import deepcopy

import torch
import torch.nn.functional as F


def get_loss(pred, ans, vocab_size, label_smoothing, pad):
    # took this "normalizing" from tensor2tensor. We subtract it for
    # readability. This makes no difference on learning.
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / float(vocab_size - 1)
    normalizing = -(
        confidence * math.log(confidence) + float(vocab_size - 1) *
        low_confidence * math.log(low_confidence + 1e-20))

    one_hot = torch.zeros_like(pred).scatter_(1, ans.unsqueeze(1), 1)
    one_hot = one_hot * confidence + (1 - one_hot) * low_confidence
    log_prob = F.log_softmax(pred, dim=1)

    xent = -(one_hot * log_prob).sum(dim=1)
    xent = xent.masked_select(ans != pad)
    loss = (xent - normalizing).mean()
    return loss


def get_accuracy(pred, ans, pad):
    pred = pred.max(1)[1]
    n_correct = pred.eq(ans)
    n_correct = n_correct.masked_select(ans != pad)
    return n_correct.sum().item() / n_correct.size(0)


def save_checkpoint(model, filepath, global_step, is_best):
    model_save_path = filepath + '/last_model.pt'
    torch.save(model.module.cpu(), model_save_path)
    torch.save(global_step, filepath + '/global_step.pt')
    if is_best:
        best_save_path = filepath + '/best_model.pt'
        shutil.copyfile(model_save_path, best_save_path)


def load_checkpoint(model_path, device, is_eval=True, model_name='/best_model.pt'):
    if is_eval:
        model = torch.load(model_path + model_name, map_location=device)
        model.eval()
        return model.to(device=device)

    model = torch.load(model_path + '/last_model.pt', map_location=device)
    global_step = torch.load(model_path + '/global_step.pt', map_location=device)
    return model.to(device=device), global_step

def save_checkpoints(model, model_params, t_step, num_checkpoints):
    # save checkpoints in model_params
    model_params[t_step] = dict()
    for key, param in model.module.named_parameters():
        if key not in model_params[t_step]:
            model_params[t_step][key] = param.detach().clone()
        else:
            model_params[t_step][key] += param.detach().clone()
    if len(model_params) > num_checkpoints:
        model_params.popitem(last = False)

def save_average_checkpoints(model, model_params, filepath):
    model_save_path = filepath + '/average_model.pt'
    avg_params = dict()
    for _, v in model_params.items():
        for key, param in v.items():
            if key not in avg_params:
                avg_params[key] = param.clone()
            else:
                avg_params[key] += param.clone()
    for k, _ in avg_params.items():
        avg_params[k].div_(len(model_params))
    avg_model = deepcopy(model.module)
    with torch.no_grad():
        for key, param in avg_model.named_parameters():
            param.copy_(avg_params[key])
    torch.save(avg_model.cpu(), model_save_path)

def save_optimizers(data_dir, t_step, optimizer, architect, perturb):
    path = data_dir + '/last/models/summary.pt'
    torch.save({
                "t_step": t_step,
                "optimizer": optimizer.optimizer,
                "architect_optimizer": architect.optimizer,
                "perturb_optimizer": perturb.optimizer
               }, path)

def load_optimizers(data_dir, optimizer, architect, perturb):
    path = data_dir + '/last/models/summary.pt'
    checkpoint = torch.load(path)
    t_step = checkpoint["t_step"]
    optimizer.optimizer = checkpoint["optimizer"]
    architect.optimizer = checkpoint["architect_optimizer"]
    perturb.optimizer = checkpoint["perturb_optimizer"]
    return t_step

def create_pad_mask(t, pad):
    mask = (t == pad).unsqueeze(-2)
    return mask


def create_trg_self_mask(target_len, device=None):
    # Prevent leftward information flow in self-attention.
    ones = torch.ones(target_len, target_len, dtype=torch.uint8,
                      device=device)
    t_self_mask = torch.triu(ones, diagonal=1).unsqueeze(0)

    return t_self_mask

def mmd(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    if kernel == "l2":
        return torch.dist(x, y, p=2)
    # reshape to 2d
    x = x.view(x.size(0), x.size(1) * x.size(2))
    y = y.view(y.size(0), y.size(1) * y.size(2))
    
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros_like(xx),
                  torch.zeros_like(xx),
                  torch.zeros_like(xx))
    
    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
    elif kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
    
    return torch.mean(XX + YY - 2. * XY)
