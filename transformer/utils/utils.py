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
    
def create_pad_mask(t, pad):
    mask = (t == pad).unsqueeze(-2)
    return mask


def create_trg_self_mask(target_len, device=None):
    # Prevent leftward information flow in self-attention.
    ones = torch.ones(target_len, target_len, dtype=torch.uint8,
                      device=device)
    t_self_mask = torch.triu(ones, diagonal=1).unsqueeze(0)

    return t_self_mask
