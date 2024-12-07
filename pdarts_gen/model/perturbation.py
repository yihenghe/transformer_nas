import torch
import numpy as np
import torch.nn as nn
import gc
from copy import deepcopy

from utils import utils

def concat(xs):
  return torch.cat([x.view(-1) for x in xs])

class Perturbation(object):
    def __init__(self, model, args):
        self.network_beta1 = 0.9
        self.network_beta2 = 0.997
        self.network_eps = 1e-09
        self.network_weight_decay = 0
        self.model = model
        self.lambda_ = args.lambda_
        self.kernel = args.kernel
        self.t_vocab_size = args.t_vocab_size
        self.label_smoothing = args.label_smoothing
        self.trg_pad_idx = args.trg_pad_idx
        self.optimizer = torch.optim.Adam(self.model.module.perturb_params(),
            lr = 3e-4, betas=(0.9, 0.997), weight_decay = 1e-3)

    def compute_unrolled_model(self, inputs, targets, eta, network_optimizer):
        pred = self.model(inputs, targets)

        pred = pred.view(-1, pred.size(-1))
        ans = targets.view(-1)

        loss = utils.get_loss(pred, ans, self.t_vocab_size,
                              self.label_smoothing, self.trg_pad_idx)
        
        theta = concat(self.model.module.weight_params()).detach()
        dtheta = concat(torch.autograd.grad(loss, self.model.module.weight_params())).detach() + self.network_weight_decay * theta
        try:
            step = network_optimizer.state[next(self.model.module.weight_params())]['step']
            m_t = concat(network_optimizer.state[v]['exp_avg'] for v in self.model.module.weight_params()) * self.network_beta1 + (1 - self.network_beta1) * dtheta
            m_t_hat = m_t / (1 - self.network_beta1 ** step)
            v_t = concat(network_optimizer.state[v]['exp_avg_sq'] for v in self.model.module.weight_params()) * self.network_beta2 + (1 - self.network_beta2) * (dtheta ** 2)
            v_t_hat = v_t / (1 - self.network_beta2 ** step)
            new_theta = theta.sub(m_t_hat / (v_t_hat ** 0.5 + self.network_eps), alpha = eta)
        except:
            new_theta = theta.sub(dtheta, alpha = eta)
        unrolled_model = self.construct_model_from_theta(new_theta)
        del pred, ans, loss, theta, dtheta, new_theta
        gc.collect()
        return unrolled_model

    def step(self, inputs_train, targets_train, inputs_valid, targets_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self.backward_step_unrolled(inputs_train, targets_train, inputs_valid, targets_valid, eta, network_optimizer)
        else:
            self.backward_step(inputs_valid, targets_valid)
        self.optimizer.step()
            

    def backward_step(self, inputs_valid, targets_valid):
        pred = self.model(inputs=inputs_valid, targets=targets_valid, perturb=True)

        pred = pred.view(-1, pred.size(-1))
        ans = targets_valid.view(-1)
        loss = utils.get_loss(pred, ans, self.t_vocab_size, self.label_smoothing,
                                  self.trg_pad_idx)
        inputs_embedded, inputs_perturbed = self.model.module.perturb_data(inputs_valid)
        loss -= self.lambda_ * utils.mmd(inputs_embedded, inputs_perturbed, self.kernel)
        loss.backward()
        del pred, ans, inputs_embedded, inputs_perturbed, loss
        gc.collect()

    def backward_step_unrolled(self, inputs_train, targets_train, inputs_valid, targets_valid, eta, network_optimizer):
        unrolled_model = self.compute_unrolled_model(inputs_train, targets_train, eta, network_optimizer)
        pred = unrolled_model(inputs=inputs_valid, targets=targets_valid, perturb=True)

        pred = pred.view(-1, pred.size(-1))
        ans = targets_valid.view(-1)
        loss = utils.get_loss(pred, ans, self.t_vocab_size, self.label_smoothing,
                                  self.trg_pad_idx)
        inputs_embedded, inputs_perturbed = unrolled_model.module.perturb_data(inputs_valid)
        loss -= self.lambda_ * utils.mmd(inputs_embedded, inputs_perturbed, self.kernel)
        loss.backward()
        del unrolled_model, pred, ans, inputs_embedded, inputs_perturbed, loss
        gc.collect()

    def construct_model_from_theta(self, theta):
        model_new = deepcopy(self.model)
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            if "alphas_encoder" in k or "alphas_decoder" in k or "i_perturb" in k:
                continue
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        del model_dict, params
        gc.collect()
        return model_new



