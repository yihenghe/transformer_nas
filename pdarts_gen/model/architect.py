import torch
import numpy as np
import torch.nn as nn
import gc
from copy import deepcopy

from utils import utils

def concat(xs):
  return torch.cat([x.view(-1) for x in xs])

class Architect(object):
    def __init__(self, model, args):
        self.network_beta1 = 0.9
        self.network_beta2 = 0.997
        self.network_eps = 1e-09
        self.network_weight_decay = 0
        self.perturb_beta1 = 0.9
        self.perturb_beta2 = 0.997
        self.perturb_eps = 1e-08
        self.perturb_weight_decay = 1e-3
        self.perturb_lr = 3e-4
        self.model = model
        self.t_vocab_size = args.t_vocab_size
        self.label_smoothing = args.label_smoothing
        self.trg_pad_idx = args.trg_pad_idx
        self.optimizer = torch.optim.Adam(self.model.module.arch_params(),
            lr = 3e-4, betas=(0.9, 0.997), weight_decay = 1e-3)

    def compute_unrolled_model_theta(self, inputs, targets, eta, network_optimizer):
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
        
    def compute_unrolled_model_delta(self, model, inputs, targets, perturb_optimizer):
        pred = model(inputs=inputs, targets=targets, perturb=True)

        pred = pred.view(-1, pred.size(-1))
        ans = targets.view(-1)

        loss = utils.get_loss(pred, ans, self.t_vocab_size,
                                self.label_smoothing, self.trg_pad_idx)
        
        delta = concat(model.module.perturb_params()).detach()
        ddelta = concat(torch.autograd.grad(loss, model.module.perturb_params())).detach() + self.perturb_weight_decay * delta
        try:
            step = perturb_optimizer.state[next(model.module.perturb_params())]['step']
            m_t = concat(perturb_optimizer.state[v]['exp_avg'] for v in model.module.perturb_params()) * self.perturb_beta1 + (1 - self.perturb_beta1) * ddelta
            m_t_hat = m_t / (1 - self.perturb_beta1 ** step)
            v_t = concat(perturb_optimizer.state[v]['exp_avg_sq'] for v in model.module.perturb_params()) * self.perturb_beta2 + (1 - self.perturb_beta2) * (ddelta ** 2)
            v_t_hat = v_t / (1 - self.perturb_beta2 ** step)
            new_delta = delta.sub(m_t_hat / (v_t_hat ** 0.5 + self.perturb_eps), alpha = self.perturb_lr)
        except:
            new_delta = delta.sub(ddelta, alpha = self.perturb_lr)
        unrolled_model = self.construct_model_from_delta(model, new_delta)
        del pred, ans, loss, delta, ddelta, new_delta
        gc.collect()
        return unrolled_model
    
    def construct_model_from_delta(self, model, delta):
        model_new = deepcopy(model)
        model_dict = model.state_dict()

        params, offset = {}, 0
        for k, v in model.named_parameters():
            if "i_perturb" in k:
                v_length = np.prod(v.size())
                params[k] = delta[offset: offset+v_length].view(v.size())
                offset += v_length

        assert offset == len(delta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        del model_dict, params, offset
        gc.collect()
        return model_new
        
    def step(self, inputs_train, targets_train, inputs_valid, targets_valid, eta, network_optimizer, unrolled, perturb_optimizer):
        self.optimizer.zero_grad()
        if unrolled:
            self.backward_step_unrolled(inputs_train, targets_train, inputs_valid, targets_valid, eta, network_optimizer, perturb_optimizer)
        else:
            self.backward_step(inputs_valid, targets_valid)
        self.optimizer.step()

    def backward_step(self, inputs_valid, targets_valid):
        pred = self.model(inputs=inputs_valid, targets=targets_valid, perturb=True)

        pred = pred.view(-1, pred.size(-1))
        ans = targets_valid.view(-1)
        loss = utils.get_loss(pred, ans, self.t_vocab_size, self.label_smoothing,
                                  self.trg_pad_idx)
        loss.backward()
        del pred, ans, loss
        gc.collect()

    def backward_step_unrolled(self, inputs_train, targets_train, inputs_valid, targets_valid, eta, network_optimizer, perturb_optimizer):
        unrolled_model_theta = self.compute_unrolled_model_theta(inputs_train, targets_train, eta, network_optimizer)
        unrolled_model = self.compute_unrolled_model_delta(unrolled_model_theta, inputs_valid, targets_valid, perturb_optimizer)
        
        pred = unrolled_model(inputs=inputs_valid, targets=targets_valid, perturb=True)

        pred = pred.view(-1, pred.size(-1))
        ans = targets_valid.view(-1)
        loss = utils.get_loss(pred, ans, self.t_vocab_size, self.label_smoothing,
                                  self.trg_pad_idx)

        loss.backward()
        
        del pred, ans, loss
        gc.collect()
        
        dalpha = [v.grad for v in unrolled_model.module.arch_params()]
        
        # w.r.t. weight
        vector_weight = [v.grad.detach() for v in unrolled_model.module.weight_params()]
        implicit_grads_weight = self.hessian_vector_product_weight(vector_weight, inputs_train, targets_train)
        
        for g, ig in zip(dalpha, implicit_grads_weight):
            g.data.sub_(ig.data, alpha=eta)
        
        del vector_weight, implicit_grads_weight
        gc.collect()
        
        # w.r.t delta
        vector_delta = [v.grad.detach() for v in unrolled_model.module.perturb_params()]
        implicit_grads_delta, vector_weight_delta = self.hessian_vector_product_delta(unrolled_model_theta, vector_delta, inputs_valid, targets_valid)
        
        for g, ig in zip(dalpha, implicit_grads_delta):
            g.data.sub_(ig.data, alpha=self.perturb_lr)
            
        del vector_delta, implicit_grads_delta
        gc.collect()
        
        implicit_grads_weight_delta = self.hessian_vector_product_weight(vector_weight_delta, inputs_train, targets_train)
        
        for g, ig in zip(dalpha, implicit_grads_weight_delta):
            g.data.add_(ig.data, alpha=eta * self.perturb_lr)
        
        for v, g in zip(self.model.module.arch_params(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)
        
        del unrolled_model, dalpha, vector_weight_delta, implicit_grads_weight_delta
        gc.collect()

    def hessian_vector_product_weight(self, vector, inputs, targets, r=1e-2):
        R = r / concat(vector).norm()
        for p, v in zip(self.model.module.weight_params(), vector):
            p.data.add_(v, alpha=R)
        pred = self.model(inputs, targets)

        pred = pred.view(-1, pred.size(-1))
        ans = targets.view(-1)

        loss = utils.get_loss(pred, ans, self.t_vocab_size,
                              self.label_smoothing, self.trg_pad_idx)
        grads_p = torch.autograd.grad(loss, self.model.module.arch_params())
        
        del pred, ans, loss
        gc.collect()
        for p, v in zip(self.model.module.weight_params(), vector):
            p.data.sub_(v, alpha=2*R)
        pred = self.model(inputs, targets)

        pred = pred.view(-1, pred.size(-1))
        ans = targets.view(-1)

        loss = utils.get_loss(pred, ans, self.t_vocab_size,
                              self.label_smoothing, self.trg_pad_idx)
        grads_n = torch.autograd.grad(loss, self.model.module.arch_params())

        for p, v in zip(self.model.module.weight_params(), vector):
            p.data.add_(v, alpha=R)

        result = [(x-y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
        del pred, ans, loss, grads_p, grads_n
        gc.collect()
        return result

    
    def hessian_vector_product_delta(self, model, vector, inputs, targets, r=1e-2):
        R = r / concat(vector).norm()
        for p, v in zip(model.module.perturb_params(), vector):
            p.data.add_(v, alpha=R)
        pred = model(inputs=inputs, targets=targets, perturb=True)

        pred = pred.view(-1, pred.size(-1))
        ans = targets.view(-1)

        loss = utils.get_loss(pred, ans, self.t_vocab_size,
                                self.label_smoothing, self.trg_pad_idx)
        grads_p_a = torch.autograd.grad(loss, model.module.arch_params(), retain_graph=True)
        grads_p_w = torch.autograd.grad(loss, model.module.weight_params(), retain_graph=False)
        
        del pred, ans, loss
        gc.collect()
        for p, v in zip(model.module.perturb_params(), vector):
            p.data.sub_(v, alpha=2*R)
        pred = model(inputs=inputs, targets=targets, perturb=True)

        pred = pred.view(-1, pred.size(-1))
        ans = targets.view(-1)

        loss = utils.get_loss(pred, ans, self.t_vocab_size,
                                self.label_smoothing, self.trg_pad_idx)
        grads_n_a = torch.autograd.grad(loss, model.module.arch_params(), retain_graph=True)
        grads_n_w = torch.autograd.grad(loss, model.module.weight_params(), retain_graph=False)

        for p, v in zip(model.module.perturb_params(), vector):
            p.data.add_(v, alpha=R)
        
        result = ([(x-y).div_(2 * R) for x, y in zip(grads_p_a, grads_n_a)], [(x-y).div_(2 * R) for x, y in zip(grads_p_w, grads_n_w)])
        del grads_p_a, grads_p_w, pred, ans, loss, grads_n_a, grads_n_w
        gc.collect()
        return result
