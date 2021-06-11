import torch
import torch.nn as nn
import numpy.random
import collections
import random
import copy


# Wrapper such that the model can be used with Scikit optimize
def makeLossModel(model, pc):
    if (not torch.is_tensor(pc)):
        pc = torch.tensor(pc)

    def lossModel(param):
        if (not torch.is_tensor(param)):
            param = torch.tensor(param)

        f = model(pc, param)
        loss = torch.mean(f**2)
        return loss.detach().cpu().numpy()

    return lossModel


# Wrapper class for fitting FRep models
class Model(nn.Module):
    def __init__(self,
                 frep_model,
                 lower_bound,
                 upper_bound,
                 param_init=[],
                 device='cpu'):
        super(Model, self).__init__()

        self._frep_model = frep_model

        # number of parameters
        n = len(lower_bound)

        self.param = []

        for i in range(n):
            if (len(param_init) == 0):
                xmin = lower_bound[i]
                xmax = upper_bound[i]
                param = torch.nn.Parameter(
                    torch.FloatTensor(1, 1).uniform_(xmin, xmax).to(device))
            else:
                parami = torch.FloatTensor([[param_init[i]]]).to(device)
                param = torch.nn.Parameter(parami)
            self.param.append(param)
            self.register_parameter('param' + str(i), param)

    def __call__(self, x):
        return self._frep_model(x, self.param)


# Fit parameters by sgd
def train(frep_model,
          lower_bound,
          upper_bound,
          point_cloud,
          param_init=[],
          num_iters=100,
          batch_size=1024,
          device='cpu'):
    model = Model(frep_model,
                  lower_bound,
                  upper_bound,
                  param_init=param_init,
                  device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    if (not torch.is_tensor(point_cloud)):
        point_cloud = torch.tensor(point_cloud)
        # point_cloud = point_cloud.float().to(device)
        point_cloud = point_cloud.to(device)

    for i in range(0, num_iters):
        batches = torch.utils.data.DataLoader(point_cloud,
                                              batch_size=batch_size,
                                              shuffle=True)

        for batch in batches:
            model.train()
            optimizer.zero_grad()
            point_batch = batch[:, 0:3]
            f = model(point_batch)
            loss = torch.mean(f**2)
            loss.backward()
            optimizer.step()

        #print('iter ' + str(i) + ': ' + str(loss))

    model.eval()

    parameters = []
    for param in model.parameters():
        parameters.append(param.detach().cpu().numpy())

    return parameters


# An implementation of regularized evolution
# See Algorithm 1 in https://arxiv.org/pdf/1802.01548.pdf

class Evaluator(object):
    def __init__(self, model_f, pc):
        if not torch.is_tensor(pc):
            self.pc = torch.tensor(pc)
        else:
            self.pc = pc
            
        self.model_f = model_f

    def evalModel(self, params):
        f = self.model_f(self.pc, params)
        return torch.mean(f**2)


class Individual(object):
    def __init__(self):
        self.params = None
        self.score = None


def randomParams(min_bounds, max_bounds, device='cpu'):
    sz = len(min_bounds)
    rand_params = []
    for i in range(sz):
        mini = min_bounds[i]
        maxi = max_bounds[i]
        rp = torch.FloatTensor(1, 1).uniform_(mini, maxi).to(device)
        rand_params.append(rp)
    return rand_params


def mutateParams(params, min_bounds, max_bounds, device='cpu'):
    new_params = copy.deepcopy(params)
    sz = len(new_params)
    idx = numpy.random.randint(low=0, high=sz)
    minidx = min_bounds[idx]
    maxidx = max_bounds[idx]
    new_params[idx] = torch.FloatTensor(1, 1).uniform_(minidx, maxidx).to(device)
    return new_params


def regularizedEvolution(evaluator, lower_bounds, upper_bounds, cycles, population_size, sample_size):
    population = collections.deque()
    history = []

    # initialize the population
    while (len(population) < population_size):
        c = Individual()
        c.params = randomParams(lower_bounds, upper_bounds)
        c.score = evaluator.evalModel(c.params)
        population.append(c)
        history.append(c)

    while (len(history) < cycles):
        # Sample models for the tournament
        sample = []
        while (len(sample) < sample_size):
            candidate = random.choice(list(population))
            sample.append(candidate)

        # Get the best model from sample
        parent = min(sample, key = lambda i: i.score)

        # Add a mutated model and remove the oldest
        child = Individual()
        child.params = mutateParams(parent.params, lower_bounds, upper_bounds)
        child.score = evaluator.evalModel(child.params)
        population.append(child)
        history.append(child)
        population.popleft()

    return history

