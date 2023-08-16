import torch
from torch import nn
from torch.nn import functional as F, Parameter


def get_bncp_layers(model):
    for name, p in model.named_modules():
        if hasattr(p, "weight") and isinstance(p, (nn.Conv2d, nn.Linear)):
            p.register_buffer("mask", torch.ones_like(p.weight))

    layer_names = {}
    layer_names[None] = None
    bncp_layers = []

    ignore_first_layer = True

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if ignore_first_layer:
                ignore_first_layer = False
            else:
                layer_names[m] = name
                bncp_layers.append(m)

    bncp_layers_dict = {}
    for i, m in enumerate(bncp_layers):
        bncp_layers_dict[m] = i
    print('=================================================')
    print(bncp_layers_dict)
    print(bncp_layers)
    return layer_names, bncp_layers, bncp_layers_dict


def array1d_repr(t, format='{:.3f}'):
    res = ''
    for i in range(len(t)):
        res += format.format(float(t[i]))
        if i < len(t) - 1:
            res += ', '

    return '[' + res + ']'


def array2d_repr(t, format='{:.3f}'):
    res = ''
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            res += format.format(float(t[i, j]))
            if i < t.shape[0] and j < t.shape[1]:
                res += ', '

    return '[' + res + ']'


class SteFloor(torch.autograd.Function):
    """
    Ste for floor function
    """

    @staticmethod
    def forward(ctx, a):
        return a.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


ste_floor = SteFloor.apply


class SteCeil(torch.autograd.Function):
    """
    Ste for ceil function
    """

    @staticmethod
    def forward(ctx, a):
        return a.ceil()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


ste_ceil = SteCeil.apply


def weight_list_to_scores(layer):
    result = layer.weight.data ** 2
    return result


def all_weight_list_to_scores(bncp_layers):
    all_weight_list_to_scores = weight_list_to_scores(bncp_layers[0]).view(-1)
    for layer in bncp_layers[1:]:
        all_weight_list_to_scores = torch.cat((all_weight_list_to_scores, weight_list_to_scores(layer).view(-1)), dim=0)

    return all_weight_list_to_scores


class LeastSsum(torch.autograd.Function):  # sum of norm of least s groups
    @staticmethod
    def forward(ctx, s, all_weight_list):
        all_weight_list = all_weight_list.view(-1)
        idx = int(s.ceil().item()) + 1
        if idx <= all_weight_list.numel():
            vec_least_sp1 = torch.topk(all_weight_list, idx, largest=False, sorted=True)[0]  # bottom s+1 individual values
            ctx.vec_sp1_least = vec_least_sp1[-1].item()  # s+1 -th value
            return vec_least_sp1[:-1].sum()  # bottom s value sum
        else:
            ctx.vec_sp1_least = all_weight_list.max().item()
            return all_weight_list.sum()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.vec_sp1_least, None


least_s_sum = LeastSsum.apply


def flops2(s, bncp_layers_dict, bncp_layers, ub=None, w=None):
    if w is not None:
        w = F.softmax(w, dim=-1)

    res = 0

    for i, layer in enumerate(bncp_layers):  # MLP pruning
        linear_r = layer.weight.numel()
        res += linear_r
    res -= ste_floor(s[0])
    return res / ub if ub is not None else res


class RC_CP_MiniMax(nn.Module):
    def __init__(self, model, resource_fn, bncp_layers, bncp_layers_dict, budget, z_init=1.0, y_init=10.0,
                 s=torch.zeros(1)):
        super(RC_CP_MiniMax, self).__init__()
        self.model = model
        device = str(next(model.parameters()).device)
        self.bncp_layers = bncp_layers
        self.bncp_layers_dict = bncp_layers_dict
        self.s = Parameter(s, requires_grad=False).to(device)
        self.s.requires_grad = True
        self.y = Parameter(torch.zeros(1), requires_grad=False).to(device)
        self.y.data.fill_(y_init)
        self.y.requires_grad = True
        self.z = Parameter(torch.tensor(float(z_init)), requires_grad=False).to(device)
        self.z.requires_grad = True
        self.resource_fn = resource_fn
        self.budget = budget

        self.__least_s_norm = torch.zeros_like(self.s.data).to(device)
        self.s_ub = torch.zeros_like(self.s.data).to(device)
        self.res_up = torch.zeros_like(self.s.data).to(device)

        for layer, idx in self.bncp_layers_dict.items():
            self.res_up[0] += layer.weight.numel()
        self.s_ub[0] = self.res_up[0] * (1 - self.budget) + 1

        self.s_max = (self.s_ub - 1 - 1e-8).clamp(min=0.0)

        print("self.s_ub", self.s_ub.data)
        print("self.s_max", self.s_max.data)
        print("self.res_up", self.res_up.data)
        print("prune budget", (1 - self.s_ub.sum() / self.res_up.sum()))
        print(self.s_ub.sum())

    def ceiled_s(self):
        return ste_ceil(self.s)

    def sloss1(self):
        s = self.ceiled_s()

        w_s_norm = least_s_sum(s[0], all_weight_list_to_scores(self.bncp_layers)).view(-1)
        result = self.y.data.dot(w_s_norm)
        return result

    def sloss2(self, budget, w=None):
        s = self.ceiled_s()
        if w is not None:
            w = w.data
        rc = self.resource_fn(s, w=w)
        return rc - budget

    def skip_resource_loss(self, budget, w):
        s = self.ceiled_s().data
        rc = self.resource_fn(s, w=w)
        return rc - budget

    def get_least_s_norm(self):
        res = self.__least_s_norm
        s = self.ceiled_s()

        all_scores = all_weight_list_to_scores(self.bncp_layers)
        all_score_list = all_scores.view(-1)

        res[0] = torch.topk(all_score_list, int(s[0].ceil().item()), largest=False, sorted=False)[0].sum().item()

        return res

    def yloss(self):
        temp = self.get_least_s_norm()
        return self.y.dot(temp)

    def zloss(self, budget):
        return self.z * (self.resource_fn(self.ceiled_s().data) - budget)


def prox_w(minimax_model, optimizer):
    lr = optimizer.param_groups[0]['lr']

    s = minimax_model.ceiled_s()
    all_scores = all_weight_list_to_scores(minimax_model.bncp_layers)
    all_score_list = all_scores.view(-1)
    # the k th smallest element of score and score_list
    s_kthvalue = torch.kthvalue(all_score_list, int(s[0].ceil().item()) + 1)[0]

    for layer in minimax_model.bncp_layers:
        scores = weight_list_to_scores(layer)  # size is weight.shape
        least_s_idx = scores < s_kthvalue
        layer.weight.data[least_s_idx] /= (1.0 + 2.0 * lr * minimax_model.y[0].item())


def proj_dual(minimax_model):
    minimax_model.y.data.clamp_(min=0.0)
    minimax_model.z.data.clamp_(min=0.0)


def prune_w_mask(minimax_model):
    s = minimax_model.ceiled_s()

    all_scores = all_weight_list_to_scores(minimax_model.bncp_layers)
    all_score_list = all_scores.view(-1)
    # the k th smallest element of score and score_list
    s_kthvalue = torch.kthvalue(all_score_list, int(s[0].ceil().item()) + 1)[0]

    for layer in minimax_model.bncp_layers:
        layer.mask.data[:] = 1
        score = weight_list_to_scores(layer)  # size is weight.shape
        least_s_idx = score < s_kthvalue
        layer.mask.data[least_s_idx] = 0
