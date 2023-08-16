import torch
import torch.onnx

from utils.prune_utils import RC_CP_MiniMax, \
    flops2, \
    prox_w, \
    proj_dual


def rcs_optimizer(optimizer, minimax_model, s_optimizer, dual_optimizer, args):
    prox_w(minimax_model, optimizer)

    s_loss1 = minimax_model.sloss1()

    if args.need_skip:
        print("need_skip at prune_optimizer")
        s_loss2 = minimax_model.sloss2(budget=args.budget, w=minimax_model.arch_parameters)
    else:
        s_loss2 = minimax_model.sloss2(budget=args.budget)
    cur_resource = s_loss2.item() + args.budget

    resource_ub = 1

    s_grad1 = torch.autograd.grad(s_loss1, minimax_model.s, only_inputs=True)[0].data \
              + args.sl2wd * (minimax_model.s.data / minimax_model.s_ub)  # >=0

    s_grad2_temp = torch.autograd.grad(s_loss2, minimax_model.s, only_inputs=True, allow_unused=True)[0].data  # <=0

    s_grad2 = s_grad2_temp.data * resource_ub

    s_optimizer.zero_grad()
    minimax_model.s.grad = s_grad1 + minimax_model.z.data * s_grad2

    overflow_idx = minimax_model.s.data >= minimax_model.s_max
    underflow_idx = minimax_model.s.data <= 0

    minimax_model.s.grad.data[overflow_idx] = minimax_model.s.grad.data[overflow_idx].clamp(min=0.0)
    minimax_model.s.grad.data[underflow_idx] = minimax_model.s.grad.data[underflow_idx].clamp(max=0.0)

    s_optimizer.step()
    minimax_model.s.data.clamp_(min=0.0)
    overflow_idx = minimax_model.s.data >= minimax_model.s_max
    minimax_model.s.data[overflow_idx] = minimax_model.s_max[overflow_idx]

    # dual update
    dual_loss = -(minimax_model.yloss() + minimax_model.zloss(budget=args.budget))
    dual_optimizer.zero_grad()
    dual_loss.backward()

    dual_optimizer.step()

    proj_dual(minimax_model)

    return cur_resource, minimax_model.s.data.cpu().numpy(), s_grad1.data.cpu().numpy(), s_grad2.data.cpu().numpy()


def build_minimax_model(model, layer_names, bncp_layers, bncp_layers_dict, args, s=torch.zeros(1), z_init=1.0, y_init=10.0):
    minimax_model = RC_CP_MiniMax(model,
                                  resource_fn=None,
                                  bncp_layers=bncp_layers,
                                  bncp_layers_dict=bncp_layers_dict,
                                  budget=args.budget,
                                  s=s,
                                  z_init=z_init,
                                  y_init=y_init
                                  )

    cost_func = lambda s_, ub_, w=None: flops2(s_, bncp_layers_dict, bncp_layers, ub=ub_, w=w)

    # resource rough overview
    resource_ub = float(cost_func(torch.zeros_like(minimax_model.s.data), None))

    # print()
    width_mult = [1, 0.75, 0.5, 0.25, 0]
    for wm in width_mult:
        r_cost = float(cost_func(torch.round((1 - wm) * minimax_model.s_ub), None))
        print('resource cost for {} model={:.8e}'.format(wm, r_cost))

    resource_fn = lambda s_, w=None: cost_func(s_, resource_ub, w)

    minimax_model.resource_fn = resource_fn

    if args.soptim == 'adam':
        s_optimizer = torch.optim.Adam([minimax_model.s],
                                       args.slr,
                                       betas=(0.0, 0.999),
                                       weight_decay=0.0)
    elif args.soptim == 'sgd':
        s_optimizer = torch.optim.SGD([minimax_model.s],
                                      args.slr,
                                      momentum=0.0,
                                      weight_decay=0.0)
    elif args.soptim == 'rmsprop':
        s_optimizer = torch.optim.RMSprop([minimax_model.s],
                                          lr=args.slr)
    else:
        raise NotImplementedError

    dual_optimizer = torch.optim.SGD([{'params': minimax_model.z, 'lr': args.zlr},
                                      {'params': minimax_model.y, 'lr': args.ylr}, ],
                                     1.0,
                                     momentum=0.0,
                                     weight_decay=0.0)

    return minimax_model, dual_optimizer, s_optimizer
