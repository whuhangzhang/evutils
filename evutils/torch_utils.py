# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from collections import OrderedDict


def load_model_weight(model, checkpoint, logger):
    # ref: https://github.com/RangiLyu/nanodet/blob/v0.4.1/nanodet/util/check_point.py
    state_dict = checkpoint['state_dict']
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                logger.log('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            logger.log('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            logger.log('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)


def save_model(model, path, epoch, iter, optimizer=None):
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    data = {'epoch': epoch,
            'state_dict': model_state_dict,
            'iter': iter}
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()

    torch.save(data, path)


def rename_state_dict_keys(source, key_transformation, target=None):
    """Rename state dict keys

    Args:
        source             -> Path to the saved state dict.
        key_transformation -> Function that accepts the old key names of the state
                            dict as the only argument and returns the new key name.
        target (optional)  -> Path at which the new state dict should be saved
                            (defaults to `source`)
        Example:
        Rename the key `layer.0.weight` `layer.1.weight` and keep the names of all
        other keys.

    Example:
        ```
        def key_transformation(old_key):
            if old_key == "layer.0.weight":
                return "layer.1.weight"
            return old_key
        rename_state_dict_keys(state_dict_path, key_transformation)
        ```
    """
    if target is None:
        target = source

    state_dict = torch.load(source)
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    torch.save(new_state_dict, target)


def fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    ref: https://github.com/Megvii-BaseDetection/YOLOX/blob/0.3.0/yolox/utils/model_utils.py
    Fuse convolution and batchnorm layers.
    check more info on https://tehnokv.com/posts/fusing-batchnorm-and-conv/

    Args:
        conv (nn.Conv2d): convolution to fuse.
        bn (nn.BatchNorm2d): batchnorm to fuse.

    Returns:
        nn.Conv2d: fused convolution behaves the same as the input conv and bn.
    """
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_model(model: nn.Module) -> nn.Module:
    """fuse conv and bn in model

    Args:
        model (nn.Module): model to fuse

    Returns:
        nn.Module: fused model
    """
    from yolox.models.network_blocks import BaseConv

    for m in model.modules():
        if type(m) is BaseConv and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, "bn")  # remove batchnorm
            m.forward = m.fuseforward  # update forward
    return model


def replace_module(module, replaced_module_type, new_module_type, replace_func=None) -> nn.Module:
    """
    Replace given type in module to a new type. mostly used in deploy.

    Args:
        module (nn.Module): model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type)
        replace_func (function): python function to describe replace logic. Defalut value None.

    Returns:
        model (nn.Module): module that already been replaced.
    """

    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model


def freeze_module(module: nn.Module, name=None) -> nn.Module:
    """freeze module inplace

    Args:
        module (nn.Module): module to freeze.
        name (str, optional): name to freeze. If not given, freeze the whole module.
            Note that fuzzy match is not supported. Defaults to None.

    Examples:
        freeze the backbone of model
        >>> freeze_moudle(model.backbone)

        or freeze the backbone of model by name
        >>> freeze_moudle(model, name="backbone")
    """
    for param_name, parameter in module.named_parameters():
        if name is None or name in param_name:
            parameter.requires_grad = False

    # ensure module like BN and dropout are freezed
    for module_name, sub_module in module.named_modules():
        # actually there are no needs to call eval for every single sub_module
        if name is None or name in module_name:
            sub_module.eval()

    return module
