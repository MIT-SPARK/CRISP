import pytorch3d
import torch
from pytorch3d.structures import join_meshes_as_batch


def get_output_shape(model, image_dim):
    """Helper function to get a PyTorch model's output dimension by using random data"""
    return model(torch.rand(*(image_dim))).data.shape


def cast2cuda(obj):
    return obj.cuda(non_blocking=True)


def num_trainable_params(*torch_models):
    """Return the number of trainable parameters

    Args:
        torch_models:
    """
    num_params = 0
    for m in torch_models:
        num_params += sum(param.numel() for param in m.parameters() if param.requires_grad)
    return num_params


def get_grad_norm(parameters):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.0)
    device = grads[0].device
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2.0).to(device) for g in grads]), 2.0)
    return total_norm.item()


def del_tensor_dictionary(x):
    """Delete a dictionary of tensors recursively"""
    for k, v in x.items():
        if isinstance(v, dict):
            del_tensor_dictionary(v)
        elif isinstance(v, list):
            del_tensor_iterable(v)
        elif isinstance(v, tuple):
            del_tensor_iterable(v)
        else:
            del v


def del_tensor_iterable(x):
    """Del iterable tensors"""
    for v in x:
        if isinstance(v, dict):
            del_tensor_dictionary(v)
        elif isinstance(v, list):
            del_tensor_iterable(v)
        elif isinstance(v, tuple):
            del_tensor_iterable(v)
        else:
            del v


def mesh_collate_fn(data):
    collated_batch = {}
    # Iterate over each key-value in the dictionary
    for key in data[0]:
        # Gather the data from the batch for the current key
        values = [item[key] for item in data]

        # Check if the key's values are tensors
        if isinstance(values[0], torch.Tensor):
            # If the values are tensors, stack them into a single tensor
            collated_batch[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], pytorch3d.structures.meshes.Meshes):
            collated_batch[key] = join_meshes_as_batch(values)
        else:
            collated_batch[key] = values

    return collated_batch
