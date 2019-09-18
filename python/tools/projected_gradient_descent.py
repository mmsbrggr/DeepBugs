import torch
import numpy
import scipy


def projected_gradient_descent(model, x, y, loss_fn, num_steps, step_size, eps, clamp=(-10, 10)):
    _x = x.clone().detach().numpy()[0]
    callee = torch.from_numpy(_x[:200]).requires_grad_(False)
    args = torch.from_numpy(_x[200:600]).requires_grad_(True)
    rest = torch.from_numpy(_x[600:]).requires_grad_(False)

    origin = args.clone().detach()

    for i in range(num_steps):
        _args = args.clone().detach().requires_grad_(True)

        input = torch.cat((callee, _args, rest)).unsqueeze(0)
        prediction = model(input)
        loss = loss_fn(prediction, y)
        loss.backward()

        with torch.no_grad():
            # Force the gradient step to be a fixed size in a certain norm
            gradients = _args.grad.sign() * step_size
            args += gradients

        # Project back into l_norm ball and correct range
        # Workaround as PyTorch doesn't have elementwise clip
        args = torch.max(torch.min(args, origin + eps), origin - eps)

        args = args.clamp(*clamp)

    return torch.cat((callee, args, rest)).unsqueeze(0).detach()


def get_distance(x1, x2):
    return numpy.linalg.norm(x1 - x2, numpy.inf), \
           numpy.linalg.norm(x1 - x2), \
           scipy.spatial.distance.cosine(x1, x2)


def get_most_similar_from_dict(dict, x, exclude):
    min_dist = 0
    min_name = None
    for name, vector in dict.items():
        dist = numpy.linalg.norm(vector - x)
        if min_name is None or dist < min_dist:
            if name not in exclude:
                min_dist = dist
                min_name = name

    return min_name
