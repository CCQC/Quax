import torch
import torch.utils.checkpoint as cp

def check_backward_validity(inputs):
    if not any(inp.requires_grad for inp in inputs):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")

def detach_variable(inputs):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)


class MyCheckpointFunction(torch.autograd.Function):
    '''
    A Checkpoint that produces 1 row of the Hessian in backward() instead of the gradient.
    Which row is determined by the second argument, a torch.tensor(idx) where idx is an integer
    '''
    @staticmethod
    def forward(ctx, run_function, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.save_for_backward(*args)
        # Compute forward pass without gradients
        # Since backward() needs to return as many tensors as there were inputs,
        with torch.no_grad():
            output = run_function(args[0])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")
        inputs = ctx.saved_tensors
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            output = ctx.run_function(detached_inputs[0])

        #print('here',output)
        output.backward(create_graph=True)
        gradient = detached_inputs[0].grad.clone().flatten()
        #hess = []
        #for g in gradient:
        #    detached_inputs[0].grad.zero_()
        #    g.backward(create_graph=True)
        #    hess.append(detached_inputs[0].grad.clone())
        #hessian = tuple(hess)
        #print("HACK HESSIAN",hessian)
        
        idx = detached_inputs[1]
        detached_inputs[0].grad.zero_()
        gradient[idx].backward(create_graph=True)
        h = detached_inputs[0].grad.clone()


        #h = hessian[detached_inputs[1]]
        #h1 = hessian[0]
        #h2 = hessian[1]

        #output.backward(create_graph=True)
        #gradient = detached_inputs[0].grad.clone().flatten()
        #detached_inputs[0].grad.zero_()
        #gradient[0].backward()
        #hess = detached_inputs[0].grad.clone()


        #if isinstance(outputs, torch.Tensor):
        #    outputs = (outputs,)
        # Assume only first argument is what you want the hessian of
        #torch.autograd.backward(outputs, args, create_graph=True)
        #torch.autograd.backward(outputs, args, create_graph=True)
        #gradients = tuple(inp.grad for inp in detached_inputs)
        #gradient = gradients[0].clone().flatten()
        #hess = []
        #for g in gradient:
        #    detached_inputs[0].grad.zero_()
        #    g.backward(create_graph=True)
        #    hess.append(detached_inputs[0].grad.clone())
        #hessian = torch.stack(hess)
        #print("HACK HESSIAN",hessian)

        # Returns None and None for run_function and preserve_rng_state
        return None, h, None
        #print((None, None) + tuple(inp.grad for inp in detached_inputs))
        #return (None, None) + tuple(inp.grad for inp in detached_inputs)


def mycheckpoint(function, *args, **kwargs):
    r"""Checkpoint a model or part of the model
    Checkpointing works by trading compute for memory. Rather than storing all
    intermediate activations of the entire computation graph for computing
    backward, the checkpointed part does **not** save intermediate activations,
    and instead recomputes them in backward pass. It can be applied on any part
    of a model.
    Specifically, in the forward pass, :attr:`function` will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. Instead, the forward pass saves the inputs tuple and the
    :attr:`function` parameter. In the backwards pass, the saved inputs and
    :attr:`function` is retreived, and the forward pass is computed on
    :attr:`function` again, now tracking the intermediate activations, and then
    the gradients are calculated using these activation values.
    .. warning::
        Checkpointing doesn't work with :func:`torch.autograd.grad`, but only
        with :func:`torch.autograd.backward`.
    .. warning::
        If :attr:`function` invocation during backward does anything different
        than the one during forward, e.g., due to some global variable, the
        checkpointed version won't be equivalent, and unfortunately it can't be
        detected.
    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.
    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.
        args: tuple containing inputs to the :attr:`function`
    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))
    return MyCheckpointFunction.apply(function, *args)



