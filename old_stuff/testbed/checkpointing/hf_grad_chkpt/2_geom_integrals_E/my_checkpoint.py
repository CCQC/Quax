import torch

class HessCheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, row_idx, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.row_idx = row_idx
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*args)
        ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        print("CHECKPOINT BACKWARD INITIATED")
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")
        inputs = ctx.saved_tensors
        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrouding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
            detached_inputs = detach_variable(inputs)
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs)

        # Function has now been computed with gradients enabled.
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        # Compute gradient of outputs
        #torch.autograd.backward(outputs, args, create_graph=True)
        print("Calling first backward")
        torch.autograd.backward(outputs, args, create_graph=True)
        grads = tuple(inp.grad.clone() for inp in detached_inputs)
        print("Gradient printing")
        print(grads)
        #torch.save(grads[0].flatten(), 'grads.pt')
        torch.save(grads[0], 'grads.pt')
        #print("Saved 'grads' file:")
        #print(grads[0].flatten())

        [inp.grad.zero_() for inp in detached_inputs]
        print("Zero-gradient check")
        print(detached_inputs[0].grad)

        #gradient = detached_inputs[0].grad.clone().flatten()
        #print('GRADIENT',gradient)
        ##detached_inputs[0].grad.zero_()
        #grads[0][0][2].backward(retain_graph=True)
        #grads[0].flatten()[ctx.row_idx].backward(retain_graph=True)

        print("Calling second backward")
        grads[0].flatten()[ctx.row_idx].backward(retain_graph=True)
       # grads[0][ctx.row_idx].backward(retain_graph=True)
        gotit = detached_inputs[0].grad.clone()
        torch.save(gotit, 'hess.pt')
        print("Saved 'hess' file:")
        print(gotit)
        #print("Currently held gradients + 1st gradients:")
        #print(gotit + grads[0].flatten())



        #print(detached_inputs[0].grad)
        #print("printing hess")
        #print(detached_inputs[0].grad)
        #print("hess printing done")
        ####grads = 
        #print("printing final hess")
        #print(detached_inputs[0].grad + grads[0])
        #print("final hess done")
        #grads[0][0][2].grad.zero_()

        ##gradients = []
        ##for i in detached_inputs:
        ##     gradients.append(i.grad.clone().flatten())
        ##    i.grad.zero_()
        ##print(gradients)
        ##gradient = tuple(gradients)

        ##gradient = detached_inputs[0].grad.clone().flatten()
        #detached_inputs[0].grad.zero_()
        ##detached_inputs = [inp.grad.zero_() for inp in detached_inputs]

        ### Hessian of 0th,1st, --->2nd<--- gradient
        ##torch.autograd.backward(gradient, args, create_graph=True)
        ##gradient[ctx.row_idx].backward(retain_graph=True)
        ##gradient2.backward(retain_graph=True)
        #gradient[ctx.row_idx].backward(retain_graph=True)
        ##gradient[ctx.row_idx].backward(create_graph=True, retain_graph=True)
        #grads = tuple(inp.grad.clone() for inp in detached_inputs) 
        #print("here?2")
        #print(grads)
        #detached_inputs[0].grad.zero_()
        print("CHECKPOINT BACKWARD FINISHED")
        return (None, None, None) + grads
        #return (None, None, None) + grads
        #return (None, None) + grads + (None,)
        #return (None, None) + grads

def hesscheckpoint(function, *args, **kwargs):
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    row_idx = kwargs.pop('row_idx', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))
    return HessCheckpointFunction.apply(function, row_idx, preserve, *args)

def detach_variable(inputs):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)


def check_backward_validity(inputs):
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor)):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")

