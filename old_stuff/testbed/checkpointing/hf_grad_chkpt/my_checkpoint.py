import torch

class HessCheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
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

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        # Compute gradient
        torch.autograd.backward(outputs, args, create_graph=True)

        # Compute hessian 'row'
        gradient = detached_inputs[0].grad.clone().flatten()
        detached_inputs[0].grad.zero_()
        # Hessian of 0th,1st, --->2nd<--- gradient
        gradient[2].backward(create_graph=True)
        grads = (detached_inputs[0].grad,)
        return (None, None) + grads
        
        #for g in gradient:
        #    detached_inputs[0].grad.zero_()
        #    g.backward(create_graph=True)
        #    print(detached_inputs[0].grad)

        ## Save gradients
        #detached_inputs_grads = []
        #for inp in detached_inputs:
        #    if isinstance(inp, torch.Tensor):
        #        if inp.grad is not None:
        ##            print("1!",inp.grad.size())
        #            inp.grad.zero_()
        #            gradient = inp.grad.clone() # adjust here TODO, taking 0th grad and doing 2nd deriv
        #            #gradient = inp.grad.clone().flatten() # adjust here TODO, taking 0th grad and doing 2nd deriv
        #            print(gradient)
        #    
        #            inp.grad.zero_()
        #            detached_inputs_grads.append(gradient)
        #        #else:
        #        #    detached_inputs_grads.append(None)
        #    #else:
        #    #    detached_inputs_grads.append(None)
        ##TODO NEED ELSIF SO NUMBER OF GRADS IS CONSISTENT
        #detached_inputs_grads = tuple(detached_inputs_grads)
        #print('detached_inputs')
        #print(detached_inputs)
        #print('detached_inputs_grad')
        #print(detached_inputs_grads)

        ### SECOND BACKWARD CALL
        #torch.autograd.backward(detached_inputs_grads, detached_inputs, create_graph=True)
        #print(detached_inputs[0].grad)
        #torch.autograd.backward(detached_inputs_grads, detached_inputs)
        #print('here!')
        ##print(detached_inputs_grads)
        #print(args)
        ##print(len(args))
        ##print(len(detached_inputs_grads))
        #torch.autograd.backward(detached_inputs_grads, args)
                
        #grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
        #              for inp in detached_inputs)
        #print(grads)


def hesscheckpoint(function, *args, **kwargs):
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    return HessCheckpointFunction.apply(function, preserve, *args)

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

