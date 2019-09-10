'''Functions for differentiating arbitrary electronic energies wrt geometry'''
import torch
import numpy as np
from itertools import permutations

def expand_tensor(unique_values, rank, nparam):
    """
    Expands derivative tensor to its full form given a set of unique values.
    
    Parameters
    ----------
    unique_values : list 
        A list of single-valued torch.tensor()'s
    rank : int
        The rank of the tensor. Hessian = 2, Third derivatives = 3, Fourth = 4 ...
    nparam : int
        The number of geometry parameters along a dimension of the derivative tensor. 
        Typically 3N or 3N-6 for cartesians and internals, respectfully.
    """
    indices = torch.tensor([i for i in range(nparam)], dtype=torch.int)
    perm = [i for i in range(rank)]
    # Indices of unique tensor elements for this number of geometry parameters and rank 
    t_idx = torch.combinations(indices,rank,with_replacement=True)
    # Initialize tensor of proper size and assign unique values to all proper positions
    tens = torch.zeros((nparam,) * rank, dtype=torch.float64, requires_grad=False)
    with torch.no_grad():
        for i,idx in enumerate(t_idx):          # for each unique value
            for p in list(permutations(perm)):  # assign position based on permutational symmetry
                if rank == 2:
                    tens[idx[p[0]], idx[p[1]]] = unique_values[i]
                if rank == 3:
                    tens[idx[p[0]], idx[p[1]], idx[p[2]]] = unique_values[i]
                if rank == 4:
                    tens[idx[p[0]], idx[p[1]], idx[p[2]], idx[p[3]]] = unique_values[i]
                if rank == 5:
                    tens[idx[p[0]], idx[p[1]], idx[p[2]], idx[p[3]], idx[p[4]]] = unique_values[i]
                if rank == 6:
                    tens[idx[p[0]], idx[p[1]], idx[p[2]], idx[p[3]], idx[p[4]],  idx[p[5]]] = unique_values[i]
    return tens

def differentiate(E, geom, order=2):
    """
    Computes all unique derivatives up to order'th order, then expands the unique elements
    into full, redundant, symmetric derivative tensors. 
    Returns the derivative tensors as an unpackable tuple.  
    This function is not readable, but is efficient. 
    See the less efficient but more intuitive slow_differentiate_nn() method below

    Parameters
    ----------
    E : torch.tensor
        A single valued tensor containing the energy, produced from a continuous
        computation from values in 'geom' 
    geom : list 
        A list of single-value torch.tensors, requires_grad=True. Must be 
        connected to 'E' through a series of PyTorch operations 

    Returns
    -------
    tuple of tensors gradient, hessian through 'order' up to sextic derivatives

    Examples:
    gradient, hessian = differentiate_nn(E,geom,order=2)
    gradient, hessian, cubic = differentiate_nn(E,geom,order=3)
    gradient, hessian, cubic, quartic = differentiate_nn(E,geom,order=4)
    gradient, hessian, cubic, quartic, quintic, sextic = differentiate_nn(E,geom,order=6)
    """
    if not isinstance(geom, list):
        raise Exception("Must supply geometry as a list of torch.tensors with requires_grad=True.")
    # number of geometry parameters, indices for each geometry
    nparam = len(geom)
    indices = torch.tensor([i for i in range(nparam)], dtype=torch.int)

    # Find unique element indices for each tensor
    h_idx = torch.combinations(indices,2,with_replacement=True)
    c_idx = torch.combinations(indices,3,with_replacement=True)
    q_idx = torch.combinations(indices,4,with_replacement=True)
    f_idx = torch.combinations(indices,5,with_replacement=True)
    s_idx = torch.combinations(indices,6,with_replacement=True)

    # Find derivative partitions.  
    h_i = torch.nonzero(torch.eq(h_idx[:,-1], h_idx[:,-2])).flatten() 
    c_i = torch.nonzero(torch.eq(c_idx[:,-1], c_idx[:,-2])).flatten()
    q_i = torch.nonzero(torch.eq(q_idx[:,-1], q_idx[:,-2])).flatten()
    f_i = torch.nonzero(torch.eq(f_idx[:,-1], f_idx[:,-2])).flatten()
    s_i = torch.nonzero(torch.eq(s_idx[:,-1], s_idx[:,-2])).flatten()

    # compute the gradient and all unique higher order derivatives
    h1, c1, q1, f1, s1 = [], [], [], [], []
    g = torch.autograd.grad(E, geom, create_graph=True)
    tmp = np.array(geom)

    for i, idx in enumerate(h_i):
        if i+1 < h_i.size()[0]:
            slice1 = slice(idx, h_i[i+1])
        else:
            slice1 = slice(idx, idx+1)
        slice2 = h_idx[:,-1][slice1]
        h = torch.autograd.grad(g[i], tmp[slice2], create_graph=True)
        h1.extend(h)

    if order > 2:
        for i, idx in enumerate(c_i):
            if i+1 < c_i.size()[0]:
                slice1 = slice(idx, c_i[i+1])
            else:
                slice1 = slice(idx, idx+1)
            slice2 = c_idx[:,-1][slice1]
            c = torch.autograd.grad(h1[i], tmp[slice2], create_graph=True)
            c1.extend(c)

    if order > 3:
        for i, idx in enumerate(q_i):
            if i+1 < q_i.size()[0]:
                slice1 = slice(idx, q_i[i+1])
            else:
                slice1 = slice(idx, idx+1)
            slice2 = q_idx[:,-1][slice1]
            q = torch.autograd.grad(c1[i], tmp[slice2], create_graph=True)
            q1.extend(q)

    if order > 4:
        for i, idx in enumerate(f_i):
            if i+1 < f_i.size()[0]:
                slice1 = slice(idx, f_i[i+1])
            else:
                slice1 = slice(idx, idx+1)
            slice2 = f_idx[:,-1][slice1]
            f = torch.autograd.grad(q1[i], tmp[slice2], create_graph=True)
            f1.extend(f)

    if order > 5:
        for i, idx in enumerate(s_i):
            if i+1 < s_i.size()[0]:
                slice1 = slice(idx, s_i[i+1])
            else:
                slice1 = slice(idx, idx+1)
            slice2 = s_idx[:,-1][slice1]
            s = torch.autograd.grad(f1[i], tmp[slice2], create_graph=True)
            s1.extend(s)

    # Build up full tensors from unique derivatives. 
    # NOTE likely temporary. Would be better to not use full tensor when deriving FCs 
    # indices of unique tensor elements through 6th order for this geometry dimension
    g = torch.stack(g)
    hess = expand_tensor(h1, 2, nparam)
    if order > 2:
        cubic = expand_tensor(c1, 3, nparam)
    else:
        return g, hess
    if order > 3:
        quartic = expand_tensor(q1, 4, nparam)
    else:
        return g, hess, cubic
    if order > 4:
        quintic = expand_tensor(f1, 5, nparam)
    else:
        return g, hess, cubic, quartic
    if order > 5:
        sextic = expand_tensor(s1, 6, nparam)
    else:
        return g, hess, cubic, quartic, quintic
    if order > 6:
        raise Exception("Derivatives beyond 6th order not implemented. Why do you even want this?")
    else:
        return g, hess, cubic, quartic, quintic, sextic

def jacobian(outputs, inputs, create_graph=True):
    """Computes the jacobian of (multidimensional) outputs with respect to inputs
    :param outputs: tensor for the output of some function
    :param inputs: tensor for the input of some function (probably a vector)
    :param create_graph: set True for the resulting jacobian to be differentible
    :returns: a tensor of size (outputs.size() + inputs.size()) containing the
        jacobian of outputs with respect to inputs
    """
    jac = outputs.new_zeros(outputs.size() + inputs.size()
                            ).view((-1,) + inputs.size())
    #for i, out in enumerate(outputs.view(-1)):
    for i, out in enumerate(outputs.reshape(-1)):
        col_i = torch.autograd.grad(out, inputs, retain_graph=True,
                              create_graph=create_graph, allow_unused=True)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            jac[i] = col_i

    if create_graph:
        jac.requires_grad_()

    return jac.view(outputs.size() + inputs.size())


def slowest_differentiate_nn(E, geom, order=4):
    """
    Takes a geometry, sends it through the NN with the transform() method.
    Returns derivative tensor of a neural network for a particular geometry.
    If order=3 it will return an unpackable tuple of the hessian and cubic derivatives 
    If order=4 it will return an unpackable tuple of the hessian, cubic, and quartic derivatives
    If order=5 it will return the hessian, cubic, quartic, and quintic derivatives
    If order=6 it will return the hessian, cubic, quartic, quintic, and sextic derivatives.

    Parameters
    ----------
    E : torch.tensor containing a scalar
        Derivatives will be taken of this quantity. It is the value of energy returned from NN, 
        connected through a series of pytorch computations to 'geometry' argument.
    geom : 1d torch.tensor() 
        A 1d tensor of geometry parameters, requires_grad=True, which were used to compute energy argument 'E'
    order : int
        Highest order of derivative to compute

    Returns
    -------
    A tuple of derivative tensors up through order'th derivatives

    WARNING: No symmetry is implemented, so the total number of derivative evaluations
    for nth order with r geometry parameters is:  
    (Sum over i=1 to n) of r^i
    """
    # Compute derivatives. Build up higher order tensors one dimension at a time.
    gradient = torch.autograd.grad(E, geom, create_graph=True)[0]
    h1, c1, q1, f1, s1 = [], [], [], [], []
    for d1 in gradient:
        h = torch.autograd.grad(d1, geom, create_graph=True)[0]
        h1.append(h)
        c2, q2, f2, s2 = [], [], [], []
        for d2 in h:
            c = torch.autograd.grad(d2, geom, create_graph=True)[0]
            c2.append(c)
            if order > 3:
                q3, f3, s3 = [], [], []
                for d3 in c:
                    q = torch.autograd.grad(d3, geom, create_graph=True)[0]
                    q3.append(q)
                    if order > 4:
                        f4, s4 = [], []
                        for d4 in q:
                            f = torch.autograd.grad(d4, geom, create_graph=True)[0]
                            f4.append(f)
                            if order > 5:
                                s5 = []
                                for d5 in f:
                                    s = torch.autograd.grad(d5, geom, create_graph=True)[0]
                                    s5.append(s)
                                s4.append(torch.stack(s5))
                            else:
                                continue
                        f3.append(torch.stack(f4))
                        if order > 5: s3.append(torch.stack(s4))
                    else:
                        continue
                if order > 3: q2.append(torch.stack(q3))
                if order > 4: f2.append(torch.stack(f3))
                if order > 5: s2.append(torch.stack(s3))
            else:
                continue
        c1.append(torch.stack(c2))
        if order > 3: q1.append(torch.stack(q2))
        if order > 4: f1.append(torch.stack(f2))
        if order > 5: s1.append(torch.stack(s2))

    hessian = torch.stack(h1)
    cubic = torch.stack(c1)
    if order == 3:
        return hessian, cubic
    elif order == 4:
        quartic = torch.stack(q1)
        return hessian, cubic, quartic
    elif order == 5:
        quartic = torch.stack(q1)
        quintic = torch.stack(f1)
        return hessian, cubic, quartic, quintic
    elif order == 6:
        quartic = torch.stack(q1)
        quintic = torch.stack(f1)
        sextic = torch.stack(s1)
        return hessian, cubic, quartic, quintic, sextic



