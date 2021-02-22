import jax.numpy as np
import jax
from jax.config import config; config.update("jax_enable_x64", True)

# Dummy hartree fock computation, couples 'basis' with 'geom' and computes dummy arrays for integrals
geom = np.array([0.500000000000,0.20000000000,-0.849220457955,0.500000000000,0.700000000000, 0.849220457955])
basis = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

test = np.arange(1000000)
test2 = np.arange(1000000)
for i in range(1000):
    test3 = np.dot(test, test2)

#x = jax.normal(jrkey, shape=(50000, 50000))
#x @ x

##@jax.jit
#def hartree_fock(geom, basis):
#    A = np.outer(geom*basis,geom*basis)
#    T = A * 2.5 + 2.75
#    V = A * 4.5 
#    G = np.einsum('ij,kl->ijkl',V,V)
#    
#    H = T + V
#    Enuc = 1.0
#    ndocc = 1
#    
#    D = np.zeros_like(H)
#    
#    for i in range(100):
#        J = np.einsum('pqrs,rs->pq', G, D)                    
#        K = np.einsum('prqs,rs->pq', G, D)                    
#        F = H + J * 2 - K                                        
#        E_scf = np.einsum('pq,pq->', F + H, D) + Enuc         
#        Fp = A.dot(F).dot(A)
#        eps, C2 = np.linalg.eigh(Fp)
#        C = A.dot(C2)                                  
#        Cocc = C[:, :ndocc]                                      
#        D = np.einsum('pi,qi->pq', Cocc, Cocc)                
#    # some funny business with returning multiple tyings... its like taking the grad of everything
#    #data = [eps, C]
#    return E_scf#, data
#
#grad_calculator = jax.grad(hartree_fock, argnums=0)
#hess_calculator = jax.hessian(hartree_fock, argnums=0)
##grad_calculator = jax.jit(jax.grad(hartree_fock, argnums=0))
##hess_calculator = jax.jit(jax.hessian(hartree_fock, argnums=0))
#
#E = hartree_fock(geom,basis)
#print(E)
#gradient = grad_calculator(geom, basis)
#print(gradient)
#hessian = hess_calculator(geom,basis)
#print(hessian)
