import psijax
import psi4
import numpy as np
import fdm
import jax

molecule = psi4.geometry("""
0 1
O   -0.000007070942     0.125146536460     0.000000000000
H   -1.424097055410    -0.993053750648     0.000000000000
H    1.424209276385    -0.993112599269     0.000000000000
units bohr
""")

g = np.array([[-0.000007070942, 0.125146536460,0.000000000000],
              [-1.424097055410,-0.993053750648,0.000000000000], 
              [ 1.424209276385,-0.993112599269,0.000000000000]])


method = 'scf'
basis_name = 'sto-3g'

def energy(geom):
    molecule.set_geometry(psi4.core.Matrix.from_array(geom))
    energy = psijax.core.energy(molecule, basis_name, method)
    return energy

findif_gradient = fdm.jacobian(energy)

# To save time, check only one element

def gradient(geom):
    molecule.set_geometry(psi4.core.Matrix.from_array(geom))
    gradient = psijax.core.derivative(molecule, basis_name, method, order=1)
    return gradient 

findif_hessian = fdm.jacobian(gradient)

def hessian(geom):
    # Set molecule geometry to input geometry so findif shift is accounted for
    molecule.set_geometry(psi4.core.Matrix.from_array(geom))
    hessian = psijax.core.derivative(molecule, basis_name, method, order=2)
    return hessian

findif_cubic = fdm.jacobian(hessian)

def cubic(geom):
    # Set molecule geometry to input geometry so findif shift is accounted for
    molecule.set_geometry(psi4.core.Matrix.from_array(geom))
    cubic = psijax.core.derivative(molecule, basis_name, method, order=3)
    return cubic

#findif_cubic = fdm.jacobian(hessian)
#estimated_cubic = findif_cubic(g)

#estimated_gradient = findif_gradient(g)
#print(gradient(g))
#print(estimated_gradient)

estimated_hessian = findif_hessian(g)
print(hessian(g))
print(estimated_hessian)

print(estimated_cubic = findif_cubic(g))
print(cubic(g)
