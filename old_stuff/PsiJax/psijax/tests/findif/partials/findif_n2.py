import psijax
import psi4
import numpy as np
import fdm
import jax

molecule = psi4.geometry("""
                         0 1
                         N 0.000000000000     0.000000000000    -1.056883600544
                         N 0.000000000000     0.000000000000     1.056883600544
                         symmetry c1
                         units bohr
                         """)

g = np.array([[0.000000000000,     0.000000000000,    -1.056883600544],
              [0.000000000000,     0.000000000000,     1.056883600544]])

method = 'scf'
basis_name = 'cc-pvdz'
print("System:", method, basis_name)

# Only change O2 y coordinate to reduce dimensionality
# Compute gradient d/dyO2, d^2/dyO2^2,...

def energy(o2y):
    geom2 = 1 * g
    geom2[0,2] = o2y
    molecule.set_geometry(psi4.core.Matrix.from_array(geom2))
    energy = psijax.core.energy(molecule, basis_name, method)
    return energy

findif_gradient = fdm.jacobian(energy)

def gradient(o2y):
    geom2 = 1 * g
    geom2[0,2] = o2y
    molecule.set_geometry(psi4.core.Matrix.from_array(geom2))
    gradient = psijax.core.partial_derivative(molecule, basis_name, method, order=1, address=(2,))
    return gradient

findif_hessian = fdm.jacobian(gradient)

def hessian(o2y):
    geom2 = 1 * g
    geom2[0,2] = o2y
    molecule.set_geometry(psi4.core.Matrix.from_array(geom2))
    hessian = psijax.core.partial_derivative(molecule, basis_name, method, order=2, address=(2,2))
    return hessian

findif_cubic = fdm.jacobian(hessian)

def cubic(o2y):
    geom2 = 1 * g
    geom2[0,2] = o2y
    molecule.set_geometry(psi4.core.Matrix.from_array(geom2))
    cubic = psijax.core.partial_derivative(molecule, basis_name, method, order=3, address=(2,2,2))
    return cubic 

findif_quartic = fdm.jacobian(cubic)

def quartic(o2y):
    geom2 = 1 * g
    geom2[0,2] = o2y
    molecule.set_geometry(psi4.core.Matrix.from_array(geom2))
    quartic = psijax.core.partial_derivative(molecule, basis_name, method, order=4, address=(2,2,2,2))
    return quartic 

findif_quintic = fdm.jacobian(quartic)

def quintic(o2y):
    geom2 = 1 * g
    geom2[0,2] = o2y
    molecule.set_geometry(psi4.core.Matrix.from_array(geom2))
    quintic= psijax.core.partial_derivative(molecule, basis_name, method, order=5, address=(2,2,2,2,2))
    return quintic

findif_sextic = fdm.jacobian(quintic)

def sextic(o2y):
    geom2 = 1 * g
    geom2[0,2] = o2y
    molecule.set_geometry(psi4.core.Matrix.from_array(geom2))
    sextic = psijax.core.partial_derivative(molecule, basis_name, method, order=6, address=(2,2,2,2,2,2))
    return sextic 


estimated_gradient = findif_gradient(g[0,2])
print("Gradient Results:")
print(gradient(g[0,2]))
print(estimated_gradient)
print("\n")

estimated_hessian = findif_hessian(g[0,2])
print("Hessian Results:")
print(hessian(g[0,2]))
print(estimated_hessian)
print("\n")

estimated_cubic= findif_cubic(g[0,2])
print("Cubic Results:")
print(cubic(g[0,2]))
print(estimated_cubic)
print("\n")

estimated_quartic = findif_quartic(g[0,2])
print("Quartic Results:")
print(quartic(g[0,2]))
print(estimated_quartic)
print("\n")

estimated_quintic = findif_quintic(g[0,2])
print("Quintic Results:")
print(quintic(g[0,2]))
print(estimated_quintic)
print("\n")


estimated_sextic = findif_sextic(g[0,2])
print("Sextic Results:")
print(sextic(g[0,2]))
print(estimated_sextic)
print("\n")






