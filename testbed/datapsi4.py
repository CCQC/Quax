""" Basis set 2 (basis2) psi4 data"""
import numpy as np 
np.set_printoptions(precision=12)

hfg = np.array([[ 0.000000000000,    0.000000000000,   -0.058866931551],
                [ 0.000000000000,    0.000000000000,    0.058866931551]])

nuc = np.array([[ 0.000000000000,    0.000000000000,    0.346656312432], 
                [ 0.000000000000,    0.000000000000,   -0.346656312432]])

cor = np.array([[ 0.000000000000,    0.000000000000,   -0.903294111535], 
                [ 0.000000000000,    0.000000000000,    0.903294111535]])

lag = np.array([[ 0.000000000000,    0.000000000000,    0.195249890463],
                [ 0.000000000000,    0.000000000000,   -0.195249890463]])

two = np.array([[ 0.000000000000,    0.000000000000,    0.308497716106],
                [ 0.000000000000,    0.000000000000,   -0.308497716106]])

tot = np.array([[ 0.000000000000,    0.000000000000,   -0.052890192534],
                [ 0.000000000000,    0.000000000000,    0.052890192534]])

mp2_correlation = np.array([[ 0.000000000000,  0.000000000000,  0.005976739020],
                            [ 0.000000000000,  0.000000000000, -0.005976739020]])


print("Hartree-Fock Gradient")
print(hfg)
print("MP2 Gradient")
print(tot)
print("MP2 Correlation Energy Gradient")
print(mp2_correlation)
print("One electron gradient")
print(cor)
print("Two electron gradient")
print(two)
print("One+Two electron gradient")
print(cor+two)
print("Lagrangian Gradient")
print(lag)



# MP2 correlation energy component of gradient
#tensor([[ 0.000000000000,  0.000000000000,  0.005976739020]
#       [ 0.000000000000,  0.000000000000, -0.005976739020]]

