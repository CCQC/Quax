import psijax
import numpy as np
import jax.numpy as jnp
np.set_printoptions(linewidth=800, precision=10)

import psi4
import time
psi4.core.be_quiet()

molecule = psi4.geometry("""
                         0 1
                         C            0.000000000000     0.000000000000    -1.141817200432
                         O            0.000000000000     0.000000000000     1.138747061525
                         H           -0.000000000000     1.762942010951    -2.238660220220
                         H            0.000000000000    -1.762942010951    -2.238660220220
                         symmetry c1
                         units bohr 
                         """)

basis_name = 'cc-pvdz'
psi4.set_memory(int(5e9))
psi4.set_options({'basis': basis_name, 'scf_type': 'pk', 'mp2_type':'conv', 'e_convergence': 1e-10, 'diis': True, 'd_convergence':1e-10, 'puream': 0, 'points':5, 'fd_project':False})

# CFOUR analytic Hessians at above geometry. In Hartree/Bohr^2
cfour_scf =np.array([[ 0.225303891 ,  0.          ,  0.          , -0.0866797003,  0.          ,  0.          , -0.0693120954,  0.          ,  0.          , -0.0693120954,  0.          ,  0.          ],
                    [ 0.          ,  0.6990439966,  0.          ,  0.          , -0.1362144166,  0.          ,  0.          , -0.28141479  ,  0.0999237678,  0.          , -0.28141479  , -0.0999237678],
                    [ 0.          ,  0.          ,  1.0676110734,  0.          ,  0.          , -0.834293342 ,  0.          ,  0.0982448668, -0.1166588658,  0.          , -0.0982448668, -0.1166588658],
                    [-0.0866797003,  0.          ,  0.          ,  0.0418173275,  0.          ,  0.          ,  0.0224311864,  0.          ,  0.          ,  0.0224311864,  0.          ,  0.          ],
                    [ 0.          , -0.1362144166,  0.          ,  0.          ,  0.0996630107,  0.          ,  0.          ,  0.0182757029,  0.0400003581,  0.          ,  0.0182757029, -0.0400003581],
                    [ 0.          ,  0.          , -0.834293342 ,  0.          ,  0.          ,  0.9190591204,  0.          ,  0.0141339125, -0.0423828892,  0.          , -0.0141339125, -0.0423828892],
                    [-0.0693120954,  0.          ,  0.          ,  0.0224311864,  0.          ,  0.          ,  0.0233421904,  0.          ,  0.          ,  0.0235387186,  0.          ,  0.          ],
                    [ 0.          , -0.28141479  ,  0.0982448668,  0.          ,  0.0182757029,  0.0141339125,  0.          ,  0.2820343957, -0.1261514526,  0.          , -0.0188953086,  0.0137726734],
                    [ 0.          ,  0.0999237678, -0.1166588658,  0.          ,  0.0400003581, -0.0423828892,  0.          , -0.1261514526,  0.1488231344,  0.          , -0.0137726734,  0.0102186207],
                    [-0.0693120954,  0.          ,  0.          ,  0.0224311864,  0.          ,  0.          ,  0.0235387186,  0.          ,  0.          ,  0.0233421904,  0.          ,  0.          ],
                    [ 0.          , -0.28141479  , -0.0982448668,  0.          ,  0.0182757029, -0.0141339125,  0.          , -0.0188953086, -0.0137726734,  0.          ,  0.2820343957,  0.1261514526],
                    [ 0.          , -0.0999237678, -0.1166588658,  0.          , -0.0400003581, -0.0423828892,  0.          ,  0.0137726734,  0.0102186207,  0.          ,  0.1261514526,  0.1488231344]])
 
cfour_mp2 = np.array([[ 0.1668531531,  0.          ,  0.          , -0.0539715894,  0.          ,  0.          , -0.0564407818,  0.          ,  0.          , -0.0564407818,  0.          ,  0.          ],
                      [ 0.          ,  0.6533008815,  0.          ,  0.          , -0.1059579298,  0.          ,  0.          , -0.2736714758,  0.1015285098,  0.          , -0.2736714758, -0.1015285098],
                      [ 0.          ,  0.          ,  1.0207601598,  0.          ,  0.          , -0.8026951416,  0.          ,  0.0996884027, -0.1090325091,  0.          , -0.0996884027, -0.1090325091],
                      [-0.0539715894,  0.          ,  0.          ,  0.0132079899,  0.          ,  0.          ,  0.0203817997,  0.          ,  0.          ,  0.0203817997,  0.          ,  0.          ],
                      [ 0.          , -0.1059579298,  0.          ,  0.          ,  0.0700522302,  0.          ,  0.          ,  0.0179528498,  0.0382784276,  0.          ,  0.0179528498, -0.0382784276],
                      [ 0.          ,  0.          , -0.8026951416,  0.          ,  0.          ,  0.8873557061,  0.          ,  0.0140334736, -0.0423302822,  0.          , -0.0140334736, -0.0423302822],
                      [-0.0564407818,  0.          ,  0.          ,  0.0203817997,  0.          ,  0.          ,  0.0158423837,  0.          ,  0.          ,  0.0202165984,  0.          ,  0.          ],
                      [ 0.          , -0.2736714758,  0.0996884027,  0.          ,  0.0179528498,  0.0140334736,  0.          ,  0.2748482995, -0.1267644069,  0.          , -0.0191296735,  0.0130425305],
                      [ 0.          ,  0.1015285098, -0.1090325091,  0.          ,  0.0382784276, -0.0423302822,  0.          , -0.1267644069,  0.1417445982,  0.          , -0.0130425305,  0.0096181931],
                      [-0.0564407818,  0.          ,  0.          ,  0.0203817997,  0.          ,  0.          ,  0.0202165984,  0.          ,  0.          ,  0.0158423837,  0.          ,  0.          ],
                      [ 0.          , -0.2736714758, -0.0996884027,  0.          ,  0.0179528498, -0.0140334736,  0.          , -0.0191296735, -0.0130425305,  0.          ,  0.2748482995,  0.1267644069],
                      [ 0.          , -0.1015285098, -0.1090325091,  0.          , -0.0382784276, -0.0423302822,  0.          ,  0.0130425305,  0.0096181931,  0.          ,  0.1267644069,  0.1417445982]])

cfour_ccsdt = np.array([[ 0.1594242119,  0.          ,  0.          , -0.0520535497,  0.          ,  0.          , -0.0536853311,  0.          ,  0.          , -0.0536853311,  0.          ,  0.          ],
                        [ 0.          ,  0.6457162756,  0.          ,  0.          , -0.1047028771,  0.          ,  0.          , -0.2705066993,  0.1008450168,  0.          , -0.2705066993, -0.1008450168],
                        [ 0.          ,  0.          ,  1.0232246423,  0.          ,  0.          , -0.8064919165,  0.          ,  0.0988325449, -0.1083663629,  0.          , -0.0988325449, -0.1083663629],
                        [-0.0520535497,  0.          ,  0.          ,  0.0125421792,  0.          ,  0.          ,  0.0197556852,  0.          ,  0.          ,  0.0197556852,  0.          ,  0.          ],
                        [ 0.          , -0.1047028771,  0.          ,  0.          ,  0.0694465831,  0.          ,  0.          ,  0.017628147 ,  0.0381298137,  0.          ,  0.017628147 , -0.0381298137],
                        [ 0.          ,  0.          , -0.8064919165,  0.          ,  0.          ,  0.8903326016,  0.          ,  0.0150587622, -0.0419203426,  0.          , -0.0150587622, -0.0419203426],
                        [-0.0536853311,  0.          ,  0.          ,  0.0197556852,  0.          ,  0.          ,  0.0142674909,  0.          ,  0.          ,  0.019662155 ,  0.          ,  0.          ],
                        [ 0.          , -0.2705066993,  0.0988325449,  0.          ,  0.017628147 ,  0.0150587622,  0.          ,  0.2720653616, -0.1264330688,  0.          , -0.0191868093,  0.0125417618],
                        [ 0.          ,  0.1008450168, -0.1083663629,  0.          ,  0.0381298137, -0.0419203426,  0.          , -0.1264330688,  0.1403413518,  0.          , -0.0125417618,  0.0099453536],
                        [-0.0536853311,  0.          ,  0.          ,  0.0197556852,  0.          ,  0.          ,  0.019662155 ,  0.          ,  0.          ,  0.0142674909,  0.          ,  0.          ],
                        [ 0.          , -0.2705066993, -0.0988325449,  0.          ,  0.017628147 , -0.0150587622,  0.          , -0.0191868093, -0.0125417618,  0.          ,  0.2720653616,  0.1264330688],
                        [ 0.          , -0.1008450168, -0.1083663629,  0.          , -0.0381298137, -0.0419203426,  0.          ,  0.0125417618,  0.0099453536,  0.          ,  0.1264330688,  0.1403413518]])

print("Testing Energies")
for method in ['scf', 'mp2', 'ccsd(t)']:
    psi_e = psi4.energy(method + '/' + basis_name)
    psijax_e = psijax.core.energy(molecule, basis_name, method)
    print(psijax_e)
    print("\n{} energies match: ".format(method), np.allclose(psi_e, psijax_e, rtol=0.0, atol=1e-6), '\n')
    print('Error:', np.abs(psijax_e - psi_e))
print("\n")
print("\n")

print("Testing Hessians")
for method in ['scf', 'mp2','ccsd(t)']:
    if method == 'scf': cfour_deriv = cfour_scf
    if method == 'mp2': cfour_deriv = cfour_mp2
    if method == 'ccsd(t)': cfour_deriv = cfour_ccsdt
    psijax_deriv = np.asarray(psijax.core.derivative(molecule, basis_name, method, order=2))
    print("\n{} hessians match: ".format(method),np.allclose(psijax_deriv, cfour_deriv,rtol=0.0,atol=1e-4), '\n')
    print('Error:', np.abs(psijax_deriv - cfour_deriv))
print("\n")
print("\n")

