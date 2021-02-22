import itertools
from math import sqrt, pi, exp
from math import exp, gamma
#from src.common import coordinate_distance
#from src.common import gaussian_product_coordinate
#from src.integrals import boys_function
#from src.integrals import boys_function_recursion
#from src.objects import PrimitiveBasis

class Basis:
    """Contracted Gaussian basis function.

    Attributes
    ----------
    primitive_gaussian_array : List[PrimitiveBasis]
    coordinates : Tuple[float, float, float]
    integral_exponents : Tuple[int, int, int]

    """
    def __init__(self, primitive_gaussian_array, coordinates, integral_exponents):
        self.primitive_gaussian_array = primitive_gaussian_array
        self.coordinates = coordinates
        self.integral_exponents = integral_exponents
        self.normalisation_memo = None

    @property
    def normalisation(self):
        """Calculates normalisation constant for the basis set and stores in self.normalisation once called.

        Returns
        -------
        self.normalisation_memo : float

        """
        if self.normalisation_memo is None:
            l, m, n = self.integral_exponents

            if len(self.primitive_gaussian_array) == 1:
                self.normalisation_memo = 1
            else:
                ans = 0.0
                for primitive_a, primitive_b in itertools.product(self.primitive_gaussian_array, repeat=2):
                    a_1 = primitive_a.exponent
                    a_2 = primitive_b.exponent
                    c_1 = primitive_a.contraction
                    c_2 = primitive_b.contraction
                    n_1 = primitive_a.normalisation
                    n_2 = primitive_b.normalisation

                    out1 = factorial2(2 * l - 1) * factorial2(2 * m - 1) * factorial2(2 * n - 1)
                    out2 = (pi / (a_1 + a_2)) ** (3 / 2)
                    out3 = (2 * (a_1 + a_2)) ** (l + m + n)
                    ans += (c_1 * c_2 * n_1 * n_2 * out1 * out2) / out3
                self.normalisation_memo = 1 / sqrt(ans)

        return self.normalisation_memo

    def value(self, x, y, z):
        """Returns the value at point x, y, z.

        Parameters
        ----------
        x : float
        y : float
        z : float

        Returns
        -------
        ans : float
        """
        ans = 0
        for primitive in self.primitive_gaussian_array:
            ans += primitive.value(x, y, z)
        return self.normalisation * ans


class PrimitiveBasis:
    """Primitive Gaussian basis function.

    Attributes
    ----------
    contraction : float
    exponent : float
    coordinates : Tuple[float, float, float]
    integral_exponents : Tuple[int, int, int]
    normalisation_memo : {None, float}
        Stores the normalisation constant once calculated.

    """
    def __init__(self, contraction, exponent, coordinates, integral_exponents):
        self.contraction = contraction
        self.exponent = exponent
        self.coordinates = coordinates
        self.integral_exponents = integral_exponents
        self.normalisation_memo = None

    @property
    def normalisation(self):
        """Calculates normalisation constant for the primitive gaussian and stores in self.normalisation once called.

        Returns
        -------
        self.normalisation_memo : float

        """
        if self.normalisation_memo is None:
            l, m, n = self.integral_exponents
            out1 = factorial2(2 * l - 1) * factorial2(2 * m - 1) * factorial2(2 * n - 1)
            out2 = (pi / (2 * self.exponent))**(3/2)
            out3 = (4 * self.exponent)**(l + m + n)
            self.normalisation_memo = 1 / sqrt((out1 * out2) / out3)
        return self.normalisation_memo

    def value(self, x, y, z):
        """Returns the value of the function at point x, y, z.

        Parameters
        ----------
        x : float
        y : float
        z : float

        Returns
        -------
        : float

        """
        r_x, r_y, r_z = self.coordinates
        l_x, l_y, l_z = self.integral_exponents
        return self.normalisation * self.contraction * (x - r_x)**l_x * (y - r_y)**l_y * (z - r_z)**l_z \
        * np.exp(- self.exponent * ((x - r_x)**2 + (y - r_y)**2 + (z - r_z)**2))


def boys_function(v, x):
    """Computes the boys function used for calculating the two electron and nuclear attraction integrals.

    Parameters
    ----------
    v : {int, float}
    x : {int, float}

    Returns
    -------
    ans : float

    Notes
    -----
    There are also no checks for if the series diverges and causes an infinite loop.

    References
    ----------
    [1] Handbook of Computational Chemistry pg. 280

    """
    # Approximation of the boys function for small x
    if x <= 25:
        i = 0
        ans = 0
        g_v = gamma(v + 0.5)
        while True:
            seq = (g_v / gamma(v + i + 1.5)) * x**i
            if seq < 1e-10:
                break
            ans += seq
            i += 1
        ans *= (1/2) * exp(-x)
        return ans

    # Approximation of the boys function for large x
    elif x > 25:
        i = 0
        ans = 0
        g_v = gamma(v + 0.5)
        while True:
            seq = (g_v / gamma(v - i + 1.5)) * x**(-i)
            if seq < 1e-10:
                break
            ans += seq
            i += 1
        ans *= (1/2) * exp(-x)
        ans = (g_v / (2*x**(v + 0.5))) - ans
        return ans


def boys_function_recursion(v, x, f_v):
    """Returns the answer to the boys function f_{v - 1}(x) using the answer for the boys function f_{v}(x).

    Parameters
    ----------
    v : {int, float}
    x : {int, float}
    f_v : float

    Returns
    -------
    : float

    References
    ----------
    [1] Handbook of Computational Chemistry pg. 280

    """
    return (exp(-x) + 2 * x * f_v) / (2 * v - 1)

def gaussian_product_coordinate(a, r_1, b, r_2):
    i = (a * r_1[0] + b * r_2[0]) / (a + b)
    j = (a * r_1[1] + b * r_2[1]) / (a + b)
    k = (a * r_1[2] + b * r_2[2]) / (a + b)
    return i, j, k

def coordinate_distance(r_1, r_2):
    return sqrt((r_1[0] - r_2[0])**2 + (r_1[1] - r_2[1])**2 + (r_1[2] - r_2[2])**2)

class ObaraSaika:

    def __init__(self):
        self.a_7 = 0
        self.r_7 = ()
        self.end_dict = {}

    def integrate(self, basis_i, basis_j, basis_k, basis_l):
        l_1 = basis_i.integral_exponents
        l_2 = basis_j.integral_exponents
        l_3 = basis_k.integral_exponents
        l_4 = basis_l.integral_exponents
        l_total = sum(l_1) + sum(l_2) + sum(l_3) + sum(l_4)

        r_1 = basis_i.coordinates
        r_2 = basis_j.coordinates
        r_3 = basis_k.coordinates
        r_4 = basis_l.coordinates

        primitives_i = basis_i.primitive_gaussian_array
        primitives_j = basis_j.primitive_gaussian_array
        primitives_k = basis_k.primitive_gaussian_array
        primitives_l = basis_l.primitive_gaussian_array

        #n_i = basis_i.normalisation
        #n_j = basis_j.normalisation
        #n_k = basis_k.normalisation
        #n_l = basis_l.normalisation
        #n = n_i * n_j * n_k * n_l

        ans = 0.0
        for g1, g2, g3, g4 in itertools.product(primitives_i, primitives_j, primitives_k, primitives_l):
            #c_1 = g1.contraction
            #c_2 = g2.contraction
            #c_3 = g3.contraction
            #c_4 = g4.contraction
            #n_1 = g1.normalisation
            #n_2 = g2.normalisation
            #n_3 = g3.normalisation
            #n_4 = g4.normalisation
            #contraction = c_1 * c_2 * c_3 * c_4 * n_1 * n_2 * n_3 * n_4 * n

            a_1 = g1.exponent
            a_2 = g2.exponent
            a_3 = g3.exponent
            a_4 = g4.exponent
            a_5 = a_1 + a_2
            a_6 = a_3 + a_4
            self.a_7 = (a_5 * a_6) / (a_5 + a_6)

            r_5 = gaussian_product_coordinate(a_1, r_1, a_2, r_2)
            r_6 = gaussian_product_coordinate(a_3, r_3, a_4, r_4)
            self.r_7 = gaussian_product_coordinate(a_5, r_5, a_6, r_6)

            r_12 = coordinate_distance(r_1, r_2)
            r_34 = coordinate_distance(r_3, r_4)
            r_56 = coordinate_distance(r_5, r_6)

            boys_x = (a_5 * a_6 * r_56**2) / (a_5 + a_6)
            boys_out1 = (2 * pi**(5/2)) / (a_5 * a_6 * sqrt(a_5 + a_6))
            boys_out2 = exp(((- a_1 * a_2 * r_12**2) / a_5) - ((a_3 * a_4 * r_34**2) / a_6))
            boys_out3 = boys_function(l_total, boys_x)
            self.end_dict = {l_total: boys_out1 * boys_out2 * boys_out3}

            m = l_total
            while m >= 1:
                boys_out3 = boys_function_recursion(m, boys_x, boys_out3)
                m -= 1
                self.end_dict[m] = boys_out1 * boys_out2 * boys_out3

            #ans += contraction * self.os_begin(0, g1, g2, g3, g4)
            ans += self.os_begin(0, g1, g2, g3, g4)

        return ans

    def os_begin(self, m, g1, g2, g3, g4):
        l_1 = g1.integral_exponents
        l_2 = g2.integral_exponents
        l_3 = g3.integral_exponents
        l_4 = g4.integral_exponents

        if l_1[0] > 0:
            return self.os_recursive(0, m, *self.os_gaussian_factory(0, g1, g2, g3, g4))
        elif l_1[1] > 0:
            return self.os_recursive(1, m, *self.os_gaussian_factory(1, g1, g2, g3, g4))
        elif l_1[2] > 0:
            return self.os_recursive(2, m, *self.os_gaussian_factory(2, g1, g2, g3, g4))
        elif l_2[0] > 0:
            return self.os_recursive(0, m, *self.os_gaussian_factory(0, g2, g1, g4, g3))
        elif l_2[1] > 0:
            return self.os_recursive(1, m, *self.os_gaussian_factory(1, g2, g1, g4, g3))
        elif l_2[2] > 0:
            return self.os_recursive(2, m, *self.os_gaussian_factory(2, g2, g1, g4, g3))
        elif l_3[0] > 0:
            return self.os_recursive(0, m, *self.os_gaussian_factory(0, g3, g4, g1, g2))
        elif l_3[1] > 0:
            return self.os_recursive(1, m, *self.os_gaussian_factory(1, g3, g4, g1, g2))
        elif l_3[2] > 0:
            return self.os_recursive(2, m, *self.os_gaussian_factory(2, g3, g4, g1, g2))
        elif l_4[0] > 0:
            return self.os_recursive(0, m, *self.os_gaussian_factory(0, g4, g3, g2, g1))
        elif l_4[1] > 0:
            return self.os_recursive(1, m, *self.os_gaussian_factory(1, g4, g3, g2, g1))
        elif l_4[2] > 0:
            return self.os_recursive(2, m, *self.os_gaussian_factory(2, g4, g3, g2, g1))
        else:
            return self.end_dict[m]

    def os_recursive(self, r, m, g1, g2, g3, g4, g5, g6, g7, g8):
        out1 = out2 = out3 = out4 = out5 = out6 = out7 = out8 = 0

        a_1 = g1.exponent
        a_2 = g2.exponent
        a_3 = g3.exponent
        a_4 = g4.exponent
        a_5 = a_1 + a_2
        a_6 = a_3 + a_4

        r_1 = g1.coordinates
        r_2 = g2.coordinates
        r_5 = gaussian_product_coordinate(a_1, r_1, a_2, r_2)

        if r_5[r] != r_1[r]:
            out1 = (r_5[r] - r_1[r]) * self.os_begin(m, g1, g2, g3, g4)
        if self.r_7[r] != r_5[r]:
            out2 = (self.r_7[r] - r_5[r]) * self.os_begin(m+1, g1, g2, g3, g4)
        if g5.integral_exponents[r] >= 0:
            out3 = self.os_int(g1.integral_exponents[r]) * (1 / (2 * a_5)) * self.os_begin(m, g5, g2, g3, g4)
            out4 = self.os_int(g1.integral_exponents[r]) * (self.a_7 / (2 * a_5 ** 2)) * self.os_begin(m+1, g5, g2, g3, g4)
        if g6.integral_exponents[r] >= 0:
            out5 = self.os_int(g2.integral_exponents[r]) * (1 / (2 * a_5)) * self.os_begin(m, g1, g6, g3, g4)
            out6 = self.os_int(g2.integral_exponents[r]) * (self.a_7 / (2 * a_5 ** 2)) * self.os_begin(m+1, g1, g6, g3, g4)
        if g7.integral_exponents[r] >= 0:
            out7 = self.os_int(g3.integral_exponents[r]) * (1 / (2 * (a_5 + a_6))) * self.os_begin(m+1, g1, g2, g7, g4)
        if g8.integral_exponents[r] >= 0:
            out8 = self.os_int(g4.integral_exponents[r]) * (1 / (2 * (a_5 + a_6))) * self.os_begin(m+1, g1, g2, g3, g8)

        #return out1 + out2 + out3 - out4 + out5 - out6 + out7 + out8
        result = out1 + out2 + out3 - out4 + out5 - out6 + out7 + out8
        #print(result)
        return result

    def os_int(self, i):
        if i == 0:
            return 1
        else:
            return i

    def os_gaussian_factory(self, r, g1, g2, g3, g4):
        d_1 = g1.contraction
        d_2 = g2.contraction
        d_3 = g3.contraction
        d_4 = g4.contraction

        a_1 = g1.exponent
        a_2 = g2.exponent
        a_3 = g3.exponent
        a_4 = g4.exponent

        r_1 = g1.coordinates
        r_2 = g2.coordinates
        r_3 = g3.coordinates
        r_4 = g4.coordinates

        l_1 = g1.integral_exponents
        l_2 = g2.integral_exponents
        l_3 = g3.integral_exponents
        l_4 = g4.integral_exponents

        if r == 0:
            g1x1 = PrimitiveBasis(d_1, a_1, r_1, (l_1[0] - 1, l_1[1], l_1[2]))
            g1x2 = PrimitiveBasis(d_1, a_1, r_1, (l_1[0] - 2, l_1[1], l_1[2]))
            g2x1 = PrimitiveBasis(d_2, a_2, r_2, (l_2[0] - 1, l_2[1], l_2[2]))
            g3x1 = PrimitiveBasis(d_3, a_3, r_3, (l_3[0] - 1, l_3[1], l_3[2]))
            g4x1 = PrimitiveBasis(d_4, a_4, r_4, (l_4[0] - 1, l_4[1], l_4[2]))
            return g1x1, g2, g3, g4, g1x2, g2x1, g3x1, g4x1
        elif r == 1:
            g1y1 = PrimitiveBasis(d_1, a_1, r_1, (l_1[0], l_1[1] - 1, l_1[2]))
            g1y2 = PrimitiveBasis(d_1, a_1, r_1, (l_1[0], l_1[1] - 2, l_1[2]))
            g2y1 = PrimitiveBasis(d_2, a_2, r_2, (l_2[0], l_2[1] - 1, l_2[2]))
            g3y1 = PrimitiveBasis(d_3, a_3, r_3, (l_3[0], l_3[1] - 1, l_3[2]))
            g4y1 = PrimitiveBasis(d_4, a_4, r_4, (l_4[0], l_4[1] - 1, l_4[2]))
            return g1y1, g2, g3, g4, g1y2, g2y1, g3y1, g4y1
        elif r == 2:
            g1z1 = PrimitiveBasis(d_1, a_1, r_1, (l_1[0], l_1[1], l_1[2] - 1))
            g1z2 = PrimitiveBasis(d_1, a_1, r_1, (l_1[0], l_1[1], l_1[2] - 2))
            g2z1 = PrimitiveBasis(d_2, a_2, r_2, (l_2[0], l_2[1], l_2[2] - 1))
            g3z1 = PrimitiveBasis(d_3, a_3, r_3, (l_3[0], l_3[1], l_3[2] - 1))
            g4z1 = PrimitiveBasis(d_4, a_4, r_4, (l_4[0], l_4[1], l_4[2] - 1))
            return g1z1, g2, g3, g4, g1z2, g2z1, g3z1, g4z1



#basis1 = Basis([PrimitiveBasis(1.0, 0.5,  (0.0,0.0, 0.9), (1,0,1))], (0.0,0.0, 0.9), (1,0,1))
#basis2 = Basis([PrimitiveBasis(1.0, 0.5,  (0.0,0.0,-0.9), (1,0,1))], (0.0,0.0,-0.9), (1,0,1))
#basis3 = Basis([PrimitiveBasis(1.0, 0.5,  (0.0,0.0, 0.9), (1,0,1))], (0.0,0.0, 0.9), (1,0,1))
#basis4 = Basis([PrimitiveBasis(1.0, 0.5,  (0.0,0.0,-0.9), (1,0,1))], (0.0,0.0,-0.9), (1,0,1))
##basis2 = Basis(1.0, 0.5,  (0.0,0.0,-0.9), (1,0,0))
##basis3 = Basis(1.0, 0.5,  (0.0,0.0, 0.9), (1,0,0))
##basis4 = Basis(1.0, 0.5,  (0.0,0.0,-0.9), (1,0,0))
#obj = ObaraSaika()
#result = obj.integrate(basis1, basis2, basis3, basis4)
#print(result)


