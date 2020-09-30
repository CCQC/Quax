import jax 
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)
config.enable_omnistaging()
import jax.numpy as np
from jax.experimental import loops
from basis_utils import flatten_basis_data, get_nbf
from integrals_utils import gaussian_product, boys, binomial_prefactor, cartesian_product, am_leading_indices, angular_momentum_combinations, fact_ratio2, neg_one_pow, binomials

def old_binomial_prefactor(k, l1, l2, PAx, PBx):
    q = jax.lax.max(-k, k-2*l2)
    q_final = jax.lax.min(k, 2*l1-k)
    with loops.Scope() as L:
      L.total = 0.
      L.q = q
      for _ in L.while_range(lambda: L.q <= q_final):
        i = (k+L.q)//2
        j = (k-L.q)//2
        L.total += PAx**(l1-i) * PBx**(l2-j) * binomials[l1,i] * binomials[l2,j]
        L.q += 2
    return L.total

def B_array(l1,l2,l3,l4,pa,pb,qc,qd,qp,g1,g2,oodelta,B_vals):
    with loops.Scope() as s:
      s.B = B_vals
      s.i2 = 0
      s.r1 = 0
      s.r2 = 0
      s.u = 0 
      s.i1 = l1 + l2  
      for _ in s.while_range(lambda: s.i1 > -1):
        Bterm = old_binomial_prefactor(s.i1,l1,l2,pa,pb) 
        tmp = s.i1
        s.r1 = s.i1 // 2
        for _ in s.while_range(lambda: s.r1 > -1):
          Bterm *= fact_ratio2[s.i1,s.r1]
          Bterm *= (4*g1)**(s.r1-s.i1)
          tmp -= 2 * s.r1
          s.i2 = l3 + l4 
          for _ in s.while_range(lambda: s.i2 > -1):
            Bterm *= neg_one_pow[s.i2]
            Bterm *= old_binomial_prefactor(s.i2,l3,l4,qc,qd) 
            tmp += s.i2
            s.r2 = s.i2 // 2
            for _ in s.while_range(lambda: s.r2 > -1):
              Bterm *= fact_ratio2[s.i2,s.r2]
              Bterm *= (4*g2)**(s.r2-s.i2)
              tmp -= 2 * s.r2
              s.u = tmp // 2
              for _ in s.while_range(lambda: s.u > -1):
                I = tmp - s.u 
                Bterm *= neg_one_pow[s.u] 
                Bterm *= fact_ratio2[tmp,s.u]
                Bterm *= qp**(tmp - 2 * s.u)
                Bterm *= oodelta**I 
                s.B = jax.ops.index_add(s.B, I, Bterm)
                s.u -= 1
              s.r2 -= 1
            s.i2 -= 1
          s.r1 -= 1
        s.i1 -= 1
      return s.B

#@partial(jax.custom_jvp, nondiff_argnums=(4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20))

#@partial(jax.custom_jvp, nondiff_argnums=(12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28))
@jax.jit
def primitive_tei(ax,ay,az,bx,by,bz,cx,cy,cz,dx,dy,dz,la,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef): 
    """
    """
    A = np.array([ax,ay,az])
    B = np.array([bx,by,bz])
    C = np.array([cx,cy,cz])
    D = np.array([dx,dy,dz])
    max_am=10 # temp, have to make foolproof for higher derivatives
    B_vals = np.zeros(4*max_am+1)
    rab2 = np.dot(A-B,A-B)
    rcd2 = np.dot(C-D,C-D)
    P = gaussian_product(aa,A,bb,B)
    Q = gaussian_product(cc,C,dd,D)
    PA = P-A
    PB = P-B
    QC = Q-C
    QD = Q-D
    QP = Q-P
    rpq2 = np.dot(P-Q,P-Q)
    gamma1 = aa + bb
    gamma2 = cc + dd
    delta = 0.25*(1/gamma1+1/gamma2)
    oodelta = 1/delta

    Bx = B_array(la,lb,lc,ld,PA[0],PB[0],QC[0],QD[0],QP[0],gamma1,gamma2,oodelta,B_vals)
    By = B_array(ma,mb,mc,md,PA[1],PB[1],QC[1],QD[1],QP[1],gamma1,gamma2,oodelta,B_vals)
    Bz = B_array(na,nb,nc,nd,PA[2],PB[2],QC[2],QD[2],QP[2],gamma1,gamma2,oodelta,B_vals)
    boys_arg = 0.25*rpq2/delta
    #boys_eval = boys(np.arange(13), boys_arg) # supports up to f functions

    with loops.Scope() as s:
      s.I = 0
      s.J = 0  
      s.K = 0 
      s.primitive = 0.
      s.I = 0 
      for _ in s.while_range(lambda: s.I < la + lb + lc + ld + 1):
        s.J = 0 
        for _ in s.while_range(lambda: s.J < ma + mb + mc + md + 1):
          s.K = 0 
          for _ in s.while_range(lambda: s.K < na + nb + nc + nd + 1):
            #s.primitive += Bx[s.I] * By[s.J] * Bz[s.K] * boys_eval[s.I + s.J + s.K]
            s.primitive += Bx[s.I] * By[s.J] * Bz[s.K] * boys(s.I + s.J + s.K, boys_arg)
            s.K += 1
          s.J += 1
        s.I += 1
      value = 2*jax.lax.pow(np.pi,2.5)/(gamma1*gamma2*np.sqrt(gamma1+gamma2)) \
              *np.exp(-aa*bb*rab2/gamma1) \
              *np.exp(-cc*dd*rcd2/gamma2)*s.primitive*coef
      return value

#@primitive_tei.defjvp
def primitive_tei_jvp(la,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef, primals, tangents):
    '''
    Okay, so 
    '''
    Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz = primals
    # These are one-hot 'basis' vectors. 
    Ax_dot,Ay_dot,Az_dot,Bx_dot,By_dot,Bz_dot,Cx_dot,Cy_dot,Cz_dot,Dx_dot,Dy_dot,Dz_dot = tangents
    primals_out = primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)

    if la == 0:
        tei_Ax = 2 * aa * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la+1,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)
    else:
        tei_Ax = 2 * aa * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la+1,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef) -\
                     la * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la-1,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)
    if ma == 0:
        tei_Ay = 2 * aa * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma+1,na,lb,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)
    else:
        tei_Ay = 2 * aa * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma+1,na,lb,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef) -\
                     ma * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma-1,na,lb,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)
    if na == 0:
        tei_Az = 2 * aa * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na+1,lb,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)
    else:                                                 
        tei_Az = 2 * aa * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na+1,lb,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef) -\
                     na * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na-1,lb,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)

    if lb == 0:
        tei_Bx = 2 * bb * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb+1,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)
    else:                                                    
        tei_Bx = 2 * bb * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb+1,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef) -\
                     lb * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb-1,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)
    if mb == 0:
        tei_By = 2 * bb * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb+1,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)
    else:                                                       
        tei_By = 2 * bb * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb+1,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef) -\
                     mb * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb-1,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)
    if nb == 0:
        tei_Bz = 2 * bb * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb+1,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)
    else:                                                          
        tei_Bz = 2 * bb * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb+1,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef) -\
                     nb * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb-1,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)

    if lc == 0:
        tei_Cx = 2 * cc * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc+1,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)
    else:                                                             
        tei_Cx = 2 * cc * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc+1,mc,nc,ld,md,nd,aa,bb,cc,dd,coef) -\
                     lc * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc-1,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)
    if mc == 0:
        tei_Cy = 2 * cc * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc,mc+1,nc,ld,md,nd,aa,bb,cc,dd,coef)
    else:                                                                
        tei_Cy = 2 * cc * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc,mc+1,nc,ld,md,nd,aa,bb,cc,dd,coef) -\
                     mc * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc,mc-1,nc,ld,md,nd,aa,bb,cc,dd,coef)
    if nc == 0:
        tei_Cz = 2 * cc * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc,mc,nc+1,ld,md,nd,aa,bb,cc,dd,coef)
    else:                                                                   
        tei_Cz = 2 * cc * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc,mc,nc+1,ld,md,nd,aa,bb,cc,dd,coef) -\
                     nc * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc,mc,nc-1,ld,md,nd,aa,bb,cc,dd,coef)

    if ld == 0:
        tei_Dx = 2 * dd * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc,mc,nc,ld+1,md,nd,aa,bb,cc,dd,coef)
    else:                                                                      
        tei_Dx = 2 * dd * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc,mc,nc,ld+1,md,nd,aa,bb,cc,dd,coef) -\
                     ld * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc,mc,nc,ld-1,md,nd,aa,bb,cc,dd,coef)
    if md == 0:
        tei_Dy = 2 * dd * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc,mc,nc,ld,md+1,nd,aa,bb,cc,dd,coef)
    else:                                                                         
        tei_Dy = 2 * dd * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc,mc,nc,ld,md+1,nd,aa,bb,cc,dd,coef) -\
                     md * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc,mc,nc,ld,md-1,nd,aa,bb,cc,dd,coef)
    if nd == 0:
        tei_Dz = 2 * dd * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd+1,aa,bb,cc,dd,coef)
    else:                                                                            
        tei_Dz = 2 * dd * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd+1,aa,bb,cc,dd,coef) -\
                     nd * primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd-1,aa,bb,cc,dd,coef)
    
    # Not sure how to handle this...
    # Does this need to be the total differential?
    tangents_out = tei_Ax * Ax_dot +  tei_Ay * Ay_dot + tei_Az * Az_dot +\
                   tei_Bx * Bx_dot +  tei_By * By_dot + tei_Bz * Bz_dot +\
                   tei_Cx * Cx_dot +  tei_Cy * Cy_dot + tei_Cz * Cz_dot +\
                   tei_Dx * Dx_dot +  tei_Dy * Dy_dot + tei_Dz * Dz_dot

    #          np.dot(A_dot,np.array([tei_Ax, tei_Ay, tei_Az])) +\
    #               np.dot(B_dot,np.array([tei_Bx, tei_By, tei_Bz])) +\
    #               np.dot(C_dot,np.array([tei_Cx, tei_Cy, tei_Cz])) +\
    #               np.dot(D_dot,np.array([tei_Dx, tei_Dy, tei_Dz]))
    return primals_out, tangents_out
    #A_dot[0] * tei_Ax

    #tangents_out = 

#primitive_tei.defjvps(lambda Ax_dot, primal_out, Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz:
                        


A = np.array([0.0,0.0,1.0])
B = np.array([0.0,0.0,-1.0])
C = np.array([0.0,0.0,2.0])
D = np.array([0.0,0.0,-2.0])
la,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd = 1,0,0,1,0,0,1,0,0,1,0,0
aa, bb, cc, dd = 1.0,1.0,1.0,1.0
coef = 1.0
#print('compiling')
#huh = primitive_tei(A,B,C,D,la,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)
#huh = primitive_tei(A,B,C,D,la,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)
#print(huh)

#what = jax.jacfwd(primitive_tei, 0)(A,B,C,D,la,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)
#print(what)

def testfunc(geom):
    Ax,Ay,Az = geom[0]
    Bx,By,Bz = geom[1]
    Cx,Cy,Cz = geom[0]
    Dx,Dy,Dz = geom[1]
    la,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd = 1,0,0,1,0,0,1,0,0,1,0,0
    aa, bb, cc, dd = 1.0,1.0,1.0,1.0
    coef = 1.0
    return primitive_tei(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Dx,Dy,Dz,la,ma,na,lb,mb,nb,lc,mc,nc,ld,md,nd,aa,bb,cc,dd,coef)

geom = np.array([[0.0,0.0,0.8], [0.0,0.0,-0.8]])
res = testfunc(geom) 
print(res)
deriv = jax.jacfwd(jax.jacfwd(testfunc))(geom)
print(deriv)



