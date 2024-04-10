# Integrate GLE with single-exponential memory using 4th order Runga-Kutta algorithm
import numpy as np
import math
from numba import njit


#double-well potential
@njit()
def integrate_sing_exp_dw(nsteps=1e6, dt=0.01, k=1, m=1, gamma=1, U0=7.482,
    x0=0, v0=1, y0=0.5, kT=2.494):
    x=np.zeros((nsteps,),dtype=np.float64)
    x[0]=x0

    fac_pot=4*U0/m
    fac_rand=math.sqrt(2*kT/gamma/dt)

    xx=x[0]
    vv=v0
    yy=y0
    for var in range(1,nsteps):
        xi=np.random.normal(0.0,1.0)

        kx1=dt*vv
        kv1=dt*(-k*(xx-yy)/m-fac_pot*(xx**3-xx))
        ky1=dt*(-k*(yy-xx)/gamma+fac_rand*xi)
        x1=xx+kx1/2
        v1=vv+kv1/2
        y1=yy+ky1/2

        kx2=dt*v1
        kv2=dt*(-k*(x1-y1)/m-fac_pot*(x1**3-x1))
        ky2=dt*(-k*(y1-x1)/gamma+fac_rand*xi)
        x2=xx+kx2/2
        v2=vv+kv2/2
        y2=yy+ky2/2

        kx3=dt*v2
        kv3=dt*(-k*(x2-y2)/m-fac_pot*(x2**3-x2))
        ky3=dt*(-k*(y2-x2)/gamma+fac_rand*xi)
        x3=xx+kx3
        v3=vv+kv3
        y3=yy+ky3

        kx4=dt*v3
        kv4=dt*(-k*(x3-y3)/m-fac_pot*(x3**3-x3))
        ky4=dt*(-k*(y3-x3)/gamma+fac_rand*xi)
        xx+=(kx1+2*kx2+2*kx3+kx4)/6
        vv+=(kv1+2*kv2+2*kv3+kv4)/6
        yy+=(ky1+2*ky2+2*ky3+ky4)/6

        x[var]=xx
       

    return x,vv,yy


#harmonic potential
@njit()
def integrate_sing_exp_harm(nsteps=1e6, dt=0.01, k=1, m=1, gamma=1, k0=1,
    x0=0, v0=1, y0=0.5, kT=2.494):
    x=np.zeros((nsteps,),dtype=np.float64)
    x[0]=x0

    fac_rand=math.sqrt(2*kT/gamma/dt)
    xx=x[0]
    vv=v0
    yy=y0
    for var in range(1,nsteps):
        xi=np.random.normal(0.0,1.0)

        kx1=dt*vv
        kv1=dt*(-k*(xx-yy)-k0*xx)/m
        ky1=dt*(-k*(yy-xx)/gamma+fac_rand*xi)
        x1=xx+kx1/2
        v1=vv+kv1/2
        y1=yy+ky1/2

        kx2=dt*v1
        kv2=dt*(-k*(x1-y1)-k0*x1)/m
        ky2=dt*(-k*(y1-x1)/gamma+fac_rand*xi)
        x2=xx+kx2/2
        v2=vv+kv2/2
        y2=yy+ky2/2

        kx3=dt*v2
        kv3=dt*(-k*(x2-y2)-k0*x2)/m
        ky3=dt*(-k*(y2-x2)/gamma+fac_rand*xi)
        x3=xx+kx3
        v3=vv+kv3
        y3=yy+ky3

        kx4=dt*v3
        kv4=dt*(-k*(x3-y3)-k0*x3)/m
        ky4=dt*(-k*(y3-x3)/gamma+fac_rand*xi)
        xx+=(kx1+2*kx2+2*kx3+kx4)/6
        vv+=(kv1+2*kv2+2*kv3+kv4)/6
        yy+=(ky1+2*ky2+2*ky3+ky4)/6

        x[var]=xx

    return x,vv,yy