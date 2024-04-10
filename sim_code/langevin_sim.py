# Integrate Langevin equation using 4th order Runga-Kutta algorithm
import numpy as np
import math
from numba import njit


@njit()
def integrate_langevin_dw(nsteps=1e6, dt=0.01, m=1, gamma=1, U0=7.482, x0=0, v0=1, kT=2.494):
    
    x=np.zeros(nsteps,dtype=np.float64)
    
    x[0]=x0
    
    fac_pot=4*U0/m
    
    fac_rand=math.sqrt(2*kT*gamma/dt)
    xx=x[0]
    vv=v0

    for var in range(1,nsteps):
        xi=np.random.normal(0.0,1.0)

        kx1=dt*vv
        kv1=dt*(-gamma*(vv)/m-fac_pot*(xx**3-xx) + fac_rand*xi/m)
        x1=xx+kx1/2
        v1=vv+kv1/2


        kx2=dt*v1
        kv2=dt*(-gamma*(v1)/m-fac_pot*(x1**3-x1) + fac_rand*xi/m)
        x2=xx+kx2/2
        v2=vv+kv2/2


        kx3=dt*v2
        kv3=dt*(-gamma*(v2)/m-fac_pot*(x2**3-x2)+ fac_rand*xi/m)
        x3=xx+kx3
        v3=vv+kv3

        kx4=dt*v3
        kv4=dt*(-gamma*(v3)/m-fac_pot*(x3**3-x3)+ fac_rand*xi/m)
        xx+=(kx1+2*kx2+2*kx3+kx4)/6
        vv+=(kv1+2*kv2+2*kv3+kv4)/6


        x[var]=xx


    return x,vv
