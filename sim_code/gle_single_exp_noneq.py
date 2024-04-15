# Integrate GLE with different friction and random force kernels (single-exponential) using 4th order Runga-Kutta algorithm
import numpy as np
import math
import numba
from numba import njit

@njit()
def integrate_sing_exp_noneq(nsteps=1e6, dt=0.01, tauD=1,taum=1,tauv=1, tauR=1, x0=0, v0=1, yv0=0.5,yR0=0.5, kT=2.494,U0=1):
    x=np.zeros((nsteps,),dtype=np.float64)
    x[0]=x0
    fac_pot=4*U0/kT
    
    xx=x[0]
    vv=v0
    yvy=yv0
    yRy=yR0
    for var in range(1,nsteps):
        xi=np.sqrt(2/dt)*np.random.normal(0.0,1.0)
        
        kx1=dt*vv
        kv1 = dt*tauD/taum*(yvy+yRy-fac_pot*(xx**3-xx))
        kyv1 = -dt*tauD/tauv*(yvy+vv)
        kyR1 = -dt*tauD/tauR*(yRy-xi)
        x1=xx+kx1/2
        v1=vv+kv1/2
        yv1 = yvy+kyv1/2
        yR1 = yRy+kyR1/2

        kx2=dt*v1
        kv2 = dt*tauD/taum*(yv1+yR1-fac_pot*(x1**3-x1))
        kyv2 = -dt*tauD/tauv*(yv1+v1)
        kyR2 = -dt*tauD/tauR*(yR1-xi)
        x2=xx+kx2/2
        v2=vv+kv2/2
        yv2 = yvy+kyv2/2
        yR2 = yRy+kyR2/2

        kx3=dt*v2
        kv3 = dt*tauD/taum*(yv2+yR2-fac_pot*(x2**3-x2))
        kyv3 = -dt*tauD/tauv*(yv2+v2)
        kyR3 = -dt*tauD/tauR*(yR2-xi)
        x3=xx+kx3
        v3=vv+kv3
        yv3 = yvy+kyv3
        yR3 = yRy+kyR3

        kx4=dt*v3
        kv4 = dt*tauD/taum*(yv3+yR3-fac_pot*(x3**3-x3))
        kyv4 = -dt*tauD/tauv*(yv3+v3)
        kyR4 = -dt*tauD/tauR*(yR3-xi)
        
        xx+=(kx1+2*kx2+2*kx3+kx4)/6
        vv+=(kv1+2*kv2+2*kv3+kv4)/6
        yvy+=(kyv1+2*kyv2+2*kyv3+kyv4)/6
        yRy+=(kyR1+2*kyR2+2*kyR3+kyR4)/6

        x[var]=xx

    return x,vv,yvy,yRy

@njit()
def integrate_sing_exp_noneq_harm_pot(nsteps=1e6, dt=0.01, tauD=1,taum=1,tauv=1, tauR=1, x0=0, v0=1, yv0=0.5,yR0=0.5, kT=2.494,k0=1):
    x=np.zeros((nsteps,),dtype=np.float64)
    x[0]=x0
    
    xx=x[0]
    vv=v0
    yvy=yv0
    yRy=yR0
    for var in range(1,nsteps):
        xi=np.sqrt(2/dt)*np.random.normal(0.0,1.0)
        
        kx1=dt*vv
        kv1 = dt*tauD/taum*(yvy+yRy-k0*xx/kT)
        kyv1 = -dt*tauD/tauv*(yvy+vv)
        kyR1 = -dt*tauD/tauR*(yRy-xi)
        x1=xx+kx1/2
        v1=vv+kv1/2
        yv1 = yvy+kyv1/2
        yR1 = yRy+kyR1/2

        kx2=dt*v1
        kv2 = dt*tauD/taum*(yv1+yR1-k0*x1/kT)
        kyv2 = -dt*tauD/tauv*(yv1+v1)
        kyR2 = -dt*tauD/tauR*(yR1-xi)
        x2=xx+kx2/2
        v2=vv+kv2/2
        yv2 = yvy+kyv2/2
        yR2 = yRy+kyR2/2

        kx3=dt*v2
        kv3 = dt*tauD/taum*(yv2+yR2-k0*x2/kT)
        kyv3 = -dt*tauD/tauv*(yv2+v2)
        kyR3 = -dt*tauD/tauR*(yR2-xi)
        x3=xx+kx3
        v3=vv+kv3
        yv3 = yvy+kyv3
        yR3 = yRy+kyR3

        kx4=dt*v3
        kv4 = dt*tauD/taum*(yv3+yR3-k0*x3/kT)
        kyv4 = -dt*tauD/tauv*(yv3+v3)
        kyR4 = -dt*tauD/tauR*(yR3-xi)
        
        xx+=(kx1+2*kx2+2*kx3+kx4)/6
        vv+=(kv1+2*kv2+2*kv3+kv4)/6
        yvy+=(kyv1+2*kyv2+2*kyv3+kyv4)/6
        yRy+=(kyR1+2*kyR2+2*kyR3+kyR4)/6

        x[var]=xx

    return x,vv,yvy,yRy

@njit()
def integrate_sing_exp_noneq_arb_pot(nsteps=1e6, dt=0.01, tauD=1,taum=1,tauv=1, tauR=1, x0=0, v0=1, yv0=0.5,yR0=0.5, kT=2.494,force_bins=[],force_matrix=[]):
    x=np.zeros((nsteps,),dtype=np.float64)
    x[0]=x0
    
    xx=x[0]
    vv=v0
    yvy=yv0
    yRy=yR0
    for var in range(1,nsteps):
        xi=np.sqrt(2/dt)*np.random.normal(0.0,1.0)
        
        kx1=dt*vv
        ff = dU2(xx,force_bins,force_matrix)   
        kv1 = dt*tauD/taum*(yvy+yRy-ff)
        kyv1 = -dt*tauD/tauv*(yvy+vv)
        kyR1 = -dt*tauD/tauR*(yRy-xi)
        x1=xx+kx1/2
        v1=vv+kv1/2
        yv1 = yvy+kyv1/2
        yR1 = yRy+kyR1/2

        kx2=dt*v1
        ff = dU2(x1,force_bins,force_matrix)   
        kv2 = dt*tauD/taum*(yv1+yR1-ff)
        kyv2 = -dt*tauD/tauv*(yv1+v1)
        kyR2 = -dt*tauD/tauR*(yR1-xi)
        x2=xx+kx2/2
        v2=vv+kv2/2
        yv2 = yvy+kyv2/2
        yR2 = yRy+kyR2/2

        kx3=dt*v2
        ff = dU2(x2,force_bins,force_matrix)   
        kv3 = dt*tauD/taum*(yv2+yR2-ff)
        kyv3 = -dt*tauD/tauv*(yv2+v2)
        kyR3 = -dt*tauD/tauR*(yR2-xi)
        x3=xx+kx3
        v3=vv+kv3
        yv3 = yvy+kyv3
        yR3 = yRy+kyR3

        kx4=dt*v3
        ff = dU2(x3,force_bins,force_matrix)   
        kv4 = dt*tauD/taum*(yv3+yR3-ff)
        kyv4 = -dt*tauD/tauv*(yv3+v3)
        kyR4 = -dt*tauD/tauR*(yR3-xi)
        
        xx+=(kx1+2*kx2+2*kx3+kx4)/6
        vv+=(kv1+2*kv2+2*kv3+kv4)/6
        yvy+=(kyv1+2*kyv2+2*kyv3+kyv4)/6
        yRy+=(kyR1+2*kyR2+2*kyR3+kyR4)/6

        x[var]=xx

    return x,vv,yvy,yRy


@njit
def dU2(x,force_bins,force_matrix):

    idx = bisection(force_bins,x)
    value = force_matrix[idx]

    return value

@njit()
def bisection(array,value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
    if (value < array[0]):
        return 0#-1
    elif (value > array[n-1]):
        return n-1
    jl = 0# Initialize lower
    ju = n-1# and upper limits.
    while (ju-jl > 1):# If we are not yet done,
        jm=(ju+jl) >> 1# compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl=jm# and replace either the lower limit
        else:
            ju=jm# or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):# edge cases at bottom
        return 0
    elif (value == array[n-1]):# and top
        return n-1
    else:
        return jl