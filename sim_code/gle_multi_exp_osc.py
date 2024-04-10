# Integrate GLE with multiple oscillating, decaying kernels using 4th order Runga-Kutta algorithm 

import numpy as np
import math

from numba import njit
from scipy import interpolate

def spline_fe_for_sim(pos,fe,dx = 10, der=1,add_bounds_right=False,start_right=0,add_bounds_left=False,start_left=0):
    dxf=(pos[1]-pos[0])/dx
    fe_spline=interpolate.splrep(pos, fe, s=0, per=0)
    force_bins=np.arange(pos[0],pos[-1],dxf)
    force_matrix=interpolate.splev(force_bins, fe_spline, der=der)

    if add_bounds_right:
        #add quadratic boundary at ends
        
        #xfine_new = np.arange(pos[0],pos[-1]*1.5,dxf)
        xfine_new = np.arange(pos[0],pos[-1],dxf)
        xfine_right = xfine_new[len(force_bins)-start_right:]
        force_fine_right = (xfine_right-force_bins[-start_right])**2*(dxf)*1e9
        force_fine_right +=force_matrix[-start_right]
        force_matrix = np.append(force_matrix[:-start_right],force_fine_right)
        force_bins = xfine_new
        
    if add_bounds_left:
        
        xfine_new = np.arange(pos[0],pos[-1],dxf)
        xfine_left = xfine_new[:start_left]
        force_fine_left = -(xfine_left-force_bins[start_left])**2*(dxf)*1e9
        force_fine_left +=force_matrix[start_left]

        force_matrix = np.append(force_fine_left,force_matrix[start_left:])
        force_bins = xfine_new
    
    return fe_spline,force_bins,force_matrix

@njit()#(numba.types.UniTuple(float64[:],3)(int64,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64),nopython=True,nogil=True)
def integrate_multi_exp_osc_arb_pot(nsteps, dt, m, ks, mys, gammays, x0, y0, v0,w0, kT,force_bins,force_matrix):
    
    n_exp = len(ks)
    x=np.zeros(nsteps,dtype=np.float64)
    x[0]=x0
    
    fac_randy = np.zeros(n_exp)
    for i in range(n_exp):
    
        fac_randy[i]=math.sqrt(2*kT*gammays[i]/dt)
        
    xx=x0
    yy=y0
    vv=v0
    ww=w0
    
    xi = np.zeros(n_exp)
    for var in range(1,int(nsteps)):
        for i in range(n_exp):
            
            xi[i] = np.random.normal(0.0,1.0) 

        fr = np.zeros(n_exp)
        for i in range(n_exp):
            
            fr[i] = fac_randy[i]*xi[i]

        kx1=dt*vv
        ky1=dt*ww
        ff = dU2(xx,force_bins,force_matrix)   
        kv1 = 0
        kv1-=dt*ff/m
        
        kw1 = np.zeros((n_exp,))
        for i in range(n_exp):
            kv1 += dt*(ks[i]*(yy[i]-xx))/m
            kw1[i]=dt*(-gammays[i]*(ww[i])/mys[i]+ks[i]*(xx-yy[i])/mys[i] + fr[i]/mys[i])
            
        
        x1=xx+kx1/2
        v1=vv+kv1/2
        y1=yy+ky1/2
        w1=ww+kw1/2
        
        
        kx2=dt*v1
        ky2=dt*w1
        ff = dU2(x1,force_bins,force_matrix)   
        kv2 = 0
        kv2-=dt*ff/m
        
        kw2 = np.zeros((n_exp,))
        for i in range(n_exp):
            kv2 += dt*(ks[i]*(y1[i]-x1))/m
            kw2[i]=dt*(-gammays[i]*(w1[i])/mys[i]+ks[i]*(x1-y1[i])/mys[i] + fr[i]/mys[i])
            
        
        x2=xx+kx2/2
        v2=vv+kv2/2
        y2=yy+ky2/2
        w2=ww+kw2/2
        
        kx3=dt*v2
        ky3=dt*w2
        ff = dU2(x2,force_bins,force_matrix)   
        kv3 = 0
        kv3-=dt*ff/m
        
        kw3 = np.zeros((n_exp,))
        for i in range(n_exp):
            kv3 += dt*(ks[i]*(y2[i]-x2))/m
            kw3[i]=dt*(-gammays[i]*(w2[i])/mys[i]+ks[i]*(x2-y2[i])/mys[i] + fr[i]/mys[i])
            
        x3=xx+kx3
        v3=vv+kv3
        y3=yy+ky3
        w3=ww+kw3
        
        kx4=dt*v3
        ky4=dt*w3
        ff = dU2(x3,force_bins,force_matrix)   
        kv4 = 0
        kv4-=dt*ff/m
        
        kw4 = np.zeros((n_exp,))
        for i in range(n_exp):
            kv4 += dt*(ks[i]*(y3[i]-x3))/m
            kw4[i]=dt*(-gammays[i]*(w3[i])/mys[i]+ks[i]*(x3-y3[i])/mys[i] + fr[i]/mys[i])
            
            
        xx=xx + (kx1+2*kx2+2*kx3+kx4)/6
        vv=vv +(kv1+2*kv2+2*kv3+kv4)/6
        
        for i in range(n_exp):
            
            yy[i]+= (ky1[i]+(2*ky2[i])+(2*ky3[i])+ky4[i])/6
            ww[i]+= (kw1[i]+(2*kw2[i])+(2*kw3[i])+kw4[i])/6
        

        x[var]=xx


    return x,vv,yy,ww

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