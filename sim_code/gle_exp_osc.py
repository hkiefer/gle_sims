#embedded Markovian simulation for an oscillating, decaying kernel, where tau_m is very long
import numpy as np
import math
from numba import njit

@njit(nopython=True)
def integrate_exp_osc(nsteps, dt, m, k, my, gammay, x0, y0, v0,w0, kT,U0):
    
    x=np.zeros(nsteps,dtype=np.float64)
    v=np.zeros(nsteps,dtype=np.float64)
    x[0]=x0
    
    fac_randy=math.sqrt(2*kT*gammay/dt)
    fac_pot=4*U0
        
    xx=x0
    yy=y0
    vv=v0
    ww=w0

    for var in range(1,int(nsteps)):
        
        xi = np.random.normal(0.0,1.0) 

        fr = fac_randy*xi

        kx1=dt*vv
        ky1=dt*ww        
        kv1=dt*(k*(yy-xx) - fac_pot*(xx**3-xx))/m
        kw1=dt*(-gammay*(ww)/my+k*(xx-yy)/my + fr/my)
            
        x1=xx+kx1/2
        v1=vv+kv1/2
        y1=yy+ky1/2
        w1=ww+kw1/2
        
        
        kx2=dt*v1
        ky2=dt*w1        
        kv2=dt*(k*(y1-x1) - fac_pot*(x1**3-x1))/m
        kw2=dt*(-gammay*(w1)/my+k*(x1-y1)/my + fr/my)
            
        x2=xx+kx2/2
        v2=vv+kv2/2
        y2=yy+ky2/2
        w2=ww+kw2/2
        
        kx3=dt*v2
        ky3=dt*w2        
        kv3=dt*(k*(y2-x2) - fac_pot*(x2**3-x2))/m
        kw3=dt*(-gammay*(w2)/my+k*(x2-y2)/my + fr/my)
         
        x3=xx+kx3
        v3=vv+kv3
        y3=yy+ky3
        w3=ww+kw3
        
        kx4=dt*v3
        ky4=dt*w3        
        kv4=dt*(k*(y3-x3) - fac_pot*(x3**3-x3))/m
        kw4=dt*(-gammay*(w3)/my+k*(x3-y3)/my + fr/my)

            
            
        xx=xx + (kx1+2*kx2+2*kx3+kx4)/6
        vv=vv +(kv1+2*kv2+2*kv3+kv4)/6
        yy+= (ky1+(2*ky2)+(2*ky3)+ky4)/6
        ww+= (kw1+(2*kw2)+(2*kw3)+kw4)/6
        

        x[var]=xx


    return x,vv,yy,ww