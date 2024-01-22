

import src.parser as param
import src.FFT as spectral

import time
import numpy as np
from dataclasses import dataclass, field 
from scipy.stats import norm

def noise(X,dt,k1,k2):

    Phi1=2*np.pi*np.random.rand(1)
    Phi2=2*np.pi*np.random.rand(1)

    F = np.sqrt(dt)*( np.cos(k1*X+Phi1) + np.cos(k2*X+Phi2) )

    return F


def Brownian(alpha,N):
    rnd   = np.sqrt(N)*norm.ppf(np.random.rand(N))
    M     = int(N/2)
    K     = np.abs(np.fft.fftfreq(N,d=1/N))
    K[0]  = 1
    frnd  = np.fft.fft(rnd)
    frnd[0] = 0
    frnd[M] = 0
    frndk   = frnd * ( K**(-alpha/2) )
    noise   = np.real(np.fft.ifft(frndk))
        
    return noise

def filterNoise(noise,Nx,N):
    # filter ration
    k = int(N/Nx)
    # signal shape information
    N   = int(N)
    M   = int(Nx)
    L   = int(M/2)
        
    # compute fft then filter
    
    fnoise    = np.fft.fft(noise)
    filtnoise = np.zeros(M,dtype=np.complex128)
    filtnoise[0:L]   = fnoise[0:L]
    filtnoise[L+1:M] = fnoise[N-L+1:N]
    # return from spectral space
    return (1/k)*np.real(np.fft.ifft(filtnoise))

def get_Noise(alpha,Nx,N):

    if Nx < N:
        return filterNoise(Brownian(alpha,N),Nx,N)
    else:
        return Brownian(alpha,N)

def adjust_noise(method,pars):

    if pars.mmpar_nnc == pars.mmpar_nnf:
        pass
    else:
        if method.prop == 'MMPAR_COARSE':
            L = int(pars.mmpar_nnf/pars.mmpar_nnc)
            for i in range(L):
                np.random.rand(8192)
        else:
            pass

def solve(pars,method,prop,X,u,U,E,KE,t0,tn,dt,Nt):

    Dt=(tn-t0)/Nt

    tstart=time.time()
    for i in range(Nt):
        T0=t0+i*Dt
        TN=T0+Dt
        if prop == 'RK1':
            RK1_step(pars,method,X,u[:,i],u[:,i+1],U[i,:,:],E[:,i],KE[i,:,:],T0,TN,dt)
        elif prop == 'AB2':
            AB2_step(pars,method,X,u[:,i],u[:,i+1],U[i,:,:],E[:,i],KE[i,:,:],T0,TN,dt)
        elif prop == 'RK2':
            RK2_step(pars,method,X,u[:,i],u[:,i+1],U[i,:,:],E[:,i],KE[i,:,:],T0,TN,dt)            
        elif prop == 'SRK2':
            SRK2_step(pars,method,X,u[:,i],u[:,i+1],U[i,:,:],E[:,i],KE[i,:,:],T0,TN,dt)            
        elif prop == 'RK4':
            RK4_step(pars,method,X,u[:,i],u[:,i+1],U[i,:,:],E[:,i],KE[i,:,:],T0,TN,dt)
        elif prop == 'SRK4':
            SRK4_step(pars,method,X,u[:,i],u[:,i+1],U[i,:,:],E[:,i],KE[i,:,:],T0,TN,dt)
        elif prop == 'fRK2':
            fRK2_step(pars,method,X,u[:,i],u[:,i+1],U[i,:,:],E[:,i],KE[i,:,:],T0,TN,dt)
        elif prop == 'fRK4':
            fRK4_step(pars,method,X,u[:,i],u[:,i+1],U[i,:,:],E[:,i],KE[i,:,:],T0,TN,dt)
        else:
            print(prop, "method is unknown!")
    tend=time.time()-tstart

    print("\tComputation of the ",method.which()," solver took :: %0.3f seconds."%tend)

def RK1_step(pars,method,X,u0,u,U,E,KE,t0,tn,dt):

    nstep=int((tn-t0)/dt)
    N = method.N
    M = int(N/2)
    NN = set_outputInterval(method,pars,dt)

    uu = np.copy(u0)
    
    t=t0
    Tavg=0.0
    incr=0
        
    which=method.which()
    
    fac = np.sqrt(2.0*1e-06/dt)

    for i in range(1,nstep+1):

        t+=dt

        if pars.noise == 1:
            Fnoise = noise(X,dt,pars.k1,pars.k2)
            uu1 = uu + dt*method.RHS(uu) + Fnoise 
        else:
            Fnoise = fac*get_Noise(0.75,N,pars.noise_dim) 
            uu1 = uu + dt*(method.RHS(uu) + Fnoise)

        
        uu1 = check_Nyquist(uu1,pars,which, M)

        uu = uu1

        tt=time.time()
        # Compute average
        [E[:],KE[:,:],U[:,:],incr] = compute_Diagnostics(uu,t,NN,i,N,U,E,KE,incr,pars)

        Tavg+=(time.time()-tt)

    print("\tComputation of averages took         :: %0.3f seconds."%Tavg)
    u[:] = uu
    E[:] = E[:]/incr


def AB2_step(pars,method,X,u0,u,U,E,KE,t0,tn,dt):

    nstep=int((tn-t0)/dt)
    N = method.N
    M = int(N/2)
    NN = set_outputInterval(method,pars,dt)

    uu = np.copy(u0)
    
    t=t0
    Tavg=0.0
    incr=0
        
    which=method.which()

    fac = np.sqrt(2.0*1e-06/dt)

    for i in range(1,nstep+1):

        t+=dt
        

        if pars.noise == 1:
            rhs = method.RHS(uu)
            Fnoise = noise(X,dt,pars.k1,pars.k2)
            if i==1:
                uu1 = uu + dt*rhs + Fnoise
            else:
                uu1 = uu + dt*(1.5*rhs - 0.5*rhs_old) + Fnoise

        else:
            Fnoise = get_Noise(0.75,N,pars.noise_dim)
            rhs = method.RHS(uu) + fac*Fnoise
            if i==1:
                uu1 = uu + dt*rhs 
            else:
                uu1 = uu + dt*(1.5*rhs - 0.5*rhs_old) 


        uu1 = check_Nyquist(uu1,pars,which,M)

        uu = uu1
        # store rhs for the next time step
        rhs_old     = rhs  
        
        tt=time.time()
        # Compute average
        [E[:],KE[:,:],U[:,:],incr] = compute_Diagnostics(uu,t,NN,i,N,U,E,KE,incr,pars)

        Tavg+=(time.time()-tt)

    print("\tComputation of averages took         :: %0.3f seconds."%Tavg)
    u[:] = uu
    E[:] = E[:]/incr

def RK2_step(pars,method,X,u0,u,U,E,KE,t0,tn,dt):

    nstep=int((tn-t0)/dt)
    N = method.N
    M = int(N/2)

    NN = set_outputInterval(method,pars,dt)

    uu = np.copy(u0)
    
    t=t0
    Tavg=0.0
    incr=0
        
    which=method.which()

    fac = np.sqrt(2.0*1e-06/dt)

    for i in range(1,nstep+1):

        t+=dt
        
    
        if pars.noise == 1:

            F1  = method.RHS(uu) + noise(X,dt,pars.k1,pars.k2)
            F2  = method.RHS(uu+dt*F1) + noise(X,dt,pars.k1,pars.k2)
            uu1 = uu + dt*(F1+F2)/2.0
        else:
            
            F1  = method.RHS(uu) + dt*fac*get_Noise(0.75,N,pars.noise_dim)
            F2  = method.RHS(uu+dt*F1) + dt*fac*get_Noise(0.75,N,pars.noise_dim)

            uu1 = uu + dt*(F1+F2)/2.0 + dt*fac*get_Noise(0.75,N)


        uu1 = check_Nyquist(uu1,pars,which,M)

        uu = uu1
        
        tt=time.time()
        # Compute average
        [E[:],KE[:,:],U[:,:],incr] = compute_Diagnostics(uu,t,NN,i,N,U,E,KE,incr,pars)

        Tavg+=(time.time()-tt)

    print("\tComputation of averages took         :: %0.3f seconds."%Tavg)
    u[:] = uu
    E[:] = E[:]/incr

def RK4_step(pars,method,X,u0,u,U,E,KE,t0,tn,dt):

    nstep=int((tn-t0)/dt)
    N = method.N
    M = int(N/2)

    NN = set_outputInterval(method,pars,dt)

    uu = np.copy(u0)
    
    t=t0
    Tavg=0.0
    incr=0
        
    which=method.which()

    fac = np.sqrt(2.0*1e-06/dt)
    #fac = np.sqrt(1e-06/dt)

    for i in range(1,nstep+1):

        t+=dt
        
    
        if pars.noise == 1:
            Fnoise = noise(X,dt,pars.k1,pars.k2)
            k1  = method.RHS(uu)
            k2  = method.RHS(uu+0.5*dt*k1)
            k3  = method.RHS(uu+0.5*dt*k2)
            k4  = method.RHS(uu+dt*k3)
            uu1 = uu + dt*(k1+2.0*k2+2.0*k3+k4)/6.0 + Fnoise
        else:
            k1  = method.RHS(uu)+dt*fac*get_Noise(0.75,N,pars.noise_dim)
            k2  = method.RHS(uu+0.5*dt*k1)+dt*fac*get_Noise(0.75,N,pars.noise_dim)
            k3  = method.RHS(uu+0.5*dt*k2)+dt*fac*get_Noise(0.75,N,pars.noise_dim)
            k4  = method.RHS(uu+dt*k3)+dt*fac*get_Noise(0.75,N,pars.noise_dim)
            
            uu1 = uu + dt*(k1+2.0*k2+2.0*k3+k4)/6.0 + dt*fac*get_Noise(0.75,N,pars.noise_dim)


        uu1 = check_Nyquist(uu1,pars,which,M)

        uu = uu1
        
        tt=time.time()
        # Compute average
        [E[:],KE[:,:],U[:,:],incr] = compute_Diagnostics(uu,t,NN,i,N,U,E,KE,incr,pars)

        Tavg+=(time.time()-tt)

    print("\tComputation of averages took         :: %0.3f seconds."%Tavg)
    u[:] = uu
    E[:] = E[:]/incr

def SRK2_step(pars,method,X,u0,u,U,E,KE,t0,tn,dt):

    nstep=int((tn-t0)/dt)
    N = method.N
    M = int(N/2)

    NN = set_outputInterval(method,pars,dt)

    uu = np.copy(u0)
    
    t=t0
    Tavg=0.0
    incr=0
        
    which=method.which()

    fac = np.sqrt(2.0*1e-06/dt)

    for i in range(1,nstep+1):

        t+=dt
        
    
        if pars.noise == 1:
            Fnoise = noise(X,dt,pars.k1,pars.k2)
            F1  = method.RHS(uu)
            F2  = method.RHS(uu+dt*F1+Fnoise)
            uu1 = uu + dt*(F1+F2)/2.0 + Fnoise
        else:
            Fnoise=fac*get_Noise(0.75,N,pars.noise_dim)

            F1  = method.RHS(uu)
            F2  = method.RHS(uu+dt*F1+dt*Fnoise)
            
            uu1 = uu + dt*(F1+F2)/2.0 + dt*Fnoise


        adjust_noise(method,pars)
        uu1 = check_Nyquist(uu1,pars,which,M)

        uu = uu1
        
        tt=time.time()
        # Compute average
        [E[:],KE[:,:],U[:,:],incr] = compute_Diagnostics(uu,t,NN,i,N,U,E,KE,incr,pars)

        Tavg+=(time.time()-tt)

    print("\tComputation of averages took         :: %0.3f seconds."%Tavg)
    u[:] = uu
    E[:] = E[:]/incr

def SRK4_step(pars,method,X,u0,u,U,E,KE,t0,tn,dt):

    nstep=int((tn-t0)/dt)
    N = method.N
    M = int(N/2)

    NN = set_outputInterval(method,pars,dt)

    uu = np.copy(u0)
    
    t=t0
    Tavg=0.0
    incr=0
        
    which=method.which()

    fac = np.sqrt(2*1e-06/dt)

    A = np.zeros(5)
    B = np.zeros(5)

    A[0] = 1.0
    A[1] = 0.25+np.sqrt(3)/6
    A[2] = 0.25+np.sqrt(3)/6
    A[3] = 0.50+np.sqrt(3)/6
    A[4] = 1.25+np.sqrt(3)/6
   
    B[0] = 1.0
    B[1] = 0.25-np.sqrt(3)/6+np.sqrt(6)/12
    B[2] = 0.25-np.sqrt(3)/6-np.sqrt(6)/12
    B[3] = 0.50-np.sqrt(3)/6
    B[4] = 1.25-np.sqrt(3)/6+np.sqrt(6)/12

    A=A/np.sqrt(2)
    B=B/np.sqrt(2)

    for i in range(1,nstep+1):

        t+=dt
        
    
        if pars.noise == 1:
            Fnoise  = noise(X,dt,pars.k1,pars.k2)
            Fnoise2 = noise(X,dt,pars.k1,pars.k2)
            k1  = method.RHS(uu)
            k2  = method.RHS(uu+0.5*dt*k1)
            k3  = method.RHS(uu+0.5*dt*k2)
            k4  = method.RHS(uu+dt*k3)
            uu1 = uu + dt*(k1+2.0*k2+2.0*k3+k4)/6.0 + Fnoise
        else:
            
            F1=fac*get_Noise(0.75,N,pars.noise_dim)
            F2=fac*get_Noise(0.75,N,pars.noise_dim)

            k1  = method.RHS(uu)+dt*(A[1]*F1+B[1]*F2)
            k2  = method.RHS(uu+0.5*dt*k1)+dt*(A[2]*F1+B[2]*F2)
            k3  = method.RHS(uu+0.5*dt*k2)+dt*(A[3]*F1+B[3]*F2)
            k4  = method.RHS(uu+dt*k3)+dt*(A[4]*F1+B[4]*F2)
            
            uu1 = uu + dt*(k1+2.0*k2+2.0*k3+k4)/6.0 + dt*(A[0]*F1+B[0]*F2) 


        uu1 = check_Nyquist(uu1,pars,which,M)

        uu = uu1
        
        tt=time.time()
        # Compute average
        [E[:],KE[:,:],U[:,:],incr] = compute_Diagnostics(uu,t,NN,i,N,U,E,KE,incr,pars)

        Tavg+=(time.time()-tt)

    print("\tComputation of averages took         :: %0.3f seconds."%Tavg)
    u[:] = uu
    E[:] = E[:]/incr

def fRK2_step(pars,method,X,u0,u,U,E,KE,t0,tn,dt):

    nstep=int((tn-t0)/dt)
    N = method.N
    M = int(N/2)

    NN = set_outputInterval(method,pars,dt)

    uu = np.copy(u0)
    
    t=t0
    Tavg=0.0
    incr=0
        
    which=method.which()

    fac = np.sqrt(2.0*1e-06/dt)

    for i in range(1,nstep+1):

        t+=dt
        
    
        if pars.noise == 1:

            F1  = method.RHS(uu) + noise(X,dt,pars.k1,pars.k2)
            F2  = method.RHS(uu+dt*F1) + noise(X,dt,pars.k1,pars.k2)
            uu1 = uu + dt*(F1+F2)/2.0
        else:
            
            F1  = method.RHS(uu) + fac*get_Noise(0.75,N,pars.noise_dim)
            F2  = method.RHS(uu+dt*F1) + fac*get_Noise(0.75,N,pars.noise_dim)
            
            uu1 = uu + dt*(F1+F2)/2.0


        uu1 = check_Nyquist(uu1,pars,which,M)

        uu = uu1
        
        tt=time.time()
        # Compute average
        [E[:],KE[:,:],U[:,:],incr] = compute_Diagnostics(uu,t,NN,i,N,U,E,KE,incr,pars)

        Tavg+=(time.time()-tt)

    print("\tComputation of averages took         :: %0.3f seconds."%Tavg)
    u[:] = uu
    E[:] = E[:]/incr

def fRK4_step(pars,method,X,u0,u,U,E,KE,t0,tn,dt):

    nstep=int((tn-t0)/dt)
    N = method.N
    M = int(N/2)

    NN = set_outputInterval(method,pars,dt)

    uu = np.copy(u0)
    
    t=t0
    Tavg=0.0
    incr=0
        
    which=method.which()

    fac = np.sqrt(2.0*1e-06/dt)

    for i in range(1,nstep+1):

        t+=dt
        
    
        if pars.noise == 1:
            Fnoise = noise(X,dt,pars.k1,pars.k2)
            k1  = method.RHS(uu)
            k2  = method.RHS(uu+0.5*dt*k1)
            k3  = method.RHS(uu+0.5*dt*k2)
            k4  = method.RHS(uu+dt*k3)
            uu1 = uu + dt*(k1+2.0*k2+2.0*k3+k4)/6.0 + Fnoise
        else:
            k1  = method.RHS(uu)+fac*get_Noise(0.75,N,pars.noise_dim)
            k2  = method.RHS(uu+0.5*dt*k1)+fac*get_Noise(0.75,N,pars.noise_dim)
            k3  = method.RHS(uu+0.5*dt*k2)+fac*get_Noise(0.75,N,pars.noise_dim)
            k4  = method.RHS(uu+dt*k3)+fac*get_Noise(0.75,N,pars.noise_dim)
            
            uu1 = uu + dt*(k1+2.0*k2+2.0*k3+k4)/6.0


        uu1 = check_Nyquist(uu1,pars,which,M)

        uu = uu1
        
        tt=time.time()
        # Compute average
        [E[:],KE[:,:],U[:,:],incr] = compute_Diagnostics(uu,t,NN,i,N,U,E,KE,incr,pars)

        Tavg+=(time.time()-tt)

    print("\tComputation of averages took         :: %0.3f seconds."%Tavg)
    u[:] = uu
    E[:] = E[:]/incr

def check_Nyquist(u,pars,which,M):

    if which == 'REFERENCE':
        if pars.SDM == 'FFT':
            u = zero_Nyquist(u,M)
    elif which == 'PAR_COARSE':
        if pars.par_SDMc == 'FFT':
            u = zero_Nyquist(u,M)
    elif which == 'PAR_FINE':
        if pars.par_SDMf == 'FFT':
            u = zero_Nyquist(u,M)
    elif which == 'MMPAR_COARSE':
        if pars.mmpar_SDMc == 'FFT':
            u = zero_Nyquist(u,M)
    elif which == 'MMPAR_FINE':
        if pars.mmpar_SDMf == 'FFT':
            u = zero_Nyquist(u,M)
    else:
        pass

    return u

def zero_Nyquist(u,M):

    fftu      = np.fft.fft(u)
    fftu[M]   = 0
    u         = np.real(np.fft.ifft(fftu))

    return u

def compute_Diagnostics(u,t,NN,i,N,U,E,KE,incr,pars):
    if t > pars.Tavg:
        if (i % NN == 0):
            # Energy Spectrum
            fftu=np.fft.fft(u)/N
            u2=0.5*np.multiply(fftu,np.conj(fftu))
            E[:]=E[:]+2.0*np.pi*np.real(u2)
            # Kinetic Energy
            KE[incr,1]=0.5*np.var(u)
            KE[incr,0]=t
            # local states
            if pars.write:
                U[incr,:] = np.copy(u)
            incr+=1
        else:
            pass
    else:
        pass

    return E,KE,U,incr

def set_outputInterval(method,pars,dt):

    if method.prop == 'PAR_COARSE':
        NN = int(pars.nn * (pars.par_dtf/dt))    
    elif method.prop == 'MMPAR_COARSE':
        NN = int(pars.mmpar_nnc)    
    elif method.prop == 'MMPAR_FINE':
        NN = int(pars.mmpar_nnf)    
    else:
        NN = pars.nn 

    return NN

