
import src.parser as param
#import sry.setupRun as setup 

import numpy as np 
from dataclasses import dataclass, field
import sys
from scipy import interpolate
from scipy import signal
import pandas as pd



@dataclass
class Interpolation():

    params : param.PARAMS
    settings : object

    lift : str      = field(init=False)
    rest : str      = field(init=False)




    def __post_init__(self) -> None:

        self.lift = self.params.mmpar_L   
        self.rest = self.params.mmpar_R   

        print("\tLift     Interpolation (coarse to fine) : ",self.lift)
        print("\tRestrict Interpolation (fine to coarse) : ",self.rest)


    def lift_state(self,u):
        
        if self.lift == "NN":
            xc = self.settings.mmpar_Xc
            xf = self.settings.mmpar_Xf

            XC = np.append(xc,2.0*np.pi)
            XF = np.append(xf,2.0*np.pi)

            U = np.append(u,u[0])

            Ifunc = interpolate.interp1d(XC, U,kind='nearest')
            uintp = Ifunc(XF)
        elif self.lift == "LIN":
            xc = self.settings.mmpar_Xc
            xf = self.settings.mmpar_Xf

            XC = np.append(xc,2.0*np.pi)
            XF = np.append(xf,2.0*np.pi)

            U = np.append(u,u[0])

            Ifunc = interpolate.interp1d(XC, U,kind='linear')
            uintp = Ifunc(XF)
        elif self.lift == "QUAD":
            xc = self.settings.mmpar_Xc
            xf = self.settings.mmpar_Xf

            XC = np.append(xc,2.0*np.pi)
            XF = np.append(xf,2.0*np.pi)

            U = np.append(u,u[0])

            Ifunc = interpolate.interp1d(XC, U,kind='quadratic')
            uintp = Ifunc(XF)
        elif self.lift == "CUBIC":
            xc = self.settings.mmpar_Xc
            xf = self.settings.mmpar_Xf

            XC = np.append(xc,2.0*np.pi)
            XF = np.append(xf,2.0*np.pi)

            U = np.append(u,u[0])

            Ifunc = interpolate.interp1d(XC, U,kind='cubic')
            uintp = Ifunc(XF)
        elif self.lift == "FFT":
            uintp = signal.resample(u,self.params.mmpar_Nf)
        elif self.lift == "CONSV1":
            uintp = self.lift_consv(u)
        elif self.lift == "CONSV2":
            uintp = self.lift_consv2(u)
        else:
            print("Lifting operator ",self.lift," unknown!")
            sys.exit()

        return uintp[0:self.params.mmpar_Nf]

    def restrict_state(self,u):

        if self.rest == "NN":
            xc = self.settings.mmpar_Xc
            xf = self.settings.mmpar_Xf

            XC = np.append(xc,2.0*np.pi)
            XF = np.append(xf,2.0*np.pi)

            U = np.append(u,u[0])

            Ifunc = interpolate.interp1d(XF, U,kind='nearest')
            uintp = Ifunc(XC)
        elif self.rest == "LIN":
            xc = self.settings.mmpar_Xc
            xf = self.settings.mmpar_Xf

            XC = np.append(xc,2.0*np.pi)
            XF = np.append(xf,2.0*np.pi)

            U = np.append(u,u[0])

            Ifunc = interpolate.interp1d(XF, U,kind='linear')
            uintp = Ifunc(XC)
        elif self.rest == "QUAD":
            xc = self.settings.mmpar_Xc
            xf = self.settings.mmpar_Xf

            XC = np.append(xc,2.0*np.pi)
            XF = np.append(xf,2.0*np.pi)

            U = np.append(u,u[0])

            Ifunc = interpolate.interp1d(XF, U,kind='quadratic')
            uintp = Ifunc(XC)
        elif self.rest == "CUBIC":
            xc = self.settings.mmpar_Xc
            xf = self.settings.mmpar_Xf

            XC = np.append(xc,2.0*np.pi)
            XF = np.append(xf,2.0*np.pi)

            U = np.append(u,u[0])

            Ifunc = interpolate.interp1d(XF, U,kind='cubic')
            uintp = Ifunc(XC)
        elif self.rest == "FFT":
            uintp = signal.resample(u,self.params.mmpar_Nc)
        elif self.rest == "CONSV1":
            uintp = self.restrict_consv(u)
        else:
            print("Restriction operator ",self.rest," unknown!")
            sys.exit()

        return uintp[0:self.params.mmpar_Nc]

    def lift_spectrum(self,E):

        nc = self.params.mmpar_Nc 
        nf = self.params.mmpar_Nf 
        M  = int(nc/2)

        Er = np.zeros(nf)

        Er[0:M]    = np.copy(E[0:M])
        Er[nf-M:nf] = np.copy(E[M:nc])
        Er[M]=0.0

        return Er

    def restrict_spectrum(self,E):

        nc = self.params.mmpar_Nc 
        nf = self.params.mmpar_Nf 
        M  = int(nc/2)

        Er = np.zeros(nc)

        Er[0:M]    = np.copy(E[0:M])
        Er[M:nc] = np.copy(E[nf-M:M])
        Er[M]      = 0.0

        return Er

    def restrict_consv(self,u_):

        u = np.copy(u_)

        Nf = self.params.mmpar_Nf
        Nc = self.params.mmpar_Nc

        Xf = self.settings.mmpar_Xf
        Xc = self.settings.mmpar_Xc

        Uintp = np.zeros(Nc)

        dV  = Xc[1]-Xc[0]
        dVf = Xf[1]-Xf[0]

        inc = np.zeros((Nc,2))
        inf = np.zeros((Nf,2))

        for i in range(Nc):
            inc[i,0] = Xc[i]-dV/2.0
            inc[i,1] = Xc[i]+dV/2.0
        for i in range(Nf):
            inf[i,0] = Xf[i]-dVf/2.0
            inf[i,1] = Xf[i]+dVf/2.0

        for i in range(Nc):
            intc = pd.Interval(inc[i,0],inc[i,1])

            W = np.nan
            I = -99

            for j in range(Nf):
                intf = pd.Interval(inf[j,0],inf[j,1])
                if intf.overlaps(intc):
                    if inf[j,1] <= inc[i,1] and inf[j,0] >= inc[i,0]:
                        I = np.append(I,j)
                        W = np.append(W,1.0)
                    elif inf[j,0] < inc[i,0]:
                        I = np.append(I,j)
                        W = np.append(W,np.abs(inf[j,1] - inc[i,0])/dVf)
                    elif inf[j,1] > inc[i,1]:
                        I = np.append(I,j)
                        W = np.append(W,np.abs(inc[i,1] - inf[j,0])/dVf)

            W = np.delete(W,0)
            I = np.delete(I,0)

            if i==0:
                I0 = -99
                W0 = np.nan
                for l in range(1,len(I)):
                    I0 = np.append(I0,-I[len(I)-l])
                    W0 = np.append(W0,W[len(I)-l])
                I0 = np.delete(I0,0)
                W0 = np.delete(W0,0)

                W = np.append(W0,W)
                I = np.append(I0,I)

            for l in range(len(I)):
                Uintp[i] += (dVf/dV)*W[l]*u[I[l]]


        return Uintp



    def lift_consv(self,u_):

        u = np.copy(u_)

        Nf = self.params.mmpar_Nf
        Nc = self.params.mmpar_Nc

        Xf = self.settings.mmpar_Xf
        Xc = self.settings.mmpar_Xc
       
        Uintp = np.zeros(Nf)

        inc = np.zeros((Nc,2))
        inf = np.zeros((Nf,2))

        dV  = Xc[1]-Xc[0]
        dVf = Xf[1]-Xf[0]

        for i in range(Nc):
            inc[i,0] = Xc[i]-dV/2.0
            inc[i,1] = Xc[i]+dV/2.0
        for i in range(Nf):
            inf[i,0] = Xf[i]-dVf/2.0
            inf[i,1] = Xf[i]+dVf/2.0

        for i in range(Nc):
            intc = pd.Interval(inc[i,0],inc[i,1])

            W = np.nan
            I = -99

            for j in range(Nf):
                intf = pd.Interval(inf[j,0],inf[j,1])
                if intf.overlaps(intc):
                    if inf[j,1] <= inc[i,1] and inf[j,0] >= inc[i,0]:
                        I = np.append(I,j)
                        W = np.append(W,1.0)
                    elif inf[j,0] < inc[i,0]:
                        I = np.append(I,j)
                        W = np.append(W,np.abs(inf[j,1] - inc[i,0])/dVf)
                    elif inf[j,1] > inc[i,1]:
                        I = np.append(I,j)
                        W = np.append(W,np.abs(inc[i,1] - inf[j,0])/dVf)

            W = np.delete(W,0)
            I = np.delete(I,0)

            if i==0:
                I0 = -99
                W0 = np.nan
                for l in range(1,len(I)):
                    I0 = np.append(I0,-I[len(I)-l])
                    W0 = np.append(W0,W[len(I)-l])
                I0 = np.delete(I0,0)
                W0 = np.delete(W0,0)

                W = np.append(W0,W)
                I = np.append(I0,I)

            for l in range(len(I)):
                Uintp[I[l]] += W[l]*u[i]


        return Uintp

    def lift_consv2(self,u_):

        u = np.copy(u_)

        Nf = self.params.mmpar_Nf
        Nc = self.params.mmpar_Nc

        Xf = self.settings.mmpar_Xf
        Xc = self.settings.mmpar_Xc
       
        Uintp = np.zeros(Nf)

        inc = np.zeros((Nc,2))
        inf = np.zeros((Nf,2))

        dV  = Xc[1]-Xc[0]
        dVf = Xf[1]-Xf[0]

        for i in range(Nc):
            inc[i,0] = Xc[i]-dV/2.0
            inc[i,1] = Xc[i]+dV/2.0
        for i in range(Nf):
            inf[i,0] = Xf[i]-dVf/2.0
            inf[i,1] = Xf[i]+dVf/2.0

        for i in range(Nc):
            intc = pd.Interval(inc[i,0],inc[i,1])

            W  = np.nan
            I  = -99

            for j in range(Nf):
                intf = pd.Interval(inf[j,0],inf[j,1])
                if intf.overlaps(intc):

                    if inf[j,1] <= inc[i,1] and inf[j,0] >= inc[i,0]:
                        I = np.append(I,j)
                        w = dVf/dVf
                        W = np.append(W,w)
                    elif inf[j,0] < inc[i,0]:
                        I = np.append(I,j)
                        w = (inf[j,1] - inc[i,0])/dVf
                        W = np.append(W,w)
                    elif inf[j,1] > inc[i,1]:
                        I = np.append(I,j)
                        w = (inc[i,1] - inf[j,0])/dVf
                        W = np.append(W,w)
            W = np.delete(W,0)
            I = np.delete(I,0)

            if i==0:
                I0 = -99
                W0 = np.nan
                for l in range(1,len(I)):
                    I0 = np.append(I0,-I[len(I)-l])
                    W0 = np.append(W0,W[len(I)-l])
                I0 = np.delete(I0,0)
                W0 = np.delete(W0,0)

                W = np.append(W0,W)
                I = np.append(I0,I)

            up = np.roll(u,-1)
            um = np.roll(u,1)


            for l in range(len(I)):
                Uintp[I[l]] += W[l]*u[i]
                if I[l] > 0:
                    if Xf[I[l]]-Xc[i] > 0:
                        Uintp[I[l]] += W[l]*(Xf[I[l]]-Xc[i])*(up[i]-um[i])*.5/dV
                    elif  Xf[I[l]]-Xc[i] < 0:
                        Uintp[I[l]] += W[l]*(Xf[I[l]]-Xc[i])*(up[i]-um[i])*.5/dV
                elif I[l] == 0:
                    pass
                else:
                    dx = Xf[I[l]] - Xc[i]
                    if i==0:
                        dx = -(2.0*np.pi - Xf[I[l]])
                    Uintp[I[l]] += W[l]*(dx)*(up[i]-um[i])*.5/dV

        return Uintp

    def interpolate(self,method,u,N):

        ### Function used for outside (of MMPAR) interpolations

        n     = np.shape(u)[0]
        x     = np.arange(0,2.0*np.pi,2.0*np.pi/n)
        xintp = np.arange(0,2.0*np.pi,2.0*np.pi/N)
        uu = np.copy(u)

        if method == 'NN':
            print("Method: NN") 
            X = np.append(x,2.0*np.pi)
            Xintp = np.append(xintp,2.0*np.pi)
            U = np.append(uu,uu[0])
            Ifunc = interpolate.interp1d(X, U,kind='nearest')
            uintp = Ifunc(xintp)

        elif method == 'LIN':
            print("Method: LIN")
            X = np.append(x,2.0*np.pi)
            Xintp = np.append(xintp,2.0*np.pi)
            U = np.append(uu,uu[0])
            Ifunc = interpolate.interp1d(X, U,kind='linear')
            uintp = Ifunc(xintp)

        elif method == 'QUAD':
            print("Method: QUAD") 
            X = np.append(x,2.0*np.pi)
            Xintp = np.append(xintp,2.0*np.pi)
            U = np.append(uu,uu[0])
            Ifunc = interpolate.interp1d(X, U,kind='quadratic')
            uintp = Ifunc(xintp)

        elif method == 'CUBIC':
            print("Method: CUBIC") 
            X = np.append(x,2.0*np.pi)
            Xintp = np.append(xintp,2.0*np.pi)
            U = np.append(uu,uu[0])
            Ifunc = interpolate.interp1d(X, U,kind='cubic')
            uintp = Ifunc(xintp)

        elif method == 'FFT':
            print("Method: FFT") 
            uintp = signal.resample(uu,N)

        else:
            print("Interpolation Method unknown!")
            sys.exit()


        return uintp[0:N]





