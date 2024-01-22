

import src.parser as param
import src.timeInt as prop
import src.FFT as FFT
import src.FDM as FDM
import src.setupRun as setup

import time
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Parareal():

    params      : param.PARAMS
    settings    : setup.simulation_setup

    cmethod     : dataclass = field(init=False)
    fmethod     : dataclass = field(init=False)

    propc       : str = field(init=False)
    propf       : str = field(init=False)

    def __post_init__(self) -> None:

        if self.params.par_SDMc == 'FFT':
            print('Coarse Method is \t: FFT')
            self.cmethod = FFT.FFT(self.params,'PAR_COARSE')

        elif self.params.par_SDMc == 'FDM':
            print('Coarse Method is \t: FDM')
            self.cmethod = FDM.FDM(self.params,'PAR_COARSE')
        else:
            print('Spatial Discretization for coarse Method unknown!')

        if self.params.par_SDMf == 'FFT':
            print('Fine Method is   \t: FFT')
            self.fmethod = FFT.FFT(self.params,'PAR_FINE')

        elif self.params.par_SDMf == 'FDM':
            print('Fine Method is   \t: FDM')
            self.fmethod = FDM.FDM(self.params,'PAR_FINE')
        else:
            print('Spatial Discretization for fine Method unknown!')

        self.propc = self.params.par_propc
        self.propf = self.params.par_propf

    def runParareal(self):

        # U0 run 

        print('\tExecute U0 run!')
        NT = self.params.par_Nt+1

        Dt = (self.params.par_Tn-self.params.par_T0)/self.params.par_Nt

        print("\tSpatial resolution  :: ",self.params.par_N)
        print("\tSimulation interval :: ",self.params.par_Tn)
        print("\tTime Slices         :: ",self.params.par_Nt)
        print("\tCoarse time step    :: ",self.params.par_dtc)
        print("\tFine   time step    :: ",self.params.par_dtf)
        print("\tCoarse propagator   :: ",self.params.par_propc)
        print("\tFine   propagator   :: ",self.params.par_propf)
        print("\tCoarse Smag Model   :: ",self.params.par_smagc)
        if self.params.par_smagc:
            print("\tCoarse Smag Const   :: ",self.params.par_Csc)
        print("\tFine   Smag Model   :: ",self.params.par_smagf)
        if self.params.par_smagf:
            print("\tFine   Smag Const   :: ",self.params.par_Csf)
        print("")
        print("\t\t Forcing fixed          :: ",self.params.freeze)

        if self.params.freeze:
            np.random.seed(self.params.seed)

        tt=time.time()

        # run first coarse run to obtain U0
        prop.solve(
                self.params,
                self.cmethod,
                self.propc,
                self.settings.par_X,
                self.settings.par_g,
                self.settings.par_Ug,
                self.settings.par_Eg,
                self.settings.par_KEg,
                self.params.par_T0,
                self.params.par_Tn,
                self.params.par_dtc,
                self.params.par_Nt) 

        self.settings.par_u  = np.copy(self.settings.par_g)
        self.settings.par_E  = np.copy(self.settings.par_Eg)
        self.settings.par_KE = np.copy(self.settings.par_KEg)

        tend=time.time()-tt
        print('\tU0 took %0.2f seconds'%tend)

        
        print("\tStart Parareal iteration ::")

        self.parareal_iterarion(Dt)

        print("\tParareal run finished!")

    def parareal_iterarion(self,Dt):

        if self.params.par_Kmax < self.params.par_Nt:
            Kmax = self.params.par_Kmax
        else:
            Kmax = self.params.par_Nt

        for i in range(Kmax): 
            ii=i+1
            print("\tIteration :: ",ii)

            # fine run
            tt = time.time()            

            tfine = tt
            
            if self.params.freeze:
                np.random.seed(self.params.seed)
                for s in range(i):
                    for m in range(int(Dt/self.params.par_dtf)):
                        out = np.random.rand(self.params.noise_dim)

            for j in range(i,self.params.par_Nt):

                T0=self.params.par_T0+j*Dt
                TN=T0+Dt
                
                #print(out[0])
                
                if self.params.par_propf == 'RK1':
                    prop.RK1_step(
                            self.params,
                            self.fmethod,
                            self.settings.par_X,
                            self.settings.par_u[:,j],
                            self.settings.par_f[:,j+1],
                            self.settings.par_Uf[j:,:],
                            self.settings.par_Ef[:,j],
                            self.settings.par_KEf[j,:,:],
                            T0,
                            TN,
                            self.params.par_dtf)
                elif self.params.par_propf == 'AB2':
                    prop.AB2_step(
                            self.params,
                            self.fmethod,
                            self.settings.par_X,
                            self.settings.par_u[:,j],
                            self.settings.par_f[:,j+1],
                            self.settings.par_Uf[j:,:],
                            self.settings.par_Ef[:,j],
                            self.settings.par_KEf[j,:,:],
                            T0,
                            TN,
                            self.params.par_dtf)
                elif self.params.par_propf == 'RK2':
                    prop.RK2_step(
                            self.params,
                            self.fmethod,
                            self.settings.par_X,
                            self.settings.par_u[:,j],
                            self.settings.par_f[:,j+1],
                            self.settings.par_Uf[j:,:],
                            self.settings.par_Ef[:,j],
                            self.settings.par_KEf[j,:,:],
                            T0,
                            TN,
                            self.params.par_dtf)
                elif self.params.par_propf == 'SRK2':
                    prop.SRK2_step(
                            self.params,
                            self.fmethod,
                            self.settings.par_X,
                            self.settings.par_u[:,j],
                            self.settings.par_f[:,j+1],
                            self.settings.par_Uf[j:,:],
                            self.settings.par_Ef[:,j],
                            self.settings.par_KEf[j,:,:],
                            T0,
                            TN,
                            self.params.par_dtf)
                elif self.params.par_propf == 'RK4':
                    prop.RK4_step(
                            self.params,
                            self.fmethod,
                            self.settings.par_X,
                            self.settings.par_u[:,j],
                            self.settings.par_f[:,j+1],
                            self.settings.par_Uf[j:,:],
                            self.settings.par_Ef[:,j],
                            self.settings.par_KEf[j,:,:],
                            T0,
                            TN,
                            self.params.par_dtf)
                elif self.params.par_propf == 'SRK4':
                    prop.SRK4_step(
                            self.params,
                            self.fmethod,
                            self.settings.par_X,
                            self.settings.par_u[:,j],
                            self.settings.par_f[:,j+1],
                            self.settings.par_Uf[j:,:],
                            self.settings.par_Ef[:,j],
                            self.settings.par_KEf[j,:,:],
                            T0,
                            TN,
                            self.params.par_dtf)
                else:
                    sys.exit("PARAREAL error :: Fine Propagator unknown!")

            # copy f to u in the kth slice
            self.settings.par_u[:,ii] = np.copy(self.settings.par_f[:,ii])
            self.settings.par_E[:,i]  = np.copy(self.settings.par_Ef[:,i])
            self.settings.par_KE[i,:,:] = np.copy(self.settings.par_KEf[i,:,:])

            tfineEnd=time.time()
            tfinepar=(tfineEnd-tfine)/(self.params.par_Nt-i)
            print("\tFine run took ",tfineEnd-tfine," seconds. Pseudo Parallel execution :: ",tfinepar," seconds.")

            print("\tStart update procedure.")

            tupdate=time.time()
            # update procedure 

            if self.params.freeze:
                np.random.seed(self.params.seed)
                for s in range(i+1):
                    for m in range(int(Dt/self.params.par_dtf)):
                        out = np.random.rand(self.params.noise_dim)
                        
            for j in range(ii,self.params.par_Nt):

                T0=self.params.par_T0+j*Dt
                TN=T0+Dt

                if self.propc == "RK1":
                   prop.RK1_step(
                           self.params,
                           self.cmethod,
                           self.settings.par_X,
                           self.settings.par_u[:,j],
                           self.settings.par_g1[:,j+1],
                           self.settings.par_Ug1[j:,:],
                           self.settings.par_Eg1[:,j],
                           self.settings.par_KEg1[j,:,:],
                           T0,
                           TN,
                           self.params.par_dtc)
                elif self.propc == "AB2":
                   prop.AB2_step(
                           self.params,
                           self.cmethod,
                           self.settings.par_X,
                           self.settings.par_u[:,j],
                           self.settings.par_g1[:,j+1],
                           self.settings.par_Ug1[j:,:],
                           self.settings.par_Eg1[:,j],
                           self.settings.par_KEg1[j,:,:],
                           T0,
                           TN,
                           self.params.par_dtc)
                elif self.params.par_propf == 'RK2':
                    prop.RK2_step(
                            self.params,
                            self.fmethod,
                            self.settings.par_X,
                            self.settings.par_u[:,j],
                            self.settings.par_g1[:,j+1],
                            self.settings.par_Ug1[j:,:],
                            self.settings.par_Eg1[:,j],
                            self.settings.par_KEg1[j,:,:],
                            T0,
                            TN,
                            self.params.par_dtf)
                elif self.params.par_propf == 'SRK2':
                    prop.SRK2_step(
                            self.params,
                            self.fmethod,
                            self.settings.par_X,
                            self.settings.par_u[:,j],
                            self.settings.par_g1[:,j+1],
                            self.settings.par_Ug1[j:,:],
                            self.settings.par_Eg1[:,j],
                            self.settings.par_KEg1[j,:,:],
                            T0,
                            TN,
                            self.params.par_dtf)
                elif self.propc == "RK4":
                   prop.RK4_step(
                           self.params,
                           self.cmethod,
                           self.settings.par_X,
                           self.settings.par_u[:,j],
                           self.settings.far_g1[:,j+1],
                            self.settings.par_Ug1[j:,:],
                           self.settings.par_Eg1[:,j],
                           self.settings.par_KEg1[j,:,:],
                           T0,
                           TN,
                           self.params.par_dtc)
                elif self.params.par_propf == 'SRK4':
                    prop.SRK4_step(
                            self.params,
                            self.fmethod,
                            self.settings.par_X,
                            self.settings.par_u[:,j],
                            self.settings.par_g1[:,j+1],
                            self.settings.par_Ug1[j:,:],
                            self.settings.par_Eg1[:,j],
                            self.settings.par_KEg1[j,:,:],
                            T0,
                            TN,
                            self.params.par_dtf)
                else:
                    print("Coarse Propagator for Parareal unknown!")
                    pass

                # UK+1 = Gk+1 + Fk - Gk
                self.settings.par_u[:,j+1] = np.copy(self.settings.par_g1[:,j+1] + self.settings.par_f[:,j+1] - self.settings.par_g[:,j+1])
                
                # Energy spectrum
                self.settings.par_E[:,j] = np.copy(self.settings.par_Ef[:,j])

                # Kinetic energy
                self.settings.par_KE[j,:,:] = np.copy(self.settings.par_KEg1[j,:,:] + self.settings.par_KEf[j,:,:] - self.settings.par_KEg[j,:,:])

            # output 

            self.settings.save_par_iteration(ii)

            # replace Gk with Gk+1
            self.settings.par_g   = np.copy(self.settings.par_g1)
            self.settings.par_Eg  = np.copy(self.settings.par_Eg1)
            self.settings.par_EKg = np.copy(self.settings.par_KEg1)
            
            tendupdate=time.time()-tupdate
            print("\tUpdate procedure took ",tendupdate," seconds.")

            tend=time.time()-tt
            print("\tIteration No ",ii," took %0.2f seconds."%tend)
            tendspt=tfinepar+tendupdate
            print("\tIteration No ",ii," took %0.2f seconds (pseudo wall time)."%tendspt)


    def apply_filter(self,Filter,u):
        # Attempt to increase stability by filtering the iterative approximations. Did not
        # provide any benefits to the iteration.

        # index  :: i-4  i-3  i-2  i-1  i  i+1  i+2  i+3  i+4
        # nproll ::  4    3    2    1      -1   -2   -3   -4
        print("Filter applied is ",Filter)

        if Filter == "B2":
            # Low-pass filter binomial over 3 points B2
            smooth = 0.25 * (np.roll(u,1)  + 2*u + np.roll(u,-1) )
        elif Filter == "B21": 
            # Low-pass filter binomial over 5 points B(2,1)
            smooth = ( -np.roll(u,2) + 4*np.roll(u,1) + 10*u + 4*np.roll(u,-1) - np.roll(u,-2) )/16
        elif Filter == "B31":
            # Low-pass filter binomial over 7 points B(3,1)
            smooth = ( np.roll(u,3) - 6*np.roll(u,2) + 15*np.roll(u,1) + 44*u + 15*np.roll(u,-1) - 6*np.roll(u,-2) + np.roll(u,-3) )/64
        elif Filter == "B41":
            # Low-pass filter binomial over 9 points B(4,1)
            smooth = ( -np.roll(u,4) +8*np.roll(u,3) - 28*np.roll(u,2) + 56*np.roll(u,1) + 186*u + \
                     56*np.roll(u,-1) - 28*np.roll(u,-2) + 8*np.roll(u,-3) - np.roll(u,-4) )/256;
        else:
            smooth = u

        return smooth

