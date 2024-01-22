

import src.parser as param
import src.timeInt as prop
import src.FFT as FFT
import src.FDM as FDM
import src.setupRun as setup
import src.interpolation as interp

import sys
import time
import numpy as np
from dataclasses import dataclass, field

@dataclass
class MMParareal():

    params      : param.PARAMS
    settings    : setup.simulation_setup
    intp        : interp.Interpolation

    cmethod     : dataclass = field(init=False)
    fmethod     : dataclass = field(init=False)

    propc       : str = field(init=False)
    propf       : str = field(init=False)

    def __post_init__(self) -> None:

        if self.params.mmpar_SDMc == 'FFT':
            print("\tCoarse Method is    :: FFT")
            self.cmethod = FFT.FFT(self.params,'MMPAR_COARSE')

        elif self.params.mmpar_SDMc == 'FDM':
            print("\tCoarse Method is    :: FDM")
            self.cmethod = FDM.FDM(self.params,'MMPAR_COARSE')
        else:
            print('Spatial Discretization for coarse Method unknown!')

        if self.params.mmpar_SDMf == 'FFT':
            print("\tFine Method is      :: FFT")
            self.fmethod = FFT.FFT(self.params,'MMPAR_FINE')

        elif self.params.mmpar_SDMf == "FDM":
            print("\tFine Method is      :: FDM")
            self.fmethod = FDM.FDM(self.params,'MMPAR_FINE')
        else:
            print('Spatial Discretization for fine Method unknown!')

        self.propc = self.params.mmpar_propc
        self.propf = self.params.mmpar_propf

    def test(self):

        print("\tTEST FUNCTION CALL!!")

    def runMMParareal(self):

        # U0 run 

        print('\tExecute U0 run!')
        NT = self.params.mmpar_Nt+1

        Dt = (self.params.mmpar_Tn-self.params.mmpar_T0)/self.params.mmpar_Nt

        print("\tSimulation interval :: ",self.params.mmpar_Tn)
        print("\tCoarse Resolution   :: ",self.params.mmpar_Nc)
        print("\tFine Resolution     :: ",self.params.mmpar_Nf)
        print("\tCoarse time step    :: ",self.params.mmpar_dtc)
        print("\tFine   time step    :: ",self.params.mmpar_dtf)
        print("\tCoarse propagator   :: ",self.params.mmpar_propc)
        print("\tFine   propagator   :: ",self.params.mmpar_propf)
        print("\tViscosity           :: ",self.params.mmpar_nu)
        print("\tCoarse Smag Model   :: ",self.params.mmpar_smagc)
        if self.params.mmpar_smagc:
            print("\tCoarse Smag Const   :: ",self.params.mmpar_Csc)
        print("\tFine   Smag Model   :: ",self.params.mmpar_smagf)
        if self.params.mmpar_smagf:
            print("\tFine   Smag Const   :: ",self.params.mmpar_Csf)
        
        print("\tForcing fixed       :: ",self.params.freeze)

        print("\tIntermediate states :: ",self.params.write)
        print("")

        if self.params.freeze:
            np.random.seed(self.params.seed)

        tt=time.time()

        # run first coarse run to obtain U0
        prop.solve(
                self.params,
                self.cmethod,
                self.propc,
                self.settings.mmpar_Xc,
                self.settings.mmpar_g,
                self.settings.mmpar_Ug,
                self.settings.mmpar_Eg,
                self.settings.mmpar_KEg,
                self.params.mmpar_T0,
                self.params.mmpar_Tn,
                self.params.mmpar_dtc,
                self.params.mmpar_Nt) 
        
        for i in range(self.params.mmpar_Nt):
            self.settings.mmpar_u[:,i+1]  = self.intp.lift_state(np.copy(self.settings.mmpar_g[:,i+1]))
            self.settings.mmpar_E[:,i]  = self.intp.lift_spectrum(np.copy(self.settings.mmpar_Eg[:,i]))
            if self.params.write:
                for l in range(self.params.mmpar_Nt):
                    self.settings.mmpar_Uu[i,l,:] = self.intp.lift_state(np.copy(self.settings.mmpar_Ug[i,l,:]))     

        self.settings.mmpar_KE = np.copy(self.settings.mmpar_KEg)
    
        print("#" * 80)
        print("\tInterpolation check R( L(G) ) = 0 :: ")
        print("\t\tInf-norm :: ",np.linalg.norm( self.intp.restrict_state( self.intp.lift_state( np.copy(self.settings.mmpar_g[:,self.params.mmpar_Nt]) )) 
                                -np.copy(self.settings.mmpar_g[:,self.params.mmpar_Nt]),np.Inf ))
        print("\t\t2-norm   :: ",np.linalg.norm( self.intp.restrict_state( self.intp.lift_state( np.copy(self.settings.mmpar_g[:,self.params.mmpar_Nt]) )) 
                                -np.copy(self.settings.mmpar_g[:,self.params.mmpar_Nt]),2 ))
        print("#" * 80)

        tend=time.time()-tt
        print('\tU0 took %0.2f seconds'%tend)

        
        print("\tStart Parareal iteration ::")

        self.parareal_iterarion(Dt)

        print("\tMicro-Macro Parareal run finished!")
        print("#" * 80)
        print("#" * 80)


    def parareal_iterarion(self,Dt):

        if self.params.mmpar_Kmax < self.params.mmpar_Nt:
            Kmax = self.params.mmpar_Kmax
        else:
            Kmax = self.params.mmpar_Nt

        for i in range(Kmax): 
            ii=i+1
            print("\tIteration :: ",ii)

            # fine run
            tt = time.time()            

            tfine = tt
           
            if self.params.freeze:
                np.random.seed(self.params.seed)
                for s in range(i):
                    for m in range(int(Dt/self.params.mmpar_dtf)):
                        out = np.random.rand(8192)

            for j in range(i,self.params.mmpar_Nt):

                T0=self.params.mmpar_T0+j*Dt
                TN=T0+Dt
                
                if self.params.mmpar_propf == 'RK1':
                    prop.RK1_step(
                            self.params,
                            self.fmethod,
                            self.settings.mmpar_Xf,
                            self.settings.mmpar_u[:,j],
                            self.settings.mmpar_f[:,j+1],
                            self.settings.mmpar_Uf[j,:,:],
                            self.settings.mmpar_Ef[:,j],
                            self.settings.mmpar_KEf[j,:,:],
                            T0,
                            TN,
                            self.params.mmpar_dtf)
                elif self.params.mmpar_propf == 'AB2':
                    prop.AB2_step(
                            self.params,
                            self.fmethod,
                            self.settings.mmpar_Xf,
                            self.settings.mmpar_u[:,j],
                            self.settings.mmpar_f[:,j+1],
                            self.settings.mmpar_Uf[j,:,:],
                            self.settings.mmpar_Ef[:,j],
                            self.settings.mmpar_KEf[j,:,:],
                            T0,
                            TN,
                            self.params.mmpar_dtf)
                elif self.params.mmpar_propf == 'RK2':
                    prop.RK2_step(
                            self.params,
                            self.fmethod,
                            self.settings.mmpar_Xf,
                            self.settings.mmpar_u[:,j],
                            self.settings.mmpar_f[:,j+1],
                            self.settings.mmpar_Uf[j,:,:],
                            self.settings.mmpar_Ef[:,j],
                            self.settings.mmpar_KEf[j,:,:],
                            T0,
                            TN,
                            self.params.mmpar_dtf)
                elif self.params.mmpar_propf == 'SRK2':
                    prop.SRK2_step(
                            self.params,
                            self.fmethod,
                            self.settings.mmpar_Xf,
                            self.settings.mmpar_u[:,j],
                            self.settings.mmpar_f[:,j+1],
                            self.settings.mmpar_Uf[j,:,:],
                            self.settings.mmpar_Ef[:,j],
                            self.settings.mmpar_KEf[j,:,:],
                            T0,
                            TN,
                            self.params.mmpar_dtf)
                elif self.params.mmpar_propf == 'RK4':
                    prop.RK4_step(
                            self.params,
                            self.fmethod,
                            self.settings.mmpar_Xf,
                            self.settings.mmpar_u[:,j],
                            self.settings.mmpar_f[:,j+1],
                            self.settings.mmpar_Uf[j,:,:],
                            self.settings.mmpar_Ef[:,j],
                            self.settings.mmpar_KEf[j,:,:],
                            T0,
                            TN,
                            self.params.mmpar_dtf)
                elif self.params.mmpar_propf == 'SRK4':
                    prop.SRK4_step(
                            self.params,
                            self.fmethod,
                            self.settings.mmpar_Xf,
                            self.settings.mmpar_u[:,j],
                            self.settings.mmpar_f[:,j+1],
                            self.settings.mmpar_Uf[j,:,:],
                            self.settings.mmpar_Ef[:,j],
                            self.settings.mmpar_KEf[j,:,:],
                            T0,
                            TN,
                            self.params.mmpar_dtf)
                else:
                    sys.exit("PARAREAL error :: Fine Propagator unknown!")

            # copy f to u in the kth slice
            self.settings.mmpar_u[:,ii] = np.copy(self.settings.mmpar_f[:,ii])
            self.settings.mmpar_E[:,i]  = np.copy(self.settings.mmpar_Ef[:,i])
            self.settings.mmpar_KE[i,:,:] = np.copy(self.settings.mmpar_KEf[i,:,:])

            tfineEnd=time.time()
            tfinepar=(tfineEnd-tfine)/(self.params.mmpar_Nt-i)
            print("\tFine run took %0.2f seconds."%(tfineEnd-tfine))
            print("\tPseudo Parallel execution :: %0.2f seconds."%tfinepar)

            print("\tStart update procedure.")

            tupdate=time.time()
            # update procedure 

            if self.params.freeze:
                np.random.seed(self.params.seed)
                for s in range(i+1):
                    for m in range(int(Dt/self.params.mmpar_dtf)):
                        out = np.random.rand(8192)

            for j in range(ii,self.params.mmpar_Nt):

                T0=self.params.mmpar_T0+j*Dt
                TN=T0+Dt

                if self.propc == "RK1":
                   prop.RK1_step(
                           self.params,
                           self.cmethod,
                           self.settings.mmpar_Xc,
                           self.intp.restrict_state(self.settings.mmpar_u[:,j]),
                           self.settings.mmpar_g1[:,j+1],
                           self.settings.mmpar_Ug1[j,:,:],
                           self.settings.mmpar_Eg1[:,j],
                           self.settings.mmpar_KEg1[j,:,:],
                           T0,
                           TN,
                           self.params.mmpar_dtc)
                elif self.propc == "AB2":
                   prop.AB2_step(
                           self.params,
                           self.cmethod,
                           self.settings.mmpar_Xc,
                           self.intp.restrict_state(self.settings.mmpar_u[:,j]),
                           self.settings.mmpar_g1[:,j+1],
                           self.settings.mmpar_Ug1[j,:,:],
                           self.settings.mmpar_Eg1[:,j],
                           self.settings.mmpar_KEg1[j,:,:],
                           T0,
                           TN,
                           self.params.mmpar_dtc)
                elif self.propc == "RK2":
                   prop.RK2_step(
                           self.params,
                           self.cmethod,
                           self.settings.mmpar_Xc,
                           self.intp.restrict_state(self.settings.mmpar_u[:,j]),
                           self.settings.mmpar_g1[:,j+1],
                           self.settings.mmpar_Ug1[j,:,:],
                           self.settings.mmpar_Eg1[:,j],
                           self.settings.mmpar_KEg1[j,:,:],
                           T0,
                           TN,
                           self.params.mmpar_dtc)
                elif self.propc == "SRK2":
                   prop.SRK2_step(
                           self.params,
                           self.cmethod,
                           self.settings.mmpar_Xc,
                           self.intp.restrict_state(self.settings.mmpar_u[:,j]),
                           self.settings.mmpar_g1[:,j+1],
                           self.settings.mmpar_Ug1[j,:,:],
                           self.settings.mmpar_Eg1[:,j],
                           self.settings.mmpar_KEg1[j,:,:],
                           T0,
                           TN,
                           self.params.mmpar_dtc)
                elif self.propc == "RK4":
                   prop.RK4_step(
                           self.params,
                           self.cmethod,
                           self.settings.mmpar_Xc,
                           self.intp.restrict(self.settings.mmpar_u[:,j]),
                           self.settings.mmpar_g1[:,j+1],
                           self.settings.mmpar_Ug1[j,:,:],
                           self.settings.mmpar_Eg1[:,j],
                           self.settings.mmpar_KEg1[j,:,:],
                           T0,
                           TN,
                           self.params.mmpar_dtc)
                elif self.propc == "SRK4":
                   prop.SRK4_step(
                           self.params,
                           self.cmethod,
                           self.settings.mmpar_Xc,
                           self.intp.restrict_state(self.settings.mmpar_u[:,j]),
                           self.settings.mmpar_g1[:,j+1],
                           self.settings.mmpar_Ug1[j,:,:],
                           self.settings.mmpar_Eg1[:,j],
                           self.settings.mmpar_KEg1[j,:,:],
                           T0,
                           TN,
                           self.params.mmpar_dtc)
                else:
                    print("Coarse Propagator for Parareal unknown!")
                    pass

                # Perform update
                if self.params.mmpar_alg == 1:
                    self.settings.mmpar_u[:,j+1] = np.copy( self.settings.mmpar_f[:,j+1]) +  \
                                                            self.intp.lift_state(np.copy(self.settings.mmpar_g1[:,j+1]) - \
                                                            np.copy(self.settings.mmpar_g[:,j+1]) \
                                                          )

                    # all velocities
                    if self.params.write:
                        for l in range(np.shape(self.settings.mmpar_Uu)[1]):
                            self.settings.mmpar_Uu[j,l,:] = np.copy(self.settings.mmpar_Uf[j,l,:]) + \
                                                                    self.intp.lift_state(np.copy(self.settings.mmpar_Ug1[j,l,:]) - \
                                                                    np.copy(self.settings.mmpar_Ug[j,l,:]) \
                                                                   )     
                else:
                    u_coarse                     = np.copy( self.settings.mmpar_g1[:,j+1] + \
                                                            self.intp.restrict_state(self.settings.mmpar_f[:,j+1]) - \
                                                            self.settings.mmpar_g[:,j+1] \
                                                          )
                    self.settings.mmpar_u[:,j+1] = np.copy( self.intp.lift_state(u_coarse) +  \
                                                            np.copy(self.settings.mmpar_f[:,j+1]) - \
                                                            self.intp.lift_state(self.intp.restrict_state(self.settings.mmpar_f[:,j+1])) \
                                                          )

                    # all velocities
                    if self.params.write:
                        for l in range(np.shape(self.settings.mmpar_Uu)[1]):
                            U_coarse                      = np.copy(self.intp.restrict_state(self.settings.mmpar_Uf[j,l,:]) + \
                                                                    np.copy(self.settings.mmpar_Ug1[j,l,:]) - \
                                                                    np.copy(self.settings.mmpar_Ug[j,l,:]) \
                                                                   )    
                            self.settings.mmpar_Uu[j,l,:] = np.copy(self.settings.mmpar_Uf[j,l,:]) + \
                                                                    self.intp.lift_state(U_coarse) - \
                                                                    self.intp.lift_state(self.intp.restrict_state(self.settings.mmpar_Uf[j,l,:]))     
                
                # Energy spectrum
                self.settings.mmpar_E[:,j] = np.copy(self.settings.mmpar_Ef[:,j])

                # Kinetic energy
                self.settings.mmpar_KE[j,:,:] = np.copy(self.settings.mmpar_KEg1[j,:,:]  + self.settings.mmpar_KEf[j,:,:] - self.settings.mmpar_KEg[j,:,:] )

                

            # output 

            self.settings.save_mmpar_iteration(ii)

            # replace Gk with Gk+1
            self.settings.mmpar_g   = np.copy(self.settings.mmpar_g1)
            self.settings.mmpar_Eg  = np.copy(self.settings.mmpar_Eg1)
            self.settings.mmpar_EKg = np.copy(self.settings.mmpar_KEg1)
            
            tendupdate=time.time()-tupdate
            print("\tUpdate procedure took %0.2f seconds."%tendupdate)

            tend=time.time()-tt
            print("\tIteration No ",ii," took %0.2f seconds."%tend)
            tendspt=tfinepar+tendupdate
            print("\tIteration No ",ii," took %0.2f seconds (pseudo wall time)."%tendspt)

            print("#" * 80)

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

