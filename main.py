#!/usr/bin/env python3

import src.parser as param
import src.setupRun as setup
import src.FFT as FFT
import src.timeInt as solver
import src.FDM as FDM
import src.PAR as parareal
import src.MMPAR as mmparareal
import src.interpolation as intp

#import cProfile # used for estimating time-consuming tasks during execution
import numpy as np
import sys

def main() -> None:
    
    print("###############################################################################")
    print("###############################################################################")
    print("######## MMPAR Simulation for the Stochastic Burger's Equation ################")
    print("###############################################################################")
    print("###############################################################################")

    
    # Read simulation.ini file
    pars     = param.PARAMS()

    # Setup arrays and initial values 
    settings = setup.simulation_setup(pars)

    ########################
    #### REFERENCE RUN #####
    ########################

    if pars.run_ref:
        if pars.SDM == "FFT":
            ref_method = FFT.FFT(pars,'REFERENCE')
        elif pars.SDM == "FDM":
            ref_method = FDM.FDM(pars,'REFERENCE')
        else:
            sys.exit("Spatial Discretization Method for REFERENCE unknown!")
        print("\tExecuting Reference Run :: ")
        
        print("\t\t Spatial Discretization :: ",pars.SDM)
        print("\t\t Spatial Resolution   n :: ",pars.N)
        print("\t\t Start Time             :: ",pars.T0)
        print("\t\t End   Time             :: ",pars.Tn)
        print("\t\t Time step              :: ",pars.dt)
        print("\t\t Viscosity              :: ",pars.nu)
        print("\t\t Propagator             :: ",pars.prop)
        print("\t\t Smagorinsky Model      :: ",pars.smag)
        if pars.smag:
           print("\t\t Smagorinsky Constant   :: ",pars.Cs)
        print("\t\t Forcing fixed          :: ",pars.freeze)        
        if pars.freeze:
            np.random.seed(pars.seed)

        solver.solve(
                pars,
                ref_method,
                pars.prop,
                settings.Xr,
                settings.ur,
                settings.U,
                settings.Er,
                settings.KEr,
                pars.T0,
                pars.Tn,
                pars.dt,
                pars.Nt)

        settings.save_reference(str(pars.SDM)+"_"+str(pars.prop))
    else:
        print("\tNo Reference run exectued!")

    print("#" * 80)
    print("#" * 80)

    #########################
    ##### PARAREAL RUN ######
    #########################

    if pars.run_par:
        PAR = parareal.Parareal(pars,settings)
        PAR.runParareal()
    else:
        print("\tNo Parareal run exectued!")

    print("#" * 80)
    print("#" * 80)

    #########################
    ####### MMPAR RUN #######
    #########################
    
    if pars.run_mmpar:
        INTP  = intp.Interpolation(pars,settings)

        MMPAR = mmparareal.MMParareal(pars,settings,INTP)    
        MMPAR.runMMParareal()
    else:
        print("No Micro-Macro Parareal run exectued!")

    print("#" * 80)
    print("#" * 80)
   

if __name__ == "__main__":
    main()

