
import configparser

from dataclasses import dataclass, field

@dataclass
class PARAMS():

    # General Settings   

    noise : int     = field(init=False)
    noise_dim : int = field(init=False)
    k1 : float      = field(init=False)
    k2 : float      = field(init=False)
    init : bool     = field(init=False)
    path_init : str = field(init=False)
    nn : int        = field(init=False)
    Nt : int        = field(init=False)
    Tavg : int      = field(init=False)
    freeze : bool   = field(init=False)
    seed : int      = field(init=False)
    write : bool    = field(init=False)

    # Output Parameters

    path : str      = field(init=False)

    # Reference Parameters

    run_ref : bool  = field(init=False)
    N : int         = field(init=False)
    Tn : float      = field(init=False)
    T0 : float      = field(init=False)
    dt : float      = field(init=False)
    Cs : float      = field(init=False)
    prop : str      = field(init=False)
    smag : bool     = field(init=False)
    SDM : str       = field(init=False)
    ADV : str       = field(init=False)

    # Parareal Parameters

    run_par : bool      = field(init=False)

    par_Kmax : int      = field(init=False)

    par_N : int         = field(init=False)
    par_Nt : int        = field(init=False)
    par_Tn : float      = field(init=False)
    par_T0 : float      = field(init=False)
    par_nu : float      = field(init=False)

    par_propc : str     = field(init=False)
    par_dtc : float     = field(init=False)
    par_SDMc : str      = field(init=False)
    par_smagc : bool    = field(init=False)
    par_Csc : float     = field(init=False)
    par_ADVc : str      = field(init=False)

    par_propf : str     = field(init=False)
    par_dtf : float     = field(init=False)
    par_SDMf : str      = field(init=False)
    par_smagf : bool    = field(init=False)
    par_Csf : float     = field(init=False)
    par_ADVf : str      = field(init=False)

    # Micro-Macro Parareal Parameters

    run_mmpar : bool    = field(init=False)

    mmpar_Kmax : int    = field(init=False)
    mmpar_T0 : float    = field(init=False)
    mmpar_Tn : float    = field(init=False)
    mmpar_nu : float    = field(init=False)
    mmpar_Nt : int      = field(init=False)

    mmpar_nnc : int     = field(init=False)
    mmpar_nnf : int     = field(init=False)

    mmpar_Nc : int      = field(init=False)
    mmpar_propc : str   = field(init=False)
    mmpar_dtc : float   = field(init=False)
    mmpar_SDMc : str    = field(init=False)
    mmpar_smagc : bool  = field(init=False)
    mmpar_Csc : float   = field(init=False)
    mmpar_ADVc : str    = field(init=False)

    mmpar_Nf : int      = field(init=False)
    mmpar_propf : str   = field(init=False)
    mmpar_dtf : float   = field(init=False)
    mmpar_SDMf : str    = field(init=False)
    mmpar_smagf : bool  = field(init=False)
    mmpar_Csf : float   = field(init=False)
    mmpar_ADVf : str    = field(init=False)

    mmpar_L : str       = field(init=False)
    mmpar_R : str       = field(init=False)

    def __post_init__(self) -> None:

        config = configparser.ConfigParser()
        config.read('simulation.ini')

        for i in range(len(config.sections())):

            # General Settings   
            if config.sections()[i] == 'SETUP':
                self.noise      = int(config['SETUP']['noise_type'])
                self.k1         = float(config['SETUP']['k1'])
                self.k2         = float(config['SETUP']['k2'])

                self.path_init  = str(config['SETUP']['path_init'])
                self.nn         = int(config['SETUP']['nn'])
                self.Tavg       = float(config['SETUP']['Tavg'])
                self.Nt         = int(config['SETUP']['Nt'])

                self.noise_dim  = int(config['SETUP']['N'])

                if config['SETUP']['freeze'] == 'yes':
                    self.freeze     = True
                else:
                    self.freeze     = False

                self.seed       = int(config['SETUP']['seed'])

                if config['SETUP']['write'] == 'yes':
                    self.write      = True
                else:
                    self.write      = False

                if config['SETUP']['init'] == 'yes':
                    self.init       = True
                else:
                    self.init       = False
                    self.path_init  = str(config['SETUP']['path_init'])

            # Output Parameters
            elif config.sections()[i] == 'OUTPUT':
                self.path   = str(config['OUTPUT']['path'])

            # Reference Parameters
            elif config.sections()[i] == 'REFERENCE':

                if config['REFERENCE']['run_ref'] == 'yes':
                    self.run_ref = True
                else:
                    self.run_ref = False

                self.N      = int(config['REFERENCE']['N'])
                self.nu     = float(config['REFERENCE']['nu'])
                self.Tn     = float(config['REFERENCE']['Tn'])
                self.T0     = float(config['REFERENCE']['T0'])
                self.dt     = float(config['REFERENCE']['dt'])
                self.Cs     = float(config['REFERENCE']['Cs'])
           
                self.prop   = str(config['REFERENCE']['prop'])
                self.SDM    = str(config['REFERENCE']['SDM'])
                self.ADV    = str(config['REFERENCE']['ADV'])

                if config['REFERENCE']['smag'] == 'yes':
                    self.smag = True
                else:
                    self.smag = False


            # Parareal Parameters
            elif config.sections()[i] == 'PARAREAL':
                if config['PARAREAL']['run_par'] == 'yes':
                    self.run_par = True
                else:
                    self.run_par = False 
          
                self.par_Kmax   = int(config['PARAREAL']['Kmax'])
                self.par_N      = int(config['PARAREAL']['N'])
                self.par_Nt     = int(config['PARAREAL']['Nt'])

                self.par_T0     = float(config['PARAREAL']['T0'])
                self.par_Tn     = float(config['PARAREAL']['Tn'])
                self.par_nu     = float(config['PARAREAL']['nu'])

                self.par_propc  = str(config['PARAREAL']['propc'])
                self.par_SDMc   = str(config['PARAREAL']['SDMc'])
                if config['PARAREAL']['smagc'] == 'yes':
                    self.par_smagc = True 
                else:
                    self.par_smagc = False
                self.par_dtc    = float(config['PARAREAL']['dtc'])
                self.par_Csc    = float(config['PARAREAL']['Csc'])
                self.par_ADVc   = str(config['PARAREAL']['ADVc'])

                self.par_propf  = str(config['PARAREAL']['propf'])
                self.par_SDMf   = str(config['PARAREAL']['SDMf'])
                if config['PARAREAL']['smagf'] == 'yes':
                    self.par_smagf = True 
                else:
                    self.par_smagf = False
                self.par_dtf    = float(config['PARAREAL']['dtf'])
                self.par_Csf    = float(config['PARAREAL']['Csf'])
                self.par_ADVf   = str(config['PARAREAL']['ADVf'])


            # Micrp-Macro Parareal Parameters
            elif config.sections()[i] == 'MMPARAREAL':
                if config['MMPARAREAL']['run_mmpar'] == 'yes':
                    self.run_mmpar = True
                else:
                    self.run_mmpar = False 

                self.mmpar_alg      = int(config['MMPARAREAL']['alg'])

                self.mmpar_Kmax     = int(config['MMPARAREAL']['Kmax'])
                self.mmpar_Nc       = int(config['MMPARAREAL']['Nc'])
                self.mmpar_Nf       = int(config['MMPARAREAL']['Nf'])
                self.mmpar_Nt       = int(config['MMPARAREAL']['Nt'])

                self.mmpar_nnc      = int(config['MMPARAREAL']['nnc'])
                self.mmpar_nnf      = int(config['MMPARAREAL']['nnf'])

                self.mmpar_T0       = float(config['MMPARAREAL']['T0'])
                self.mmpar_Tn       = float(config['MMPARAREAL']['Tn'])
                self.mmpar_nu       = float(config['MMPARAREAL']['nu'])

                self.mmpar_propc    = str(config['MMPARAREAL']['propc'])
                self.mmpar_SDMc     = str(config['MMPARAREAL']['SDMc'])
                if config['MMPARAREAL']['smagc'] == 'yes':
                    self.mmpar_smagc = True 
                else:
                    self.mmpar_smagc = False
                self.mmpar_dtc      = float(config['MMPARAREAL']['dtc'])
                self.mmpar_Csc      = float(config['MMPARAREAL']['Csc'])
                self.mmpar_ADVc     = str(config['MMPARAREAL']['ADVc'])

                self.mmpar_propf    = str(config['MMPARAREAL']['propf'])
                self.mmpar_SDMf     = str(config['MMPARAREAL']['SDMf'])
                if config['MMPARAREAL']['smagf'] == 'yes':
                    self.mmpar_smagf = True 
                else:
                    self.mmpar_smagf = False
                self.mmpar_dtf      = float(config['MMPARAREAL']['dtf'])
                self.mmpar_Csf      = float(config['MMPARAREAL']['Csf'])
                self.mmpar_ADVf     = str(config['MMPARAREAL']['ADVf'])

                self.mmpar_L        = str(config['MMPARAREAL']['L'])
                self.mmpar_R        = str(config['MMPARAREAL']['R'])

