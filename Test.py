# -*- coding: utf-8 -*-

import sys, os
#sys.path.insert(0, "/path/to/PyEOC")
from PyEOC import *     # import PyEOC core class

Structure = PyEOC(
    'SBN',  # structure name included in the PyEOC class
    # measurement data: static  reflectivity vs angle (TE and TM)
    #                   dynamic reflectivity vs angle (TE and TM)
    #                   four files with tab-separated columns
    # 'SBN' data extracted from Cuniot-Ponsard et al. in JAP 109, 014107 (2011)
    'SBN_Reflectivity_TE.txt',      'SBN_Reflectivity_TM.txt',
    'SBN_Reflectivity_Dyn_TE.txt',  'SBN_Reflectivity_Dyn_TM.txt'
)

Structure.wavelength = 633  # laser wavelength in nm
Structure.voltage = 1.0     # applied voltage amplitude in volts

# the incident angle theta starting three values and range
# the choosen theta values should correspond to a "smooth" and "different" part of the DR and delta_R / delta_? data
Structure.theta_manual  = [35.0 * Structure.toradian, 40.0 * Structure.toradian, 45.0 * Structure.toradian]
Structure.thetaDelta    =   2.0 * Structure.toradian
Structure.thetaStart    =  30.0 * Structure.toradian
Structure.thetaEnd      =  70.0 * Structure.toradian

Structure.thickness[1] =  22.6  # Pt thickness (nm)
Structure.thickness[2] = 758    # SBN thickness (nm)
Structure.refractiveindexo[2] = 2.30 + 0.0515624j
Structure.refractiveindexe[2] = 2.26 + 0.0515624j

Structure.fit_dynamic = True    # fit the dynamic reflectivity?
Structure.fit(report = True)    # start fitting and report
Structure.plot()                # plot the fitted curves
#Structure.plotpoynting()        # plot the intensity profile
