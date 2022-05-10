# -*- coding: utf-8 -*-

# Electro-optic coefficients calculation (PyEOC)
# Implementation of the method published by Cuniot-Ponsard et al. in JAP 109, 014107 (2011) -- https://doi.org/10.1063/1.3514083
# Code written by:
#   Pr. Sidi Hamady
#   Université de Lorraine, France
#   sidi.hamady@univ-lorraine.fr
# Released under the MIT license (https://opensource.org/licenses/MIT)
# See Copyright Notice in COPYRIGHT
# https://github.com/sidihamady/PyEOC
# Sidi Ould Saad Hamady, "PyEOC: a Python Package for Determination of the Electro-Optic Coefficients of Thin-Film Materials", 2022.

# -----------------------------------------------------------------------------------------------
# import S J Byrnes' code -- https://pypi.python.org/pypi/tmm -- https://arxiv.org/abs/1603.02720
import tmmCore as tmm

# -----------------------------------------------------------------------------------------------
# import standard modules
import sys, os, time
import math
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as pl
try:
    if sys.version_info[0] < 3:
        import Tkinter as Tk
    else:
        import tkinter as Tk
    # end if
except:
    pass
# -----------------------------------------------------------------------------------------------

class PyEOC(object):
    """ the PyEOC core class """

    # -------------------------------------------------------------------------------------------
    def __init__(self,
        structure = 'SBN',
        filename_RTE = '', filename_RTM = '',
        filename_RTEdyn = '', filename_RTMdyn = ''):

        """ PyEOC constructor """

        self.verbose = False

        self.knownstructures = ['SBN',]  # to complete by your own structures

        # theta angle range
        self.toradian   = np.pi / 180.0
        self.todegree   =  1.0 / self.toradian
        self.thetaStart = 20.0 * self.toradian
        self.thetaEnd   = 80.0 * self.toradian
        self.thetaDelta =  5.0 * self.toradian

        self.reset(structure, filename_RTE, filename_RTM, filename_RTEdyn, filename_RTMdyn)

        # number of lines to skip in each file (title and label lines for example)
        self.linestoskip = 2
        # data separator: '\t' for tabulation, by default
        self.separator = '\t'

        self.fontsize = 12

        self.parameters_count = len(self.layers) * 3
        self.parameters_maxcount = self.layers_maxcount * 3

        # laser wavelength in nm
        self.wavelength = 633.0
        # applied voltage amplitude in volts
        self.voltage = 1.0

        # set self.plotderiv to plot delta_R/delta_d, delta_R/delta_n and delta_R/delta_k
        self.plotderiv = True

        self.theta_manual = None
        self.theta_pos = [None, None]
        self.theta_val = [None, None]

        # variations in thickness (d) and refractive index (n and k)
        # TE ('s') polarization
        self.Ddo = None
        self.Dno = None
        self.Dko = None
        # TM ('p') polarization
        self.Dde = None
        self.Dne = None
        self.Dke = None
        self.d   = [None] * self.layers_maxcount
        self.no  = [None] * self.layers_maxcount
        self.ko  = [None] * self.layers_maxcount
        self.ne  = [None] * self.layers_maxcount
        self.ke  = [None] * self.layers_maxcount

        # p = TM, s = TE
        self.polarization = 'p'
        self.polarizationall = ['s', 'p']

        self.loaded = False

        self.totald  = 0.0
        self.totalda = 0.0

        self.fit_initTE = None
        self.fit_initTM = None
        self.fit_init   = [self.fit_initTE, self.fit_initTM]

        self.fitting = False
        self.fittingdyn = False
        self.evalcount = 0
        self.evalcountdyn = 0
        self.fitted = False

        self.fit_TE = None
        self.fit_TE_deriv  = None
        self.fit_TE_derivd = None
        self.fit_TE_derivn = None
        self.fit_TE_derivk = None

        self.fit_TEdyn = None

        self.fit_TM = None
        self.fit_TM_deriv  = None
        self.fit_TM_derivd = None
        self.fit_TM_derivn = None
        self.fit_TM_derivk = None

        self.fit_TMdyn = None

        self.fit_result = [self.fit_TE, self.fit_TM]

        self.fit_deriv  = [self.fit_TE_deriv,  self.fit_TM_deriv]
        self.fit_derivd = [self.fit_TE_derivd, self.fit_TM_derivd]
        self.fit_derivn = [self.fit_TE_derivn, self.fit_TM_derivn]
        self.fit_derivk = [self.fit_TE_derivk, self.fit_TM_derivk]

        self.fit_dyn = [self.fit_TEdyn, self.fit_TMdyn]

        # Fitting parameters
        self.fit_dynamic = True
        self.fit_derivtol = 1e-5
        self.fit_derivtol_min = 1e-12
        self.fit_tol = 1e-6
        self.fit_reltol = 1e-6
        self.fitdyn_tol = 1e-10
        self.fit_bounded = True
        self.fit_paramTE = None
        self.fit_paramTM = None
        self.fit_param = [self.fit_paramTE, self.fit_paramTM]
        self.fit_paramall = []
        self.fit_stddev = [1.0, 1.0]
        self.fit_mean = [1.0, 1.0]
        self.fitdyn_stddev = [1.0, 1.0]
        self.fitdyn_mean = [1.0, 1.0]
        self.inJac = False

        self.stopfilename = 'stop.txt'

    # end __init__

    # -------------------------------------------------------------------------------------------
    def reset(self, structure = 'SBN',
        filename_RTE = '', filename_RTM = '',
        filename_RTEdyn = '', filename_RTMdyn = ''):
        """ PyEOC constructor """

        self.structure = structure

        self.checked = False

        self.error = False
        self.message = None

        self.r13 = None
        self.d33 = None
        self.DkDV = None

        # to complete by your own structures
        if self.structure == 'SBN':

            # -----------------------------------------------------------------------------------
            # multilayers structure data
            # the default structure is the one published by Cuniot-Ponsard et al. in JAP 109, 014107 (2011)
            # complete and modify to suit your needs. In theory, the number of layers in not limited but in practice the fitting...
            # ... algorithm usually diverges for more than about seven layers (more or less, depending on the structure)
            self.title = 'SBN (reference)'
            self.layers = ['Pt', 'SBN', 'Pt', 'MgO']
            self.layers_count = len(self.layers)
            self.layers_maxcount = 7
            # the active layer position (beginning with zero, without counting air)
            self.active_layer = 1

            # -----------------------------------------------------------------------------------
            # the thickness and refractive index data are used as a starting point for the fitting algorithm
            self.thickness = [
                np.inf,                        # air
                22.6,                          # layer 1 (platinum)
                754.5,                         # layer 2 (SBN)
                70,                            # layer 3 (platinum)
                500000,                        # layer 4 (substrate)
                np.inf                         # air
            ]
            self.refractiveindexo = [
                1.0,                           # air
                2.33 + 4.14j,                  # layer 1 (platinum)
                2.3 + 0.0515j,                 # layer 2 (SBN)
                2.33 + 4.14j,                  # layer 3 (same than layer 1, platinum)
                1.7346 + 0.0j,                 # layer 4 (substrate)
                1.0                            # air
            ]
            self.refractiveindexe = [
                1.0,                           # air
                2.33 + 4.14j,                  # layer 1 (platinum)
                2.26 + 0.0515j,                # layer 2 (SBN)
                2.33 + 4.14j,                  # layer 3 (same than layer 1, platinum)
                1.7346 + 0.0j,                 # layer 4 (substrate)
                1.0                            # air
            ]

            # -----------------------------------------------------------------------------------
            # Parameters to include in the fitting procedure
            # set to False for parameters already fitted for a previous identical layer
            # (case of two layers of the same material)
            self.fit_includeparam = [
                    True,                      # layer 1 thickness (d), platinum
                    True,                      # layer 1 refractive index real part (n), platinum
                    True,                      # layer 1 refractive index real part (k), platinum
                    True,                      # layer 2 d -- active layer, SBN
                    True,                      # layer 2 n -- active layer, SBN
                    True,                      # layer 2 k -- active layer, SBN
                    True,                      # layer 3 d, platinum
                    False,                     # layer 3 n (same than layer 1, platinum)
                    False,                     # layer 3 k (same than layer 1, platinum)
                    False,                     # layer 4 d
                    False,                     # layer 4 n
                    False                      # layer 4 k
            ]

            # -----------------------------------------------------------------------------------
            # layers parameters upper and lower limits
            self.bounds = (
                [
                    21.0,                      # layer 1 lower thickness (d, in nm) limit, platinum
                    2.30,                      # layer 1 lower refractive index real part (n) limit, platinum
                    4.0,                       # layer 1 lower refractive index real part (k) limit, platinum
                    750.0,                     # layer 2 lower d limit -- active layer, SBN
                    2.24,                      # layer 2 lower n limit -- active layer, SBN
                    0.050,                     # layer 2 lower k limit -- active layer, SBN
                    68.0,                      # layer 3 lower d limit, platinum
                    2.30,                      # layer 3 lower n limit (same than layer 1, platinum)
                    4.0,                       # layer 3 lower k limit (same than layer 1, platinum)
                    490000.0,                  # layer 4 lower d limit, MgO
                    1.70,                      # layer 4 lower n limit, MgO
                    0.00000                    # layer 4 lower k limit, MgO
                ],
                [
                    23.0,                      # layer 1 upper thickness (d, in nm) limit, platinum
                    2.35,                      # layer 1 upper refractive index real part (n) limit, platinum
                    4.2,                       # layer 1 upper refractive index real part (k) limit, platinum
                    760.0,                     # layer 2 upper d limit -- active layer, SBN
                    2.32,                      # layer 2 upper n limit -- active layer, SBN
                    0.052,                     # layer 2 upper k limit -- active layer, SBN
                    72.0,                      # layer 3 upper d limit, platinum
                    2.35,                      # layer 3 upper n limit (same than layer 1, platinum)
                    4.2,                       # layer 3 upper k limit (same than layer 1, platinum)
                    510000.0,                  # layer 4 upper d limit, MgO
                    1.75,                      # layer 4 upper n limit, MgO
                    0.00001                    # layer 4 upper k limit, MgO
                ]
            )

            # -----------------------------------------------------------------------------------
            # coherency option (usually it is not necessay to switch to incoherent case)
            # up and bottom medium (usually air) should be set as incoherent
            self.coherency = [
                'i',                           # air
                'c',                           # layer 1, platinum
                'c',                           # layer 2, SBN
                'c',                           # layer 3, platinum
                'c',                           # layer 4, MgO
                'i'                            # air
            ]

            # -----------------------------------------------------------------------------------
            # the incident angle theta starting three values and range
            # the choosen theta values should correspond to a "smooth" and "different" part of the DR and delta_R / delta_? data
            self.theta_manual  = [35.0 * self.toradian, 40.0 * self.toradian, 45.0 * self.toradian]
            self.thetaDelta    =   2.0 * self.toradian
            self.thetaStart    =  30.0 * self.toradian
            self.thetaEnd      =  70.0 * self.toradian

        # end if

        # measurement data: static  reflectivity vs angle (TE and TM)
        #                   dynamic reflectivity vs angle (TE and TM)
        #                   four files with the two tab-separated columns each
        self.filenameTE     = filename_RTE
        self.filenameTM     = filename_RTM
        self.filenameTEdyn  = filename_RTEdyn
        self.filenameTMdyn  = filename_RTMdyn

        self.filter         = False

    # end reset

    # ---------------------------------------------------------------------------------------
    def check(self):

        if not (self.structure in self.knownstructures):
            self.error = True
            self.message = "------> unknown structure '%s'" % str(self.structure)
            self.disp("\n" + self.message)
            return False
        # end if

        if (self.layers_count > self.layers_maxcount):
            self.error = True
            self.message = "------> the number of layers exceeds the limit (%d)" % self.layers_maxcount
            self.disp("\n" + self.message)
            return False
        # end if

        if (self.layers_count < 2)                                              \
            or (len(self.thickness) != len(self.refractiveindexo))              \
            or (len(self.thickness) != len(self.refractiveindexe))              \
            or (len(self.thickness) != (self.layers_count + 2))                 \
            or (len(self.bounds[0]) != len(self.bounds[1]))                     \
            or (len(self.bounds[0]) != (self.layers_count * 3))                 \
            or (len(self.coherency) != (self.layers_count + 2)):
            self.error = True
            self.message = "------> layers parameters not consistent"
            self.disp("\n" + self.message)
            return False
        # end if

        # for the default structure, get the embedded data if necessary
        if (self.structure == 'SBN') and (not os.path.isfile(self.filenameTE) or not os.path.isfile(self.filenameTEdyn) or not os.path.isfile(self.filenameTM) or not os.path.isfile(self.filenameTMdyn)):
            try:
                eocdir = os.path.dirname(os.path.realpath(__file__))
                # measurement data: static  reflectivity vs angle (TE and TM)
                #                   dynamic reflectivity vs angle (TE and TM)
                #                   four files with the two tab-separated columns each
                self.filenameTE     = os.path.join(eocdir, 'SBN_Reflectivity_TE.txt')
                self.filenameTM     = os.path.join(eocdir, 'SBN_Reflectivity_TM.txt')
                self.filenameTEdyn  = os.path.join(eocdir, 'SBN_Reflectivity_Dyn_TE.txt')
                self.filenameTMdyn  = os.path.join(eocdir, 'SBN_Reflectivity_Dyn_TM.txt')
            except:
                self.filenameTE = ''
                self.filenameTM = ''
                self.filenameTEdyn = ''
                self.filenameTMdyn = ''
                pass
            # end try
        # end if
        
        for fname in [self.filenameTE, self.filenameTM, self.filenameTEdyn, self.filenameTMdyn]:
            if not os.path.isfile(fname):
                self.error = True
                self.message = "------> data file '%s' not found" % ('?' if (fname == '') else fname)
                self.disp("\n" + self.message)
                return False
            # end if
        # end for

        self.min_indexn = 0.001
        self.max_indexn = 10.0
        self.min_indexk = 0.0
        self.max_indexk = 10.0
        self.min_thickness = 1.0
        self.max_thickness = 1000000.0
        for ii in range(0, self.layers_count - 1):
            # check layer parameters consistency
            ll = 3 * ii
            if (self.thickness[ii+1] < self.min_thickness) or (self.thickness[ii+1] > self.max_thickness)                                       \
               or (self.bounds[0][ll] < self.min_thickness) or (self.bounds[1][ll] > self.max_thickness)                                        \
               or (self.thickness[ii+1] < self.bounds[0][ll]) or (self.thickness[ii+1] > self.bounds[1][ll]):
                self.error = True
                self.message = "------> layer %d invalid thickness value / bounds: %g / [%g, %g]" % (ii + 1, self.thickness[ii+1], self.bounds[0][ll], self.bounds[1][ll])
                self.disp("\n" + self.message)
                return False
            # end if
            if    (self.bounds[0][ll+1] < self.min_indexn) or (self.bounds[1][ll+1] > self.max_indexn)                                          \
               or (self.bounds[0][ll+2] < self.min_indexk) or (self.bounds[1][ll+2] > self.max_indexk)                                          \
               or (self.refractiveindexo[ii+1].real < self.min_indexn) or (self.refractiveindexo[ii+1].real > self.max_indexn)                  \
               or (self.refractiveindexo[ii+1].real < self.bounds[0][ll+1]) or (self.refractiveindexo[ii+1].real > self.bounds[1][ll+1])        \
               or (self.refractiveindexo[ii+1].imag < self.min_indexk) or (self.refractiveindexo[ii+1].imag > self.max_indexk)                  \
               or (self.refractiveindexo[ii+1].imag < self.bounds[0][ll+2]) or (self.refractiveindexo[ii+1].imag > self.bounds[1][ll+2])        \
               or (self.refractiveindexe[ii+1].real < self.min_indexn) or (self.refractiveindexe[ii+1].real > self.max_indexn)                  \
               or (self.refractiveindexe[ii+1].real < self.bounds[0][ll+1]) or (self.refractiveindexe[ii+1].real > self.bounds[1][ll+1])        \
               or (self.refractiveindexe[ii+1].imag < self.min_indexk) or (self.refractiveindexe[ii+1].imag > self.max_indexk)                  \
               or (self.refractiveindexe[ii+1].imag < self.bounds[0][ll+2]) or (self.refractiveindexe[ii+1].imag > self.bounds[1][ll+2]):
                self.error = True
                self.message = "------> layer %d invalid index value / bounds" % (ii + 1)
                self.disp("\n" + self.message)
                return False
            # end if
        # end for

        self.update()

        self.checked = True
        return self.checked

    # end check

    # ---------------------------------------------------------------------------------------
    def update(self):

        for ii in range(0, self.layers_count - 1):
            # layers with same name should have the same refractive index
            for rr in range(ii+1, self.layers_count):
                if (self.layers[ii] == self.layers[rr]):
                    self.refractiveindexo[1+rr] = self.refractiveindexo[1+ii]
                    self.refractiveindexe[1+rr] = self.refractiveindexe[1+ii]
                # end if
            # end for
        # end for

        # update layers parameters (d, n, k)
        self.totald = 0.0
        self.totalda = 0.0
        for ii in range(0, self.layers_count):
            self.d[ii]  = self.thickness[1+ii]
            self.no[ii] = self.refractiveindexo[1+ii].real
            self.ko[ii] = self.refractiveindexo[1+ii].imag
            self.ne[ii] = self.refractiveindexe[1+ii].real
            self.ke[ii] = self.refractiveindexe[1+ii].imag
            self.totald += self.d[ii]
            if (ii <= self.active_layer):
                self.totalda += self.d[ii]
            # end if
        # end for

        # set the initial parameters for the fitting algorithm
        self.fit_init[0] = []
        self.fit_init[1] = []
        for rr in range(0, len(self.d)):
            if self.d[rr] == None:
                break
            # end if
            self.fit_init[0].append(self.d[rr])
            self.fit_init[0].append(self.no[rr])
            self.fit_init[0].append(self.ko[rr])
            self.fit_init[1].append(self.d[rr])
            self.fit_init[1].append(self.ne[rr])
            self.fit_init[1].append(self.ke[rr])
        # end for

    # end update

    # ---------------------------------------------------------------------------------------
    def disp(self, strT):
        if (self.verbose or self.error) and (not self.inJac):
            print(strT)
        # end if
    # end disp

    # ---------------------------------------------------------------------------------------
    # utility function : update the parameters (d, n, k)
    def updateParametersArray(self, dnk_array):

        ipol = 0 if (self.polarization == 's') else 1

        # some basic check
        argc = len(dnk_array)
        if argc > self.parameters_maxcount:
            self.error = True
            self.message = "------> invalid number of arguments given to 'updateParameters'"
            self.disp("\n" + self.message)
            return
        # end if

        # construct the parameters list
        params_tmp = []
        ll = 0
        for rr in range(0, len(self.fit_init[ipol])):
            if self.fit_includeparam[rr]:
                params_tmp.append(dnk_array[ll])
                ll += 1
            else:
                params_tmp.append(self.fit_init[ipol][rr])
            # end if
        # end for

        # construct the thickness and refractive index list
        ll = 0
        for rr in range(0, len(params_tmp), 3):
            self.thickness[1+ll] = params_tmp[rr]
            if ipol == 0:
                self.refractiveindexo[1+ll] = params_tmp[rr+1] + 1j * params_tmp[rr+2]
            else:
                self.refractiveindexe[1+ll] = params_tmp[rr+1] + 1j * params_tmp[rr+2]
            # end if
            ll += 1
        # end for

        # layers with same name should have the same refractive index...
        # ... example : top and bottom Pt contact
        for ii in range(0, self.layers_count - 1):
            for rr in range(ii+1, self.layers_count):
                if (self.layers[ii] == self.layers[rr]):
                    if ipol == 0:
                        self.refractiveindexo[1+rr] = self.refractiveindexo[1+ii]
                    else:
                        self.refractiveindexe[1+rr] = self.refractiveindexe[1+ii]
                    # end if
                # end if
            # end for
        # end for

    # end updateParametersArray

    # ---------------------------------------------------------------------------------------
    # utility function : update the parameters (d, n, k)
    def updateParameters(self, *dnk_args):

        self.updateParametersArray(dnk_args)

    # end updateParameters

    # ---------------------------------------------------------------------------------------
    # calculate the reflectivity vs angle, using the S J byrnes' code
    def fitfunc_static_array(self, theta, dnk_array):

        if self.fitting:
            self.evalcount += 1
        # end if

        if os.path.isfile(self.stopfilename):
            try:
                os.unlink(self.stopfilename)
            except:
                pass
            # end try
            print("\n------> fitting stopped by the user after %d iterations" % (self.evalcount))
            exit(1)
        # end if

        isArray = isinstance(theta, (list, tuple, np.ndarray))

        self.updateParametersArray(dnk_array)

        n_list = np.array(self.refractiveindexo if (self.polarization == 's') else self.refractiveindexe, dtype=complex)
        d_list = np.array(self.thickness, dtype=float)

        if not isArray:
            return tmm.inc_tmm(self.polarization, n_list, d_list, self.coherency, theta, self.wavelength)['R']        # end if

        reflectivity = []
        for th in theta:
            reflectivity.append(tmm.inc_tmm(self.polarization, n_list, d_list, self.coherency, th, self.wavelength)['R'])
        # end for

        return reflectivity

    # end self.fitfunc_static_array

    # ---------------------------------------------------------------------------------------
    # calculate the reflectivity vs angle, using the S J byrnes' code
    def fitfunc_static(self, theta, *dnk_args):

        return self.fitfunc_static_array(theta, dnk_args)

    # end fitfunc_static

    # ---------------------------------------------------------------------------------------
    def fitfunc_static_jac(self, theta, *dnk_args):
        self.inJac = True
        R0 = np.array(self.fitfunc_static_array(theta, dnk_args))
        dnk = []
        parcount = len(dnk_args)
        for ii in range(0, parcount):
            dnk.append(dnk_args[ii])
        # end for
        jac = np.array([[]])
        for ii in range(0, parcount):
            dnk0 = dnk[ii]
            dpar = (self.fit_derivtol * math.fabs(dnk0))
            if dpar < self.fit_derivtol_min:
                dpar = self.fit_derivtol_min
            # end if
            dnk[ii] = dnk0 + dpar
            RP = np.array(self.fitfunc_static_array(theta, dnk))
            if ii == 0:
                jac = (RP - R0) / dpar
            else:
                jac = np.vstack((jac, (RP - R0) / dpar))
            # end if
            dnk[ii] = dnk0
        # end for
        self.inJac = False
        return jac.transpose()
    # end fitfunc_static_jac

    # ---------------------------------------------------------------------------------------
    # calculate the intensity vs position in the structure, using the S J byrnes' code
    def poynting(self, theta, position, *dnk_args):

        self.updateParameters(*dnk_args)

        n_list = np.array(self.refractiveindexo if (self.polarization == 's') else self.refractiveindexe, dtype=complex)
        d_list = np.array(self.thickness, dtype=float)

        tmmdata = tmm.coh_tmm(self.polarization, n_list, d_list, theta, self.wavelength)

        poyn = []
        for pos in position:
            layer, pos_in_layer = tmm.find_in_structure_with_inf(d_list, pos)
            data = tmm.position_resolved(layer, pos_in_layer, tmmdata)
            poyn.append(data['poyn'])
        # end for
        poyn = np.array(poyn)

        return poyn

    # end poynting

    # ---------------------------------------------------------------------------------------
    # calculate the reflectivity vs angle...
    # ... for the two polarizations 's' and 'p'
    def calculate(self, report = False, plot = False):

        if not self.checked:
            self.check()
        # end if

        # load the experimental data and set the angle range
        self.load()

        if self.error:
            return
        # end if

        # setlayers (d,n,k)
        self.fit_param[0] = []
        self.fit_param[1] = []
        for rr in range(0, len(self.fit_init[0])):
           self.fit_param[0].append(self.fit_init[0][rr])
           self.fit_param[1].append(self.fit_init[1][rr])
        # end for

        self.calc_reflectivity(ipol = 0)
        self.calc_reflectivity(ipol = 1)
        if self.error:
            return
        # end if

        # calculate the derivatives delta_R/delta_d, delta_R/delta_n, delta_R/delta_k
        self.calc_deriv(ipol = 0)
        self.calc_deriv(ipol = 1)

        # calculate DR
        self.calc_dynamic(ipol = 0)
        self.calc_dynamic(ipol = 1)

        self.disp("\n------> parameters (d en nm) :")
        ll = 0
        for rr in range(0, self.parameters_count, 3):
            self.disp("\n------> %7s (Layer %d): d, (n, k)   :   %g, (%g + j%g, %g + j%g)" % (self.layers[ll], ll+1, self.fit_init[0][rr], self.fit_init[0][rr+1], self.fit_init[0][rr+2], self.fit_init[1][rr+1], self.fit_init[1][rr+2]))
            ll += 1
        # end if

        if self.r13 and self.d33:
            self.disp("\n------> r13 = %g pm/V   d33 = %g pm/V   Dk/DV = %g 1/V" % (self.r13, self.d33, self.DkDV))
        # end if

        if report:
            self.report()
        # end if

        if plot:
            self.plot()
        # end if

    # end calculate

    # ---------------------------------------------------------------------------------------
    # plot the intensity profile in the structure
    def plotpoynting(self):

        if not self.checked:
            self.check()
        # end if

        # plot intensity vs position
        figT = pl.figure(num=1, figsize=(12,6), facecolor='#FFFFFF', linewidth=1.0, frameon=True)
        figT.canvas.set_window_title(self.structure + ': Poynting intensity profile in the structure')
        tManager = pl.get_current_fig_manager()
        img = Tk.PhotoImage(file = os.path.dirname(os.path.abspath(__file__)) + '/iconmain.gif')
        tManager.window.tk.call('wm', 'iconphoto', tManager.window._w, img)

        ylim = [None, None]

        for polarization in ['s', 'p']:

            # calculate intensity
            self.polarization = polarization
            position = np.linspace(-100, self.totalda + 100, num=1001)
            nx = self.no if (self.polarization == 's') else self.ne
            kx = self.ko if (self.polarization == 's') else self.ke
            poyn = self.poynting(0.0, position,
                self.d[0], nx[0], kx[0],
                self.d[1], nx[1], kx[1],
                self.d[2], nx[2], kx[2],
                self.d[3], nx[3], kx[3],
                self.d[4], nx[4], kx[4],
                self.d[5], nx[5], kx[5],
                self.d[6], nx[6], kx[6])

            psp = [221, 222] if polarization == 's' else [223, 224]

            for pp in range(0,2):

                splot = pl.subplot(psp[pp])
                TX = 'TE' if polarization == 's' else 'TM'

                if pp == 1:
                    iT = np.where(position >= self.thickness[1])
                    positionT = np.take(position, iT, axis=0).reshape(-1)
                    poynT = np.take(poyn, iT, axis=0).reshape(-1)
                else:
                    positionT = position
                    poynT = poyn
                # end if

                pl.plot(positionT, poynT, 'b-' if polarization == 'p' else 'r-', linewidth=2.5, zorder=3)
                pl.tick_params(axis='x', which='major', labelsize=self.fontsize+2)
                pl.tick_params(axis='y', which='major', labelsize=self.fontsize+2)

                if pp == 1:
                    pl.ylim(0., 10. ** math.ceil(math.log10(np.max(poynT))))
                # end if

                tgca = pl.gca()
                if polarization == 's':
                    tgca.set_xticklabels([])
                # end if
                if polarization == 'p':
                    pl.xlabel('Position in the structure (nm)', fontsize=self.fontsize+2)
                if pp == 0:
                    pl.ylabel('%s intensity profile' % TX, fontsize=self.fontsize+2)
                # end if
                    
                Nlayers = len(self.thickness) - 3 # exclude top and bottom air, and the thick substrate
                if (Nlayers >= 2):
                    xlp = 0.
                    xl = self.thickness[1]
                    layercolors = ['#F7CD72', '#B1E8F0', '#F2BB44']
                    for ii in range(2, Nlayers + 2):
                        pl.axvline(xl, 0., 1., color='#BA9634', linestyle=':', zorder=2)
                        pl.axvspan(xlp, xl, alpha=0.5, color=layercolors[ii-2] if (ii - 2) < 3 else layercolors[2])
                        xlp = xl
                        xl += self.thickness[ii]
                    # end for
                # end if

                pl.xlim(0., self.totalda + self.thickness[-3])

                # end if
                if polarization == 's':
                    ylim[pp] = pl.ylim()
                else:
                    pl.ylim(ylim[pp])
                # end if
                splot.set_ylim(ymin=0.)
                pl.grid()
        
            # end for

        # end for

        pl.subplots_adjust(left=0.1, bottom=0.1, right=0.96, top=0.94, wspace=0.4, hspace=0.4)
        figT.tight_layout(h_pad=2, w_pad=4)
        pl.show()

    # end plotpoynting

    # ---------------------------------------------------------------------------------------
    # load the experimental data, filter and interpolate
    def load(self):

        if not self.checked:
            self.check()
        # end 

        if self.verbose:
            self.message = "------> loading the experimental data..."
            self.disp("\n" + self.message)
        # end if

        try:
            # load the experimental data (reflectivity vs angle (in degree))
            dataTE = np.loadtxt(self.filenameTE, delimiter=self.separator, skiprows=self.linestoskip, usecols=(0,1))
            dataTEdyn = np.loadtxt(self.filenameTEdyn, delimiter=self.separator, skiprows=self.linestoskip, usecols=(0,1))
            dataTM = np.loadtxt(self.filenameTM, delimiter=self.separator, skiprows=self.linestoskip, usecols=(0,1))
            dataTMdyn = np.loadtxt(self.filenameTMdyn, delimiter=self.separator, skiprows=self.linestoskip, usecols=(0,1))
        except Exception as excT:
            excType, excObj, excTb = sys.exc_info()
            excFile = os.path.split(excTb.tb_frame.f_code.co_filename)[1]
            self.message  = "------> error: the experimental data cannot be loaded:\n        %s\n        in %s (line %d)\n" % (str(excT), excFile, excTb.tb_lineno)
            self.error = True
            self.disp("\n" + self.message)
            return
        # end try

        self.dataTEX = dataTE[:,0]*self.toradian
        self.dataTEY = dataTE[:,1]
        self.dataTEXdyn = dataTEdyn[:,0]*self.toradian
        self.dataTEYdyn = dataTEdyn[:,1]
        dataTEXpoints = len(self.dataTEX)
        dataTEYpoints = len(self.dataTEY)
        dataTEXdynpoints = len(self.dataTEX)
        dataTEYdynpoints = len(self.dataTEY)
        if ((dataTEXpoints < 20) or (dataTEYpoints < 20) or (dataTEXpoints > 10000) or (dataTEYpoints > 10000)
            or (dataTEXdynpoints < 20) or (dataTEYdynpoints < 20) or (dataTEXdynpoints > 10000) or (dataTEYdynpoints > 10000)
            or (dataTEXpoints != dataTEYpoints) or (dataTEXdynpoints != dataTEYdynpoints)):
            self.error = True
            self.message = "------> error: data size not consistent : [%d %d %d %d]" % (dataTEXpoints, dataTEYpoints, dataTEXdynpoints, dataTEYdynpoints)
            self.disp("\n" + self.message)
            return
        # end if

        self.dataTMX = dataTM[:,0]*self.toradian
        self.dataTMY = dataTM[:,1]
        self.dataTMXdyn = dataTMdyn[:,0]*self.toradian
        self.dataTMYdyn = dataTMdyn[:,1]

        arThetaStart = [self.dataTEX[0], self.dataTEXdyn[0], self.dataTMX[0], self.dataTMXdyn[0]]
        arThetaEnd = [self.dataTEX[len(self.dataTEX) - 1], self.dataTEXdyn[len(self.dataTEXdyn) - 1], self.dataTMX[len(self.dataTMX) - 1], self.dataTMXdyn[len(self.dataTMXdyn) - 1]]
        for ii in range(0, len(arThetaStart)):
            if (self.thetaStart < arThetaStart[ii]):
                self.thetaStart = arThetaStart[ii]
            # end if
            if (self.thetaEnd > arThetaEnd[ii]):
                self.thetaEnd = arThetaEnd[ii]
            # end if
        # end for
        if self.thetaStart >= (self.thetaEnd - 3.0*self.thetaDelta):
            self.error = True
            self.message = "------> error: angle values not consistent : [%g %g]" % (self.thetaStart*self.todegree, self.thetaEnd*self.todegree)
            self.disp("\n" + self.message)
            return
        # end if
        if (self.theta_manual) and (isinstance(self.theta_manual, (list, tuple, np.ndarray))) and (len(self.theta_manual) == 3):
            self.theta_val[0] = []
            self.theta_val[1] = []
            for rr in range(0, len(self.theta_manual)):
                self.theta_val[0].append(self.theta_manual[rr])
                self.theta_val[1].append(self.theta_manual[rr])
            # end for
        # end if

        if self.filter:
            # filter the experimental data using the Savitzky-Golay algorithm
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
            # the window length (savgol_window) and other parameters should be chosen carefully to preserve the experimental data
            savgol_window = 5
            if (savgol_window % 2) == 0:
                savgol_window += 1
            # end if
            if savgol_window > 51:
                savgol_window = 51
            savgol_order = 2
            self.dataTEY = savgol_filter(self.dataTEY, savgol_window, savgol_order)
            self.dataTEYdyn = savgol_filter(self.dataTEYdyn, savgol_window, savgol_order)
            self.dataTMY = savgol_filter(self.dataTMY, savgol_window, savgol_order)
            self.dataTMYdyn = savgol_filter(self.dataTMYdyn, savgol_window, savgol_order)
        # end if self.filter

        self.dataTX = [self.dataTEX, self.dataTMX]
        self.dataTY = [self.dataTEY, self.dataTMY]
        self.rchisq = np.zeros(2)
        # the measurement error for the static reflectivity (0.5% of the measured value),
        # to adapt considering the uncertainty on the measured data
        self.measurerr = np.array([0.005, 0.005])

        self.dataTXdyn = [self.dataTEXdyn, self.dataTMXdyn]
        self.dataTYdyn = [self.dataTEYdyn, self.dataTMYdyn]
        self.rchisqDyn = np.zeros(2)
        # the measurement error for the dynamic reflectivity (10% of the measured value),
        # to adapt considering the uncertainty on the measured data
        self.measurerrDyn = np.array([0.10, 0.10])

        if self.verbose:
            self.message = "------> data loaded: [%d %d %d %d] points" % (dataTEXpoints, dataTEYpoints, dataTEXdynpoints, dataTEYdynpoints)
            self.disp("\n" + self.message)
        # end if

        self.loaded = True

    # end load

    def reducedChiSquared(self, dataY, dataYmodel, dataYerr, Nparams):
        try:
            rchis = np.sum(((dataY - dataYmodel) / dataYerr) ** 2.) / (len(dataY) - Nparams)
            return rchis
        except:
            return 0.
        # end try
    # end reducedChiSquared

    # ---------------------------------------------------------------------------------------
    # fit the experimental static reflectivity to get (d,n,k) for each layer
    def fit_static(self, ipol):

        if not self.checked:
            self.check()
        # end if

        if not self.loaded:
            self.load()
        # end if

        if self.error:
            return
        # end if

        # perform the fitting for static reflectivity for s and p polarizations

        self.polarization = self.polarizationall[ipol]

        init_tmp = []
        bounds_tmp = ([], [])
        for rr in range(0, len(self.fit_init[ipol])):
            if self.fit_includeparam[rr]:
                init_tmp.append(self.fit_init[ipol][rr])
                bounds_tmp[0].append(self.bounds[0][rr])
                bounds_tmp[1].append(self.bounds[1][rr])
            # end if
        # end for

        self.fitting = True
        self.evalcount = 0

        starttime = self.getclock()
        self.disp("\n------> static fitting ('%s' polarization) in progress..." % (self.polarization))

        try:

            self.fit_param[ipol] = np.array(self.fit_init[ipol])

            result_tmp,covar = curve_fit(self.fitfunc_static, self.dataTX[ipol], self.dataTY[ipol], p0=init_tmp, bounds=bounds_tmp, check_finite=True, method='trf', xtol=self.fit_tol, ftol=self.fit_tol, gtol=self.fit_tol, jac=self.fitfunc_static_jac)
            self.rchisq[ipol] = self.reducedChiSquared(self.dataTY[ipol], self.fitfunc_static(self.dataTX[ipol], *result_tmp), self.measurerr[ipol] * self.dataTY[ipol], len(result_tmp))

            self.fit_param[ipol] = []
            ll = 0
            for rr in range(0, len(self.fit_init[ipol])):
                if self.fit_includeparam[rr]:
                    self.fit_param[ipol].append(result_tmp[ll])
                    ll += 1
                else:
                    self.fit_param[ipol].append(self.fit_init[ipol][rr])
                # end if
            # end for rr

            # layers with same name should have the same refractive index
            ll = 0
            mm = 0
            for nn in range(0, self.layers_count - 1):
                mm = ll + 3
                for rr in range(nn+1, self.layers_count):
                    if (self.layers[nn] == self.layers[rr]):
                        self.fit_param[ipol][mm+1] = self.fit_param[ipol][ll+1]
                        self.fit_param[ipol][mm+2] = self.fit_param[ipol][ll+2]
                    # end if
                    mm += 3
                # end for rr
                ll += 3
            # end for nn

            self.fit_param[ipol] = np.array(self.fit_param[ipol])
            self.fitted = True
            self.fitting = False

        except ValueError:
            self.error = True
            self.message = "------> static fitting ('%s' polarization) ended (duration : %g seconds)\ndata not valid" % (self.polarization, self.getclock() - starttime)
            self.disp("\n" + self.message)
            self.fit_param[ipol] = []
            for rr in range(0, len(self.fit_init[ipol])):
                self.fit_param[ipol].append(self.fit_init[ipol][rr])
            # end for rr
            self.fitting = False

        except RuntimeError:
            self.error = True
            self.message = "------> static fitting ('%s' polarization) ended (duration : %g seconds)\nconvergence not achieved" % (self.polarization, self.getclock() - starttime)
            self.disp("\n" + self.message)
            self.fit_param[ipol] = []
            for rr in range(0, len(self.fit_init[ipol])):
                self.fit_param[ipol].append(self.fit_init[ipol][rr])
            # end for rr
            self.fitting = False

        else:
            self.disp("\n------> static fitting ('%s' polarization) ended (duration : %g seconds)\nnumber of evaluations: %d" % (self.polarization, self.getclock() - starttime, self.evalcount))

        # end try

    # end fit_static

    # ---------------------------------------------------------------------------------------
    # calculate static reflectivity vs theta
    def calc_reflectivity(self, ipol):

        if self.error:
            return
        # end if

        self.polarization = self.polarizationall[ipol]

        icount = len(self.fit_param[ipol])
        if (icount < self.parameters_maxcount):
            for rr in range(icount, self.parameters_maxcount):
                self.fit_param[ipol] = np.append(self.fit_param[ipol], None)
            # end for
        # end if

        self.fit_result[ipol] = self.fitfunc_static(self.dataTX[ipol],
            self.fit_param[ipol][ 0], self.fit_param[ipol][ 1], self.fit_param[ipol][ 2],
            self.fit_param[ipol][ 3], self.fit_param[ipol][ 4], self.fit_param[ipol][ 5],
            self.fit_param[ipol][ 6], self.fit_param[ipol][ 7], self.fit_param[ipol][ 8],
            self.fit_param[ipol][ 9], self.fit_param[ipol][10], self.fit_param[ipol][11],
            self.fit_param[ipol][12], self.fit_param[ipol][13], self.fit_param[ipol][14],
            self.fit_param[ipol][15], self.fit_param[ipol][16], self.fit_param[ipol][17],
            self.fit_param[ipol][18], self.fit_param[ipol][19], self.fit_param[ipol][20])

        stddev = 0.0
        rmoy = 0.0
        for rr in range(0, len(self.dataTX[ipol])):
            stddev += (self.fit_result[ipol][rr] - self.dataTY[ipol][rr]) * (self.fit_result[ipol][rr] - self.dataTY[ipol][rr])
            rmoy += self.dataTY[ipol][rr]
        # end for
        self.fit_paramall.append(self.fit_param[ipol])
        self.fit_stddev[ipol] = np.sqrt(stddev)
        self.fit_mean[ipol] = rmoy / float(len(self.dataTY[ipol]))
        if not self.fitting:
            self.disp("\n------> %s: static stddev = %g   mean = %g" % ("TE" if ipol == 0 else "TM", self.fit_stddev[ipol], self.fit_mean[ipol]))
        # end if

    # end calc_reflectivity

    # ---------------------------------------------------------------------------------------
    # calculate the reflectivity derivatives : delta_R/delta_d, delta_R/delta_n, delta_R/delta_k ...
    # ... using forward finite difference formula : (R[i+1] - R[i]) / delta
    def calc_deriv(self, ipol):

        if self.error:
            return
        # end if

        iparA = self.active_layer * 3

        self.polarization = self.polarizationall[ipol]

        self.fit_result[ipol] = self.fitfunc_static(self.dataTX[ipol],
            self.fit_param[ipol][ 0], self.fit_param[ipol][ 1], self.fit_param[ipol][ 2],
            self.fit_param[ipol][ 3], self.fit_param[ipol][ 4], self.fit_param[ipol][ 5],
            self.fit_param[ipol][ 6], self.fit_param[ipol][ 7], self.fit_param[ipol][ 8],
            self.fit_param[ipol][ 9], self.fit_param[ipol][10], self.fit_param[ipol][11],
            self.fit_param[ipol][12], self.fit_param[ipol][13], self.fit_param[ipol][14],
            self.fit_param[ipol][15], self.fit_param[ipol][16], self.fit_param[ipol][17],
            self.fit_param[ipol][18], self.fit_param[ipol][19], self.fit_param[ipol][20])

        self.fit_derivd[ipol] = []
        self.fit_derivn[ipol] = []
        self.fit_derivk[ipol] = []
        self.fit_deriv[ipol] = [self.fit_derivd[ipol], self.fit_derivn[ipol], self.fit_derivk[ipol]]

        for ipar in range(iparA, iparA + 3):

            # save parameter value
            parambak = self.fit_param[ipol][ipar]

            dpar = (self.fit_derivtol * math.fabs(parambak))
            if dpar < self.fit_derivtol_min:
                dpar = self.fit_derivtol_min
            # end if

            # use the forward formula, by default
            derivforward_TE = None
            derivforward_TM = None
            derivforward = [derivforward_TE, derivforward_TM]
            self.fit_param[ipol][ipar] += dpar
            derivforward[ipol] = self.fitfunc_static(self.dataTX[ipol],
                self.fit_param[ipol][ 0], self.fit_param[ipol][ 1], self.fit_param[ipol][ 2],
                self.fit_param[ipol][ 3], self.fit_param[ipol][ 4], self.fit_param[ipol][ 5],
                self.fit_param[ipol][ 6], self.fit_param[ipol][ 7], self.fit_param[ipol][ 8],
                self.fit_param[ipol][ 9], self.fit_param[ipol][10], self.fit_param[ipol][11],
                self.fit_param[ipol][12], self.fit_param[ipol][13], self.fit_param[ipol][14],
                self.fit_param[ipol][15], self.fit_param[ipol][16], self.fit_param[ipol][17],
                self.fit_param[ipol][18], self.fit_param[ipol][19], self.fit_param[ipol][20])

            for rr in range(0, len(self.dataTX[ipol])):
                self.fit_deriv[ipol][ipar - iparA].append((derivforward[ipol][rr] - self.fit_result[ipol][rr]) / dpar)
            # end for

            # restore parameter value
            self.fit_param[ipol][ipar] = parambak

        # end for

    # end calc_deriv

    # ---------------------------------------------------------------------------------------
    # calculate the dynamic reflectivity
    # DR(theta) = (delta_R/delta_d)(theta) Dd  +  (delta_R/delta_n)(theta) Dn  +  (delta_R/delta_k)(theta) Dk
    def calc_dynamic(self, ipol, theta = None):

        if self.error:
            return
        # end if

        isArray = isinstance(theta, (list, tuple, np.ndarray))

        DR = None

        # 1. get three angles
        self.theta_pos[ipol] = []
        self.theta_val[ipol] = []
        derivd = self.fit_deriv[ipol][0]
        derivn = self.fit_deriv[ipol][1]
        derivk = self.fit_deriv[ipol][2]
        prevtheta = self.dataTX[ipol][0]
        sysline1 = []
        sysline2 = []
        sysline3 = []
        sysmatrix = [sysline1, sysline2, sysline3]
        ith = 0
        for rr in range(0, len(derivd)):
            if (not self.theta_manual) or (not isinstance(self.theta_manual, (list, tuple, np.ndarray))) or (len(self.theta_manual) != 3):
                if (((derivd[rr] > self.fit_tol) and (derivn[rr] > self.fit_tol) and (derivk[rr] > self.fit_tol))          \
                   or ((derivd[rr] < -self.fit_tol) and (derivn[rr] < -self.fit_tol) and (derivk[rr] < -self.fit_tol)))    \
                   and (self.dataTX[ipol][rr] > (prevtheta + self.thetaDelta)):
                    self.theta_pos[ipol].append(rr)
                    self.theta_val[ipol].append(self.dataTX[ipol][rr])
                    sysmatrix[ith].append(derivn[rr])
                    sysmatrix[ith].append(derivd[rr])
                    sysmatrix[ith].append(derivk[rr])
                    prevtheta = self.dataTX[ipol][rr]
                    ith += 1
                    if ith >= 3:
                        break
                    # end if
                # end if
            else:
                if (self.dataTXdyn[ipol][rr] >= self.theta_manual[ith]):
                    self.theta_pos[ipol].append(rr)
                    self.theta_val[ipol].append(self.dataTX[ipol][rr])
                    sysmatrix[ith].append(derivn[rr])
                    sysmatrix[ith].append(derivd[rr])
                    sysmatrix[ith].append(derivk[rr])
                    ith += 1
                    if ith >= 3:
                        break
                    # end if
                # end if
            # end if
        # end for

        if ith != 3:
            self.error = True
            self.message = "------> cannot determine coherent angle values for EO coefficients calculation"
            self.disp("\n" + self.message)
            return False
        # end if

        # 2. get the measured DR for the three angles
        ith = 0
        Rdyn = []
        for rr in range(0, len(self.dataTXdyn[ipol])):
            if (self.dataTXdyn[ipol][rr] >= self.theta_val[ipol][ith]):
                Rdyn.append(self.dataTYdyn[ipol][rr])
                ith += 1
                if ith >= 3:
                    break
                # end if
            # end if
        # end for

        if ith != 3:
            self.error = True
            self.message = "------> cannot calculate the EO coefficients: parameters not valid"
            self.disp("\n" + self.message)
            return False
        # end if

        if not self.fitting:
            self.disp("\n------> angle range : [%g %g]   tri-angle = [%g, %g, %g]" % (self.thetaStart*self.todegree, self.thetaEnd*self.todegree, self.theta_val[ipol][0]*self.todegree, self.theta_val[ipol][1]*self.todegree, self.theta_val[ipol][2]*self.todegree))
        # end if

        # 3. calculate Dd, Dn and Dk by solving the 3x3 linear system to get Dd, Dno and Dko
        # DR(theta1) = (delta_R/delta_d)(theta1) Dd  +  (delta_R/delta_n)(theta1) Dno  +  (delta_R/delta_k)(theta1) Dko
        # DR(theta2) = (delta_R/delta_d)(theta2) Dd  +  (delta_R/delta_n)(theta2) Dno  +  (delta_R/delta_k)(theta2) Dko
        # DR(theta2) = (delta_R/delta_d)(theta3) Dd  +  (delta_R/delta_n)(theta3) Dno  +  (delta_R/delta_k)(theta3) Dko
        # DR taken from experimental dynamic measurements
        matA = np.array([sysmatrix[0], sysmatrix[1], sysmatrix[2]])
        vecB = np.array(Rdyn)
        vecX = np.linalg.solve(matA, vecB)
        Dnx = vecX[0]
        Ddx = vecX[1]
        Dkx = vecX[2]
        if ipol == 0:
            # 's' polarization
            self.Dno = vecX[0]
            self.Ddo = vecX[1]
            self.Dko = vecX[2]
            # calculate the EO coefficient
            iparA = self.active_layer * 3
            self.r13 = -2.0 * 1e3 * self.fit_param[ipol][iparA] * self.Dno / (self.fit_param[ipol][iparA + 1] * self.fit_param[ipol][iparA + 1] * self.fit_param[ipol][iparA + 1] * self.voltage)
            self.d33 = self.Ddo * 1e3 / self.voltage
            self.DkDV = self.Dko / self.voltage
        else:
            # 'p' polarization
            self.Dne = vecX[0]
            self.Dde = vecX[1]
            self.Dke = vecX[2]
        # end if

        # 4. calculate the dynamic reflectivity DR
        self.fit_dyn[ipol] = []
        stddev = 0.0
        rmoy = 0.0
        for rr in range(0, len(self.dataTXdyn[ipol])):
            DRt = derivn[rr]*Dnx + derivd[rr]*Ddx + derivk[rr]*Dkx
            self.fit_dyn[ipol].append(DRt)
            stddev += (DRt - self.dataTYdyn[ipol][rr]) * (DRt - self.dataTYdyn[ipol][rr])
            rmoy += self.dataTYdyn[ipol][rr]
            if (theta is not None) and (not isArray) and (DR is None):
                if (theta >= self.dataTXdyn[ipol][rr]):
                    DR = DRt
                # end if
            # end if
        # end for
        self.fitdyn_stddev[ipol] = np.sqrt(stddev)
        self.fitdyn_mean[ipol] = rmoy / float(len(self.dataTYdyn[ipol]))
        if not self.fitting:
            self.disp("\n------> dynamic stddev = %g   mean = %g" % (self.fitdyn_stddev[ipol], self.fitdyn_mean[ipol]))
        # end if

        if theta is not None:
            if isArray:
                funcDR = interp1d(self.dataTXdyn[ipol], self.fit_dyn[ipol], kind='linear')
                DR = funcDR(theta)
            # end if
        else:
            DR = np.array(self.fit_dyn[ipol])
        # end if

        return DR

    # end calc_dynamic

    # ---------------------------------------------------------------------------------------
    def fitfunc_dynamic_jac(self, theta, dx, nx, kx):
        self.inJac = True
        R0 = np.array(self.fitfunc_dynamic(theta, dx, nx, kx))
        dnk = [dx, nx, kx]
        parcount = len(dnk)
        jac = np.array([[]])
        for ii in range(0, parcount):
            dnk0 = dnk[ii]
            dpar = (self.fit_derivtol * math.fabs(dnk0))
            if dpar < self.fit_derivtol_min:
                dpar = self.fit_derivtol_min
            # end if
            dnk[ii] = dnk0 + dpar
            RP = np.array(self.fitfunc_dynamic(theta, dnk[0], dnk[1], dnk[2]))
            if ii == 0:
                jac = (RP - R0) / dpar
            else:
                jac = np.vstack((jac, (RP - R0) / dpar))
            # end if
            dnk[ii] = dnk0
        # end for
        self.inJac = False
        return jac.transpose()
    # end fitfunc_dynamic_jac

    # ---------------------------------------------------------------------------------------
    def fitfunc_dynamic(self, theta, dx, nx, kx):

        if self.fitting:
            self.evalcount += 1
        # end if

        if os.path.isfile(self.stopfilename):
            try:
                os.unlink(self.stopfilename)
            except:
                pass
            # end try
            print("\n------> fitting stopped by the user after %d iterations" % (self.evalcount))
            exit(1)
        # end if

        ipol = 0 if (self.polarization == 's') else 1

        # 1. set parameters
        iparA = self.active_layer * 3
        self.fit_param[ipol][iparA]     = dx
        self.fit_param[ipol][iparA + 1] = nx
        self.fit_param[ipol][iparA + 2] = kx

        # 2. calculate R
        self.calc_reflectivity(ipol)
        if self.error:
            exit(1)
        # end if

        # 3. calculate the derivatives delta_R/delta_d, delta_R/delta_n, delta_R/delta_k
        self.calc_deriv(ipol)
        if self.error:
            exit(1)
        # end if

        # 4. calculate DR
        DR = self.calc_dynamic(ipol, theta)
        if self.error:
            exit(1)
        # end if

        return DR

    # end fitfunc_dynamic
    
    # ---------------------------------------------------------------------------------------
    # consistency check: the layers parameters found by fitting TE or TM...
    # ... should be consistent (taking into account the active layer index)
    def check_consistency(self):

        epsS = math.fabs(self.fit_stddev[0] / self.fit_mean[0])
        epsP = math.fabs(self.fit_stddev[1] / self.fit_mean[1])
        if (epsS < 0.3) or (epsP < 0.3):
            ifrom = 0 if (epsS < epsP) else 1
            ito =   1 if (epsS < epsP) else 0
            for nn in range(0, self.layers_count - 1):
                mm = nn * 3
                if nn == self.active_layer:
                    self.fit_param[ito][mm] = self.fit_param[ifrom][mm]
                else:
                    self.fit_param[ito][mm] = self.fit_param[ifrom][mm]
                    self.fit_param[ito][mm + 1] = self.fit_param[ifrom][mm + 1]
                    self.fit_param[ito][mm + 2] = self.fit_param[ifrom][mm + 2]
                # end if
            # end for
            epsS0 = epsS
            epsP0 = epsP
            self.calc_reflectivity(ipol = 0)
            self.calc_reflectivity(ipol = 1)
            epsS = math.fabs(self.fit_stddev[0] / self.fit_mean[0])
            epsP = math.fabs(self.fit_stddev[1] / self.fit_mean[1])
            print("\n------> coefficient of variation: TE: %g (%g)   TM: %g (%g)" % (epsS, epsS0, epsP, epsP0))
        # end if

    # end check_consistency

    def getclock(self):
        if sys.version_info[0] < 3:
            # Python 2.7.x
            return self.clock()
        else:
            # Python 3.x
            return time.perf_counter()
        # end if
    # end getclock

    # ---------------------------------------------------------------------------------------
    # fit the experimental static and dynamic reflectivity and calculate...
    # ... the electro-optical coefficient, if possible
    def fit(self, report = False):

        if not self.checked:
            self.check()
        # end if

        self.verbose = report

        # load the experimental data and set the angle range
        self.load()

        if self.error:
            return
        # end if

        self.fitting = True

        starttime = self.getclock()

        if report: print("\n------> fitting...")

        # fit the static reflectivity to get layers (d,n,k)
        self.fit_static(ipol = 0)
        self.fit_static(ipol = 1)

        # still fitting...
        self.fitting = True

        # calculate the static reflectivity R
        self.calc_reflectivity(ipol = 0)
        self.calc_reflectivity(ipol = 1)

        # consistency check: the layers parameters found by fitting TE or TM...
        # ... should be consistent (taking into account the active layer index)
        self.check_consistency()

        if self.fit_dynamic:
            # calculate the derivatives delta_R/delta_d, delta_R/delta_n, delta_R/delta_k
            self.calc_deriv(ipol = 0)
            self.calc_deriv(ipol = 1)

            # calculate the dynamic reflectivity DR
            self.calc_dynamic(ipol = 0)
            self.calc_dynamic(ipol = 1)

            # if necessary, fit the theoretical dynamic reflectivity DR with the experimental ones
            epsDynS = math.fabs(self.fitdyn_stddev[0] / self.fitdyn_mean[0])
            epsDynP = math.fabs(self.fitdyn_stddev[1] / self.fitdyn_mean[1])

            if (epsDynS > 0.1) or (epsDynP > 0.1):

                iparA = self.active_layer * 3
                for ipol in range(0, len(self.polarizationall)):
                    self.polarization = self.polarizationall[ipol]

                    init_tmp = []
                    bounds_tmp = ([], [])
                    for rr in range(iparA, iparA+3):
                        init_tmp.append(self.fit_param[ipol][rr])
                        bounds_tmp[0].append(self.bounds[0][rr])
                        bounds_tmp[1].append(self.bounds[1][rr])
                    # end for
                    self.evalcount = 0
                    starttime = self.getclock()
                    if report: self.disp("\n------> dynamic fitting ('%d-%s' polarization) in progress..." % (ipol, self.polarization))

                    try:
                        result_tmp,covar = curve_fit(self.fitfunc_dynamic, self.dataTXdyn[ipol], self.dataTYdyn[ipol], p0=init_tmp, bounds=bounds_tmp, check_finite=True, method='trf', xtol=self.fitdyn_tol, ftol=self.fitdyn_tol, gtol=self.fitdyn_tol, jac=self.fitfunc_dynamic_jac)
                        self.rchisqDyn[ipol] = self.reducedChiSquared(self.dataTYdyn[ipol], self.fitfunc_dynamic(self.dataTXdyn[ipol], *result_tmp), self.measurerrDyn[ipol] * self.dataTYdyn[ipol], len(result_tmp))
                        if report: self.disp("\n------> dynamic fitting ('%d-%s' polarization) ended (duration : %g seconds)\nnumber of evaluations: %d" % (ipol, self.polarization, self.getclock() - starttime, self.evalcount))

                    except ValueError as excT:
                        self.polarization = self.polarizationall[ipol]
                        self.error = True
                        self.message = "------> dynamic fitting ('%d-%s' polarization) ended (duration : %g seconds)%s" % (ipol, self.polarization, self.getclock() - starttime, str(excT))
                        if report: self.disp("\n" + self.message)
                        self.fit_param[ipol] = []
                        for rr in range(0, len(self.fit_init[ipol])):
                            self.fit_param[ipol].append(self.fit_init[ipol][rr])
                        # end for rr
                        break

                    except RuntimeError as excT:
                        self.polarization = self.polarizationall[ipol]
                        self.error = True
                        self.message = "------> dynamic fitting ('%d-%s' polarization) ended (duration : %g seconds)\n%s" % (ipol, self.polarization, self.getclock() - starttime, str(excT))
                        if report: self.disp("\n" + self.message)
                        self.fit_param[ipol] = []
                        for rr in range(0, len(self.fit_init[ipol])):
                            self.fit_param[ipol].append(self.fit_init[ipol][rr])
                        # end for rr
                        break

                    else:
                        if report: self.disp("\n------> dynamic fitting ('%d-%s' polarization) ended (duration : %g seconds)\nnumber of evaluations: %d" % (ipol, self.polarization, self.getclock() - starttime, self.evalcount))
                    # end try curve_fit

                # end for polar

                # consistency re-check: the layers parameters found by fitting TE or TM...
                # ... should be consistent (taking into account the active layer index)
                self.check_consistency()

                # calculate DR
                self.calc_dynamic(ipol = 0)
                self.calc_dynamic(ipol = 1)

            # end if relerr

        # end if

        # update the initial values
        for ipol in range(0, len(self.polarizationall)):
            self.fit_init[ipol] = []
            ll = 0
            for rr in range(0, self.parameters_count):
                 self.fit_init[ipol].append(self.fit_param[ipol][rr])
            # end for
        # end for polar

        self.fitting = False
                
        if report: self.report()

    # end fit

    # ---------------------------------------------------------------------------------------
    def report(self):

        if self.error:
            return
        # end fit

        for ipol in range(0, len(self.polarizationall)):
            print("\n------> obtained parameters for '%s' polarization (d in nm):" % self.polarizationall[ipol])
            ll = 0
            for rr in range(0, self.parameters_count, 3):
                print("\n        %7s (Layer %d): d, n, k : %7g %7g %7g  (%7g %7g %7g)" % (self.layers[ll], ll + 1, self.fit_param[ipol][rr], self.fit_param[ipol][rr+1], self.fit_param[ipol][rr+2], self.d[ll], self.ne[ll] if (ipol == 0) else self.no[ll], self.ke[ll] if (ipol == 0) else self.ko[ll]))
                ll += 1
            # end for
        # end for polar

        if self.fit_dynamic and self.r13 and self.d33:
            print("\n------> r13 = %g pm/V   d33 = %g pm/V    Dk/DV = %g 1/V" % (self.r13, self.d33, self.DkDV))
        # end if

        if self.fit_dynamic:
            print("\n------> reduced chi squared (static) = (%g, %g)\n        reduced chi squared (dynamic) = (%g, %g)" % (self.rchisq[0], self.rchisq[1], self.rchisqDyn[0], self.rchisqDyn[1]))
        else:
            print("\n------> reduced chi squared (static) = (%g, %g)" % (self.rchisq[0], self.rchisq[1]))
        # end if

    # end report

    # ---------------------------------------------------------------------------------------
    def plot(self, derviplot = 'TE'):

        if self.error:
            return
        # end fit

        figT = pl.figure(num=1, figsize=(10,6), facecolor='#FFFFFF', linewidth=1.0, frameon=True)
        figT.canvas.set_window_title(self.structure + ': Static and Dynamic Reflectivity') 
        tManager = pl.get_current_fig_manager()
        img = Tk.PhotoImage(file = os.path.dirname(os.path.abspath(__file__)) + '/iconmain.gif')
        tManager.window.tk.call('wm', 'iconphoto', tManager.window._w, img)

        gridT = pl.GridSpec(6, 2) if self.fit_dynamic else pl.GridSpec(6, 1)

        pl.subplot(gridT[:-3,0])
        labelPol = [r'TE - experimental', r'TM - experimental']
        labelPolTheory = [r'TE - theoretical', r'TM - theoretical']
        curveStyle = ['o', 's']
        curveColor = ['#ab5b61', '#465f9e']
        curveColorFit = ['red', 'blue']
        for ipol in range(0, len(self.polarizationall)):
            pl.plot(self.dataTX[ipol]*self.todegree, self.dataTY[ipol], curveStyle[ipol], markersize=6.0, markevery=1, markerfacecolor='none', color=curveColor[ipol], label=labelPol[ipol], zorder=3)
            pl.plot(self.dataTX[ipol]*self.todegree, self.fit_result[ipol], linewidth=2.0, color=curveColorFit[ipol], label=labelPolTheory[ipol], zorder=3)
        # end for
        pl.ylabel(r'${Reflectivity\ R}$', fontsize=self.fontsize)
        pl.tick_params(axis='x', which='major', labelsize=self.fontsize)
        pl.tick_params(axis='y', which='major', labelsize=self.fontsize)
        pl.xlim(self.thetaStart*self.todegree, self.thetaEnd*self.todegree)
        pl.legend(loc='best')
        tgca = pl.gca()
        tgca.axes.get_xaxis().grid(True)
        tgca.axes.get_yaxis().grid(True)
        tgca.set_xticklabels([])
        pl.legend(loc='best', fontsize=self.fontsize-3)

        pl.subplot(gridT[3:,0])
        for ipol in range(0,2):
            dataTYdyn_ar = 1e5 * np.array(self.dataTYdyn[ipol])
            pl.plot(self.dataTXdyn[ipol]*self.todegree, dataTYdyn_ar, curveStyle[ipol], markersize=6.0, markevery=1, markerfacecolor='none', color=curveColor[ipol], label=labelPol[ipol], zorder=3)
            if self.fit_dynamic:
                fitresult_dyn_ar = 1e5 * np.array(self.fit_dyn[ipol])
                pl.plot(self.dataTXdyn[ipol]*self.todegree, fitresult_dyn_ar, linewidth=2.0, color=curveColorFit[ipol], label=labelPolTheory[ipol], zorder=3)
            # end if
        # end for
        pl.ylabel(r'${10^{5}\ \times\ \ \Delta R}$', fontsize=self.fontsize)
        pl.tick_params(axis='y', which='major', labelsize=self.fontsize)
        pl.xlim(self.thetaStart*self.todegree, self.thetaEnd*self.todegree)
        pl.legend(loc='best', fontsize=self.fontsize-3)
        tgca = pl.gca()
        tgca.axes.get_xaxis().grid(True)
        tgca.axes.get_yaxis().grid(True)

        pl.xlabel(r'${Angle\ of\ incidence\ \Theta\ (degree)}$', fontsize=self.fontsize)
        pl.tick_params(axis='x', which='major', labelsize=self.fontsize)

        if self.fit_dynamic:

            ipol = 0 if (derviplot == 'TM') else 0    # 0 for # TE polarization and 1 for TM polarization

            curveColor = ['#A8703B', '#754E99', '#177349']

            if 0 == ipol:
                labelD = [r'${\delta R / \delta d \ (10^{-4}\ nm^{-1})}$', r'${\delta R / \delta n_{o}}$', r'${\delta R / \delta k_{o}}$']
            else:
                labelD = [r'${\delta R / \delta d \ (10^{-4}\ nm^{-1})}$', r'${\delta R / \delta n_{e}}$', r'${\delta R / \delta k_{e}}$']
            # end if
            fit_deriv_d = 1e4 * np.array(self.fit_deriv[ipol][0])
            fit_deriv_n =       np.array(self.fit_deriv[ipol][1])
            fit_deriv_k =       np.array(self.fit_deriv[ipol][2])
            fit_deriv_ar = [fit_deriv_d, fit_deriv_n, fit_deriv_k]
            grid_ar = [gridT[0:-4,1], gridT[2:-2,1], gridT[4:,1]]
            for icurve in range(0, len(labelD)):
                pl.subplot(grid_ar[icurve])
                pl.plot(self.dataTX[ipol]*self.todegree, fit_deriv_ar[icurve], linewidth=2.5, color=curveColor[icurve])
                pl.ylabel(labelD[icurve], fontsize=self.fontsize)
                pl.tick_params(axis='x', which='major', labelsize=self.fontsize)
                pl.tick_params(axis='y', which='major', labelsize=self.fontsize)
                tgca = pl.gca()
                if icurve < (len(labelD) - 1):
                    tgca.set_xticklabels([])
                    tgca.axes.get_xaxis().grid(True)
                    tgca.axes.get_yaxis().grid(True)
                # end if

                pl.xlim(self.thetaStart*self.todegree, self.thetaEnd*self.todegree)
            # end for
            tgca.axes.get_xaxis().set_visible(True)
            tgca.axes.get_xaxis().grid(True)
            tgca.axes.get_yaxis().grid(True)
            pl.xlabel(r'${Angle\ of\ incidence\ \Theta\ (degree)}$', fontsize=self.fontsize)
            pl.tick_params(axis='y', which='major', labelsize=self.fontsize)
        # end if

        if self.fit_dynamic:
            if self.r13 and self.d33:
                chi2min = 0.2
                chi2max = 1.8
                if (self.rchisq[0] >= chi2min) and (self.rchisq[0] <= chi2max) and (self.rchisq[1] >= chi2min) and (self.rchisq[1] <= chi2max)and (self.rchisqDyn[0] >= chi2min) and (self.rchisqDyn[0] <= chi2max) and (self.rchisqDyn[1] >= chi2min) and (self.rchisqDyn[1] <= chi2max):
                    strT = r"$r_{13}\ =\ %.3f\ pm/V\ \ \ d_{33}\ =\ %.3f\ pm/V\ \ \ reduced\ \chi^2:\ static:\ (%.3f,\ %.3f)\ \ \ dynamic:\ (%.3f,\ %.3f)$" % (self.r13, self.d33, self.rchisq[0], self.rchisq[1], self.rchisqDyn[0], self.rchisqDyn[1])
                    tcolor = "#1661AB"
                else:
                    strT = r"$r_{13}\ =\ %.3f\ pm/V\ \ \ d_{33}\ =\ %.3f\ pm/V\ \ \ reduced\ \chi^2:\ ! inconsistent\  values:\ (%.3f,\ %.3f)\ \ \ (%.3f,\ %.3f)$" % (self.r13, self.d33, self.rchisq[0], self.rchisq[1], self.rchisqDyn[0], self.rchisqDyn[1])
                    tcolor = "#CC6318"
                # end if
                pl.figtext(0.5, 0.97, strT, fontsize=self.fontsize, color=tcolor, ha="center")
            # end if
            else:
                pl.figtext(0.5, 0.97, r"$r_{13},\ d_{33}\ not\ available$", fontsize=self.fontsize, color="#CC1818", ha="center")
            # end if
        # end if

        pl.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.94, wspace=0.4, hspace=0.4)
        figT.tight_layout(rect=(0.02,0.02,0.98,0.96), h_pad=1, w_pad=3)
        pl.show()

    # end plot

# end class PyEOC
