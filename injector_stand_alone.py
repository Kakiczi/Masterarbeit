# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# -*-coding:utf8-*-

from __future__ import print_function

"""
This file is part of SkyLab

Skylab is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

ps_injector
===========

Point Source Injection classes. The interface with the core
PointSourceLikelihood - Class requires the methods

    fill - Filling the class with Monte Carlo events

    sample - get a weighted sample with mean number `mu`

    flux2mu - convert from a flux to mean number of expected events

    mu2flux - convert from a mean number of expected events to a flux

"""

# python packages
import logging
from functions4 import *

# scipy-project imports
import numpy as np
from numpy.lib.recfunctions import drop_fields, append_fields
import scipy.interpolate
import healpy as hp

# get module logger
def trace(self, message, *args, **kwargs):
    r""" Add trace to logger with output level beyond debug

    """
    if self.isEnabledFor(5):
        self._log(5, message, args, **kwargs)

logging.addLevelName(5, "TRACE")
logging.Logger.trace = trace

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

_deg = 4
_ext = 3

def set_pars(self, **kwargs):
    r"""Constructor with basic settings needed in all LLH classes.

    """

    # Set all attributes passed, warn if private or not known
    for attr, value in kwargs.iteritems():
        if not hasattr(self, attr):
            logger.error((">>>> {0:s} does not have attribute '{1:s}', "+
                          "skipping...").format(self.__repr__(), attr))
            continue
        if attr.startswith("_"):
            logger.error((">>>> _{0:s} should be considered private and "+
                          "for internal use only!").format(attr))
        setattr(self, attr, value)

    return

#~ def rotate(ra1, dec1, ra2, dec2, ra3, dec3, beta):
	#~ r""" Rotate ra1 and dec1 in a way that ra2 and dec2 will exactly map
	#~ onto ra3 and dec3, respectively. All angles are treated as radians.
	#~ """
	#~ print("Reco dir original", ra1, dec1)
	#~ print("True dir original", ra2, dec2)
	#~ print("True dir new", ra3, dec3)
#~ 
	#~ # turn rightascension and declination into zenith and azimuth for healpy
	#~ phi1 = ra1 - np.pi
	#~ zen1 = np.pi/2. - dec1
	#~ phi2 = ra2 - np.pi
	#~ zen2 = np.pi/2. - dec2
	#~ phi3 = ra3 - np.pi
	#~ zen3 = np.pi/2. - dec3
	#~ print("Converted to phi and zen:")
	#~ print("Reco dir original", phi1, zen1)
	#~ print("True dir original", phi2, zen2)
	#~ print("True dir new", phi3, zen3)
#~ 
	#~ # no rotation, just identity
	#~ x = np.array([hp.rotator.rotateDirection(
									#~ hp.rotator.get_rotation_matrix((0, 0, 0.))[0],
									#~ z, p) for z, p, dz, dp in zip(zen1, phi1, zen2, phi2)])
									#~ 
	#~ x = np.array([hp.rotator.rotateDirection(
									#~ hp.rotator.get_rotation_matrix((dp, -dz+zen3, -phi3), eulertype="Y")[0],
									#~ z, p) for z, p, dz, dp in zip(zen1, phi1, zen2, phi2)])
#~ 
	#~ zen = np.array([i[0] for i in x])
	#~ phi = np.array([i[1] for i in x])
	#~ print ("Reco dir new phi and zen ", phi, zen)
#~ 
	#~ dec = np.pi/2. - zen
	#~ ra = phi + np.pi
	#~ print("Reco dir new RA and dec", ra, dec)
	#~ return np.atleast_1d(ra), np.atleast_1d(dec)
	
def rotate(ra1, dec1, ra2, dec2, ra3, dec3, beta, rotate_this=True):
    r""" Rotate ra1 and dec1 in a way that ra2 and dec2 will exactly map
    onto ra3 and dec3, respectively. All angles are treated as radians.

    """
    
    # turn rightascension and declination into zenith and azimuth for healpy
    phi1 = ra1 - np.pi
    zen1 = np.pi/2. - dec1
    phi2 = ra2 - np.pi
    zen2 = np.pi/2. - dec2
    phi3 = ra3 - np.pi
    zen3 = np.pi/2. - dec3

    #~ print("source zenith ", zen3)
    #~ print("source phi ", phi3)
    # no rotation, just identity
    x = np.array([hp.rotator.rotateDirection(
                    hp.rotator.get_rotation_matrix((0, 0, 0.))[0],
                    z, p) for z, p, dz, dp in zip(zen1, phi1, zen2, phi2)])
    
    #~ print("Start\n", x)
    if rotate_this:
			#"""
			# rotate RA back
			x = np.array([hp.rotator.rotateDirection(
											hp.rotator.get_rotation_matrix((dp, 0, 0.))[0],
											z, p) for z, p, dz, dp in zip(zen1, phi1, zen2, phi2)])

			#print "Rotate to zero RA\n", x

			
			# rotate to pole (back in zen)
			x = np.array([hp.rotator.rotateDirection(
									 hp.rotator.get_rotation_matrix((0, -dz, 0))[0],
									 z , p) for z, p, dz, dp in zip(x[:,0], x[:,1], zen2, phi2)])

			#~ print("This should be pole\n",x)
			
			# rotate with beta in case of oversampling
			x = np.array([hp.rotator.rotateDirection(
									 hp.rotator.get_rotation_matrix((be, 0, 0))[0],
									 z , p) for z, p, be in zip(x[:,0], x[:,1], beta)])
			#~ print("This should be pole, rotated with beta\n",x)

			# Rotate to final dec
			x = np.array([hp.rotator.rotateDirection(
									hp.rotator.get_rotation_matrix((0, zen3, 0))[0],
									x[:,0], x[:,1])])
			
			#print "This should be at final dec\n", x
			
			# Rotate to final RA
			x = np.array([hp.rotator.rotateDirection(
									hp.rotator.get_rotation_matrix((-phi3, 0, 0))[0],
									x[:,0], x[:,1])])
			
			#~ print("This should be at final coordinates\n", x)
    #"""    
    if not rotate_this:
			x = np.array([hp.rotator.rotateDirection(
											hp.rotator.get_rotation_matrix((dp, -dz+zen3, -phi3), eulertype="Y")[0],
											z, p, do_rot=rotate_this) for z, p, dz, dp in zip(zen1, phi1, zen2, phi2)])
    
    #~ print("End\n", x)
    #~ print(np.shape(x))
    
    zen = np.array([i[0] for i in x])
    phi = np.array([i[1] for i in x])
    
    #~ print(zen)
    #~ print(phi)
    dec = np.pi/2. - zen
    ra = phi + np.pi    

    return np.atleast_1d(ra), np.atleast_1d(dec)

def rotate_struct(ev, ra, dec, beta, rotate_this=True):
    r"""Wrapper around the rotate-method in skylab.coords for structured
    arrays.

    Parameters
    ----------
    ev : structured array
        Event information with ra, sinDec, plus true information

    ra, dec : float
        Coordinates to rotate the true direction onto

    beta : ndarray
        Rotation of reconstructed direction around the true direction

    Returns
    --------
    ev : structured array
        Array with rotated value, true information is deleted

    """
    names = ev.dtype.names

    # Function call
    ev["ra"], rot_dec = rotate(ev["ra"], np.arcsin(ev["sinDec"]),
                               ev["trueRa"], ev["trueDec"],
                               ra, dec, beta, rotate_this=rotate_this)
                               
                               
###############Gute Idee. Funktioniert aber leider so nicht######################
    #print(np.degrees(rot_dec))	
    #if type(rot_dec[0])!=np.float64:
		#for k,i  in enumerate(rot_dec[0]):
			#if i<np.radians(-5):
				#while rot_dec[0][k]<np.radians(-5):
					#t=rotate_to_valid_angle(ra, dec, ev["ra"][k] ,i)
					#rot_dec[0][k]=t[1]
					#ev["ra"][k]=t[0]				
    #else:
			#if rot_dec[0]<np.radians(-5):
				#print("Ob diese Zeile wohl jemals jemand als Ausgabe lesen wird?")
				#while rot_dec[0]<np.radians(-5):
					#t=rotate_to_valid_angle(ra, dec, ev["ra"][0] , rot_dec[0])
					#rot_dec[0]=t[1]
					#ev["ra"][0]=t[0]	
    #print(rot_dec)           
##############################################################################################
           
    if "dec" in names:
        ev["dec"] = rot_dec
    ev["sinDec"] = np.sin(rot_dec)

    ev = np.lib.recfunctions.append_fields(ev, "trueRotRa", np.zeros(len(ev)), dtypes="<f4", usemask=False)  
    ev = np.lib.recfunctions.append_fields(ev, "trueRotDec", np.zeros(len(ev)), dtypes="<f4", usemask=False)  
                               
    ev["trueRotRa"], ev["trueRotDec"] = rotate(ev["trueRa"], ev["trueDec"],
                               ev["trueRa"], ev["trueDec"],
                               ra, dec, beta, rotate_this=rotate_this)   
 
    # "delete" Monte Carlo information from sampled events
 #   non_mc = [name for name in names
       #            if name not in ["trueRa", "trueDec", "trueE", "ow"]]

    # ev = ev[non_mc].copy()
#    ev = ev.copy()    
    ev=ev[np.where(ev["dec"]>np.radians(-5))]
    return ev

 
class Injector(object):
    r"""Base class for Signal Injectors defining the essential classes needed
    for the LLH evaluation.

    """

    def __init__(self, *args, **kwargs):
        r"""Constructor: Define general point source features here...

        """
        self.__raise__()

    def __raise__(self):
        raise NotImplementedError("Implemented as abstract in {0:s}...".format(
                                    self.__repr__()))

    def fill(self, *args, **kwargs):
        r"""Filling the injector with the sample to draw from, work only on
        data samples known by the LLH class here.

        """
        self.__raise__()

    def flux2mu(self, *args, **kwargs):
        r"""Internal conversion from fluxes to event numbers.

        """
        self.__raise__()

    def mu2flux(self, *args, **kwargs):
        r"""Internal conversion from mean number of expected neutrinos to
        point source flux.

        """
        self.__raise__()

    def sample(self, *args, **kwargs):
        r"""Generator method that returns sampled events. Best would be an
        infinite loop.

        """
        self.__raise__()

class PointSourceInjector(Injector):
    r"""Class to inject a point source into an event sample.

    """
    _src_dec = np.nan
    _sinDec_bandwidth = 0.1
    _sinDec_range = [-1., 1.]

    _E0 = 1.
    _GeV = 1.e3
    _e_range = [0., np.inf]

    _names = tuple()

    _random = np.random.RandomState()
    _seed = None

    def __init__(self, gamma, **kwargs):
        r"""Constructor. Initialize the Injector class with basic
        characteristics regarding a point source.

        Parameters
        -----------
        gamma : float
            Spectral index, positive values for falling spectra

        kwargs : dict
            Set parameters of class different to default

        """

        # source properties
        self.gamma = gamma

        # Set all other attributes passed to the class
        set_pars(self, **kwargs)

        return

    def __str__(self):
        r"""String representation showing some more or less useful information
        regarding the Injector class.

        """
        sout = ("\n{0:s}\n"+
                67*"-"+"\n"+
                "\tSpectral index     : {1:6.2f}\n"+
                "\tSource declination : {2:5.1f} deg\n"
                "\tlog10 Energy range : {3:5.1f} to {4:5.1f}\n").format(
                         self.__repr__(),
                         self.gamma, np.degrees(self.src_dec),
                         *self.e_range)
        sout += 67*"-"

        return sout

    @property
    def sinDec_range(self):
        return self._sinDec_range

    @sinDec_range.setter
    def sinDec_range(self, val):
        if len(val) != 2:
            raise ValueError("SinDec range needs only upper and lower bound!")
        if val[0] < -1 or val[1] > 1:
            logger.warn("SinDec bounds out of [-1, 1], clip to that values")
            val[0] = max(val[0], -1)
            val[1] = min(val[1], 1)
        if np.diff(val) <= 0:
            raise ValueError("SinDec range has to be increasing")
        self._sinDec_range = [float(val[0]), float(val[1])]
        return

    @property
    def e_range(self):
        return self._e_range

    @e_range.setter
    def e_range(self, val):
        if len(val) != 2:
            raise ValueError("Energy range needs upper and lower bound!")
        if val[0] < 0. or val[1] < 0:
            logger.warn("Energy range has to be non-negative")
            val[0] = max(val[0], 0)
            val[1] = max(val[1], 0)
        if np.diff(val) <= 0:
            raise ValueError("Energy range has to be increasing")
        self._e_range = [float(val[0]), float(val[1])]
        return

    @property
    def GeV(self):
        return self._GeV

    @GeV.setter
    def GeV(self, value):
        self._GeV = float(value)

        return

    @property
    def E0(self):
        return self._E0

    @E0.setter
    def E0(self, value):
        self._E0 = float(value)

        return

    @property
    def random(self):
        return self._random

    @random.setter
    def random(self, value):
        self._random = value

        return

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, val):
        logger.info("Setting global seed to {0:d}".format(int(val)))
        self._seed = int(val)
        self.random = np.random.RandomState(self.seed)

        return

    @property
    def sinDec_bandwidth(self):
        return self._sinDec_bandwidth

    @sinDec_bandwidth.setter
    def sinDec_bandwidth(self, val):
        if val < 0. or val > 1:
            logger.warn("Sin Declination bandwidth {0:2e} not valid".format(
                            val))
            val = min(1., np.fabs(val))
        self._sinDec_bandwidth = float(val)

        self._setup()

        return

    @property
    def src_dec(self):
        return self._src_dec

    @src_dec.setter
    def src_dec(self, val):
        if not np.fabs(val) < np.pi / 2.:
            logger.warn("Source declination {0:2e} not in pi range".format(
                            val))
            return
        if not (np.sin(val) > self.sinDec_range[0]
                and np.sin(val) < self.sinDec_range[1]):
            logger.error("Injection declination not in sinDec_range!")
        self._src_dec = float(val)

        self._setup()

        return

    def _setup(self):
        r"""If one of *src_dec* or *dec_bandwidth* is changed or set, solid
        angles and declination bands have to be re-set.

        """

        A, B = self._sinDec_range

        m = (A - B + 2. * self.sinDec_bandwidth) / (A - B)
        b = self.sinDec_bandwidth * (A + B) / (B - A)

        sinDec = m * np.sin(self.src_dec) + b

        min_sinDec = max(A, sinDec - self.sinDec_bandwidth)
        max_sinDec = min(B, sinDec + self.sinDec_bandwidth)

        self._min_dec = np.arcsin(min_sinDec)
        self._max_dec = np.arcsin(max_sinDec)

        # solid angle of selected events
        self._omega = 2. * np.pi * (max_sinDec - min_sinDec)

        return

    def _weights(self):
        r"""Setup weights for given models.

        """
        # weights given in days, weighted to the point source flux
        self.mc["ow"] *= self.mc["trueE"]**(-self.gamma) / self._omega

        self._raw_flux = np.sum(self.mc["ow"], dtype=np.float)

        # normalized weights for probability
        self._norm_w = self.mc["ow"] / self._raw_flux

        # double-check if no weight is dominating the sample
        if self._norm_w.max() > 0.1:
            logger.warn("Warning: Maximal weight exceeds 10%: {0:7.2%}".format(
                            self._norm_w.max()))

        return

    def fill(self, src_dec, mc):
        r"""Fill the Injector with MonteCarlo events.

        Parameters
        -----------
        src_dec : float
            Source location

        mc : structured array
            Monte Carlo events

        """
        print(mc.dtype.names)
        self.src_dec = src_dec

        band_mask = ((np.sin(mc["trueDec"]) > np.sin(self._min_dec))
                     &(np.sin(mc["trueDec"]) < np.sin(self._max_dec)))
        band_mask &= ((mc["trueE"] / self.GeV > self.e_range[0])
                      &(mc["trueE"] / self.GeV < self.e_range[1]))

        logger.info(("{0:7.2%} ({1:5d}) of the MC events are in declination "+
                     "band ({2:7.2f} deg {3:7.2f} deg)").format(
                        np.sum(band_mask, dtype=np.float)/len(band_mask),
                        np.sum(band_mask, dtype=np.int),
                        np.degrees(self._min_dec), np.degrees(self._max_dec)))

        if not np.any(band_mask):
            raise ValueError("No events were selected for injection!")

        # get MC event's in the selected events

        self.mc = mc[band_mask]

        self._weights()

        print("Selected {0:6d} events for injection at {1:7.2f}deg".format(
                    len(self.mc), np.degrees(self.src_dec)))
        self.mc = rotate_struct(self.mc, np.pi, self.src_dec)
        print("\tRotated events towards (180.0, {0:+4.1f})deg".format(
                    np.degrees(self.src_dec)))

        return

    def flux2mu(self, flux):
        r"""Convert a flux to mean number of expected events.

        Converts a flux :math:`\Phi_0` to the mean number of expected
        events using the spectral index :math:`\gamma`, the
        specified energy unit `x GeV` and the point of normalization `E0`.

        The flux is calculated as follows:

        .. math::

            \frac{d\Phi}{dE}=\Phi_0\,E_0^{2-\gamma}
                                \left(\frac{E}{E_0}\right)^{-\gamma}

        In this way, the flux will be equivalent to a power law with
        index of -2 at the normalization energy `E0`.

        """

        gev_flux = (flux
                        * (self.E0 * self.GeV)**(self.gamma - 1.)
                        * (self.E0)**(self.gamma - 2.))

        return self._raw_flux * gev_flux

    def mu2flux(self, mu):
        r"""Calculate the corresponding flux in [GeV^(gamma - 1) s^-1 cm^-2]
        for a given number of mean source events.

        """

        gev_flux = mu / self._raw_flux

        return (gev_flux
                    * self.E0**(2. - self.gamma)
                    * (self.E0 * self.GeV)**(1. - self.gamma))

    def sample(self, mean_mu, poisson=True):
        r""" Generator to get sampled events for a Point Source location.

        Parameters
        -----------
        mean_mu : float
            Mean number of events to sample

        Returns
        --------
        sam_ev : iterator
            sampled_events for each loop iteration, call with *next()*

        Optional Parameters
        --------------------
        poisson : bool
            Use poisson fluctuations, otherwise sample exactly *mean_mu*

        """

        # generate event numbers using poissonian events
        while True:
            num = (self.random.poisson(mean_mu)
                        if poisson else int(np.around(mean_mu)))

            logger.debug(("Generated number of sources: {0:3d} "+
                          "of mean {1:5.1f} sources").format(num, float(mean_mu)))

            # create numpy array with *num* entries
            sam_ev = np.empty((num, ), dtype=self.mc.dtype)

            # if no events should be sampled, return empty lists
            if num < 1:
                yield sam_ev[[name for name in sam_ev.dtype.names
                              if name not in ["trueRa", "trueDec",
                                              "trueE", "ow"]]]

                continue

            ind = self.random.choice(len(self.mc), size=num, p=self._norm_w)

            # get the events that were sampled
            sam_ev = self.mc[ind]
            
            # return the sampled events in each iteration
            yield sam_ev
            
class PointSourceInjector2(PointSourceInjector):
    _src_ra = np.nan
    mc = None
    
    @property
    def src_ra(self):
        return self._src_ra

    @src_ra.setter
    def src_ra(self, val):
        if not np.fabs(val) < 2*np.pi:
            logger.warn("Source declination {0:2e} not in 2*pi range".format(val))
            return
        self._src_ra = float(val)

        self._setup()

        return

    
    def _setup(self):
        
        PointSourceInjector._setup(self)
        if not self.mc is None:
            self.band_mask = ((np.sin(self.mc["trueDec"]) > np.sin(self._min_dec))
                         &(np.sin(self.mc["trueDec"]) < np.sin(self._max_dec)))
            self.band_mask &= ((self.mc["trueE"] / self.GeV > self.e_range[0])
                          &(self.mc["trueE"] / self.GeV < self.e_range[1]))

            logger.info(("{0:7.2%} ({1:5d}) of the MC events are in declination "+
                         "band ({2:7.2f} deg {3:7.2f} deg)").format(
                            np.sum(self.band_mask, dtype=np.float)/len(self.band_mask),
                            np.sum(self.band_mask, dtype=np.int),
                            np.degrees(self._min_dec), np.degrees(self._max_dec)))

            if not np.any(self.band_mask):
                raise ValueError("No events were selected for injection!")
                
        return
    
    def fill(self, src_dec, mc):
        r"""Fill the Injector with MonteCarlo events.

        Parameters
        -----------
        src_dec : float
            Source location

        mc : structured array
            Monte Carlo events

        """
        self.src_dec = src_dec
        self.mc = mc
		#self._weights()
        self._setup()

    def sample(self, mean_mu, poisson=True, replace=True, rotate_this=True):
        r""" Generator to get sampled events for a Point Source location.

        Parameters
        -----------
        mean_mu : float
            Mean number of events to sample

        Returns
        --------
        sam_ev : iterator
            sampled_events for each loop iteration, call with *next()*

        Optional Parameters
        --------------------
        poisson : bool
            Use poisson fluctuations, otherwise sample exactly *mean_mu*

        """

        # generate event numbers using poissonian events
        while True:
            num = (self.random.poisson(mean_mu)
                        if poisson else int(np.around(mean_mu)))

            logger.debug(("Generated number of sources: {0:3d} "+
                          "of mean {1:5.1f} sources").format(num, float(mean_mu)))
            # create numpy array with *num* entries
            test=[]
            for key in self.mc.dtype.names:
               if str(self.mc.dtype[key])=="float32":
                   test.append('<f4')
               if str(self.mc.dtype[key])=="float64":
                   test.append('<f8')    
            sam_ev = np.empty((num, ), dtype=zip(np.concatenate([self.mc.dtype.names, ['trueRotRa', 'trueRotDec']]), test+2*['<f4']))
            # if no events should be sampled, return empty lists
            if num < 1:
                yield sam_ev[[name for name in sam_ev.dtype.names ]]

                continue
						#self._norm_w[self.band_mask]/np.sum(self._norm_w[self.band_mask]) self.mc["astro"][self.band_mask]/np.sum(self.mc["astro"][self.band_mask])
            ### astro weights are used, re-weighting has to be done with the MC weights at the moment
            ### changing Gamma has no effect
            ind = self.random.choice(len(self.mc[self.band_mask]), size=num, p=self.mc["astro"][self.band_mask]/np.sum(self.mc["astro"][self.band_mask]), replace=replace)
            # events that might be sampled multiple times will be rotated
            # around the true direction vector
            beta = [self.random.uniform(0., 2.*np.pi) if i in ind[:k] else 0.
                    for k, i in enumerate(ind)]
            #~ print("non-zero beta:", np.where(beta !=0.))
            # get the events that were sampled
            sam_ev = self.mc[self.band_mask][ind]
            lsam_ev=len(sam_ev)
            print("Generated source at Dec", np.degrees(self.src_dec), "deg, RA", np.degrees(self.src_ra), "deg with", num, "events")
            # rotate the true direction onto the point source
            sam_ev = rotate_struct(sam_ev, self.src_ra, self.src_dec, beta, rotate_this=rotate_this)
            while len(sam_ev)!=lsam_ev:
				ind2 = self.random.choice(len(self.mc[self.band_mask]), size=lsam_ev-len(sam_ev), p=self.mc["astro"][self.band_mask]/np.sum(self.mc["astro"][self.band_mask]), replace=replace)
				beta = [self.random.uniform(0., 2.*np.pi) if i in ind2[:k] or i in ind else 0.
                    for k, i in enumerate(ind2)]
				sam_temp=self.mc[self.band_mask][ind2]
				ind=np.concatenate((ind,ind2))
				sam_temp=rotate_struct(sam_temp, self.src_ra, self.src_dec, beta, rotate_this=rotate_this)
				sam_ev=np.append(sam_ev,sam_temp)
            #rotate_struct(sam_ev, sam_ev["trueRa"], sam_ev["trueDec"], beta)
            # return the sampled events in each iteration
            yield sam_ev
        
        
        #self.mc = rotate_struct(self.mc, np.pi, self.src_dec)
        #print("\tRotated events towards (180.0, {0:+4.1f})deg".format(
        #            np.degrees(self.src_dec)))

