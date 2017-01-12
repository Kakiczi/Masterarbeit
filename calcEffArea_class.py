import numpy as np
import scipy
from scipy.signal import convolve2d

class calcEffArea():
	def __init__(self):
		self._enum = dict()
		self._sams = dict()
		self._gamma_binmids=np.linspace(2,2.4,20)
		self.sindec_bins=np.linspace(0.0, 1 , 200)
		self._nuhist=dict()
		self._gamma_def=2.14
		self._nuspline = dict()
		self._sindec_binmids=[]
		for i in range(len(self.sindec_bins)-1):
			self._sindec_binmids.append((self.sindec_bins[i]+self.sindec_bins[i+1])/2)
       
	def add_sample(self, name, mc):
		enum = max(self._enum) + 1 if self._enum else 0
		self._enum[enum] = name
		self._sams[enum] = mc

        # add mc info for injection
		if len(self._enum) == 1:
			self.mc = np.lib.recfunctions.append_fields( mc, "enum", enum * np.ones(len(mc)), dtypes=np.int, usemask=False)
		else:
			self.mc = np.append(self.mc, np.lib.recfunctions.append_fields(mc, "enum", enum * np.ones(len(mc)), dtypes=np.int, usemask=False))

        # create histogram of signal expectation for this sample
		x = np.sin(mc["trueDec"])
		hist = np.vstack([np.histogram(x, weights=mc["ow"]* mc["trueE"]**(-gm), bins=self.sindec_bins)[0] for gm in self._gamma_binmids])
		hist = hist.T
		
		# take the mean of the histogram neighbouring bins
		nwin = 5
		filter_window = np.ones((nwin, nwin), dtype=np.float)
		filter_window /= np.sum(filter_window)
		self._nuhist[enum] = (convolve2d(hist, filter_window, mode="same")/ convolve2d(np.ones_like(hist), filter_window, mode="same"))
		return 
		
	def powerlaw_weights(self, src_dec, **fit_pars):
		gamma = fit_pars.pop("gamma", self._gamma_def)
		
		# check if samples and splines are both equal, otherwise re-do spline
		if set(self._nuhist) != set(self._nuspline):
			# delete all old splines
			for key in self._nuspline.iterkeys():
				del self._nuspline[key]
		
			hist_sum = np.sum([i for i in self._nuhist.itervalues()], axis=0)
		
			# calculate ratio and spline this
			for key, hist in self._nuhist.iteritems():
				rel_hist = np.log(hist) - np.log(hist_sum)
		
				self._nuspline[key] = scipy.interpolate.RectBivariateSpline(self._sindec_binmids, self._gamma_binmids,rel_hist, kx=2, ky=2, s=0)
		
		out_dict = dict([(key, np.exp(val(np.sin(src_dec), gamma))) for key, val in self._nuspline.iteritems()])
		
		return out_dict
