"""
Notes.

-rise time of gcamp is considered constant as of now
-when the stochastic parameters are implemented, often abs() is used, meaning they are no longer truly gaussian
"""

import cPickle as pickle
import pylab as pl
import cv2
import numpy as np
import os
rand = np.random
from scipy.io import savemat
from scipy.ndimage.filters import gaussian_filter
try:
    from tifffile import imsave as timsave
except:
    pass

class Cell(object):
    #trailing underscores refer to pixels, otherwise units of Ds (ex. micrometers)
    def __init__(self, sim):
        self._center = 0.0
        self._radius = 1.0
        self._nuc_radius = 1.0
        self.sim = sim
    @property
    def center(self):
        return self._center
    @center.setter
    def center(self, val):
        val = np.array(val)
        self._center = val
        self.center_ = np.rint(val / self.sim.Ds).astype(int)
    @property
    def radius(self):
        return self._radius
    @radius.setter
    def radius(self, val):
        self._radius = val
        self.radius_ = np.rint(val / self.sim.Ds).astype(int)
    @property
    def nuc_radius(self):
        return self._nuc_radius
    @nuc_radius.setter
    def nuc_radius(self, val):
        self._nuc_radius = val
        self.nuc_radius_ = np.rint(val / self.sim.Ds).astype(int)
    def compute_mask(self):
        y,x = sim.image_size
        cy,cx = self.center_
        r = self.radius_
        nr = self.nuc_radius_
        dy,dx = np.ogrid[-cy:y-cy, -cx:x-cx]
        pos_array = dy*dy + dx*dx
        pos_array += rand.normal(*sim.soma_circularity_noise, size=pos_array.shape)
        pos_array = np.rint(pos_array)
        self.mask = pos_array <= r*r
        self.mask_with_nucleus = np.copy(self.mask)
        self.mask[pos_array <= nr*nr] = False
        
        idx0 = [np.floor(j/2.) for j in sim.jitter_pad] 
        idx1 = [isz-np.ceil(j/2.) for isz,j in zip(sim.image_size,sim.jitter_pad)]
        self.mask_im = self.mask[idx0[0]:idx1[0], idx0[1]:idx1[1]]
        self.mask_im_with_nucleus = self.mask_with_nucleus[idx0[0]:idx1[0], idx0[1]:idx1[1]]
        self.was_in_fov = bool(np.sum(self.mask_im_with_nucleus))

class Simulation(object):
    MEAN,STD = 0,1
    Y,X = 0,1

    def __init__(self, fname='noname'):
        self.fname = fname

        # time and space
        self.Ds = 1.1 #micrometers/pixel
        self.image_size_final = [32, 128] #pixels
        self.jitter_pad = [20, 20] #pixels
        self.jitter_lambda = 1.0 #poisson
        self.image_size = [i+j for i,j in zip(self.image_size_final, self.jitter_pad)]
        self.field_size = [self.Ds*i for i in self.image_size] #micrometers
        self.Ts = 0.064 #s/frame

        # the biological tissue
        self.soma_radius = [3., 0.2] #micrometers
        self.soma_circularity_noise_world = [0., 0.15] #micrometers
        self.soma_circularity_noise = [ss/self.Ds for ss in self.soma_circularity_noise_world] #pixels
        self.nucleus_radius = [0.45, 0.05] #as proportion of soma radius. in application, constrains std to only decrease but not increase size
        self.soma_density_field = 80 #cells per frame area
        self.soma_density = self.soma_density_field / np.product(self.field_size) #cells/micrometer_squared
        self.ca_rest = 0.050 #micromolar
        self.neuropil_density = 0.8 #neuropil probability at any given point

        # the imaging equipment
        self.imaging_background = 0.1
        self.imaging_noise_lam = 3.0
        self.imaging_noise_mag = 1.5 #when movie is 0-1.0
        self.imaging_filter_sigma = [0., 0.2, 0.2]

        # indicator
        self.tau_gcamp_rise = 0.084 #s (58ms t1/2 rise dvidied by ln2)
        self.tau_gcamp_decay = 0.102 #s (71ms t1/2 decay divided by ln2)
        self.gcamp_kd = 0.29 #micromolar
        self.gcamp_nh = 2.46 #hill coef

        # the biological event
        self.tau_ca_decay = [0.250, 0.050] #s
        self.ca_per_ap = 0.200 #micromolar
        self.stim_onset = 0.5 #s
        self.stim_f = 250. #spikes/s
        self.stim_dur = 0.500 #s
        self.stim_gap = 1.5 #s
        self.stim_n = 8
        self.duration = (self.stim_onset + self.stim_dur + self.stim_gap) * self.stim_n #s
        self.cell_timing_offset = [0.050, 0.030] #seconds
        # these are working on values from 0-1:
        self.cell_magnitude = [1.0, 0.01] #magnitude of cells' *ca* response amplitudes relative to each other
        self.cell_baseline = [1.0, 0.01] #magnitude of cells' *ca* baseline values relative to each other. This is a multiplier to the bseline Ca
        self.cell_gain = [1.0, 0.01] #note that this refers to indicator while magnitude and baseline refer to calcium. it's the baseline *and* magnitude of a cell's fluorescent response relative to other cells (i.e. a multiplying factor for converting ca to f)
        self.cell_expression = [0.0, 100.0] #this value also refers to indicator and not ca. Importantly, it represents the proportion of the global resting *fluorescence* level that will be *added* to the fluorescence
        self.neuropil_mag = 0.9 #as a fraction of average cell magnitude
        self.neuropil_baseline = 0.9 #as a fraction of average cell baseline
        self.incell_ca_dist_noise = [-1, 0.1] #distribution of ca/fluo within cell, mean is mean of cell signal, 2nd value is fraction of that to make std
        self.npil_ca_dist_noise = [-1, 1.0]

    def ca2f(self, ca):
        def response_curve(c): 
            return (c**self.gcamp_nh)/(self.gcamp_kd + c**self.gcamp_nh)
        #convert a calcium concentration to an ideal fluorescence maximal value
        f = response_curve(ca)
        return f

    def generate_stim(self, shift):
        #time array, onset time, frequency of APs, duration of stim
        #assumes t has constant sampling rate
        t = self.t
        onsets = [(self.stim_onset+self.stim_dur+self.stim_gap)*i + self.stim_onset for i in xrange(self.stim_n)]
        onsets = [o + abs(shift) for o in onsets]
         
        stim = np.zeros_like(t)
        self.idxs_start = [np.argmin(np.abs(onset-t)) for onset in onsets]
        self.idxs_end = [np.argmin(np.abs((onset+self.stim_dur)-t)) for onset in onsets]
        if self.stim_f*self.Ts > 1.0: #more than one spike per sample
            self.stim_f_use = 1/self.Ts
            self.sps = self.stim_f/self.stim_f_use
        else:
            self.sps = 1.0
            self.stim_f_use = self.stim_f
        idxs = np.concatenate([np.arange(idx_start, idx_end, np.rint(1./(self.stim_f_use*self.Ts)), dtype=int) for idx_start,idx_end in zip(self.idxs_start, self.idxs_end)])
        stim[idxs] = self.sps
        return stim

    def generate_ca(self, stim, tau_decay, mag, bl):
        t = self.t
        ca = np.zeros_like(t)
        bsl = bl*self.ca_rest
        ca[0] = bsl
        for idx,dt in zip(xrange(1,len(ca)), t[1:]-t[:-1]):
            ca[idx] = bsl + (ca[idx-1] - bsl) * np.exp(-dt/tau_decay) + stim[idx-1]*self.ca_per_ap*mag
        return ca

    def generate_fluo(self, ca, gain=1.0, fbl=0.0):
        t = self.t
        f_ideal = np.zeros_like(ca)
        fbsl = fbl*self.ca2f(self.ca_rest)
        f_ideal[0] = self.ca2f(ca[0])
        f = np.zeros_like(f_ideal)
        f[0] = f_ideal[0]

        for idx,dt in zip(xrange(1,len(ca)), t[1:]-t[:-1]):
            f_ideal[idx] = self.ca2f(ca[idx])
            if f_ideal[idx-1] > f[idx-1]: #on the rise
                f[idx] = min(f[idx-1] + dt/self.tau_gcamp_rise * (f_ideal[idx-1] - f[idx-1]), f_ideal[idx-1])
            elif f_ideal[idx-1] < f[idx-1]: #on the decay
                f[idx] = max(f[idx-1] + dt/self.tau_gcamp_decay * (f_ideal[idx-1] - f[idx-1]), f_ideal[idx-1])
            else: #not rising or decaying
                f[idx] = f[idx-1]

        return fbsl + f * gain

    def generate_cells(self):
        n_cells = np.rint(self.soma_density * np.product(self.field_size))
        cells = []
        for c in np.arange(n_cells):
            cell = Cell(self)
            cell.center = [rand.randint(0,i) for i in self.field_size]
            cell.radius = rand.normal(*self.soma_radius)
            cell.nuc_radius = min(self.nucleus_radius[0]*cell.radius, rand.normal(*(np.array(self.nucleus_radius)*cell.radius)))
            cell.mag = abs(rand.normal(*self.cell_magnitude))
            cell.baseline = abs(rand.normal(*self.cell_baseline)) #will be the multiplying factor to the standard ca baseline
            cell.tau_cdecay = max(1.e-10, rand.normal(*self.tau_ca_decay))
            cell.gain = abs(rand.normal(*self.cell_gain))
            cell.expression = abs(rand.normal(*self.cell_expression))
            cell.offset = rand.normal(*self.cell_timing_offset)
            cell.compute_mask()
            cells.append(cell)
        return cells
    def generate_neuropil(self):
        npil = Cell(self)
        npil.mask = rand.random(self.image_size)<self.neuropil_density
        npil.tau_cdecay = max(1.e-10,rand.normal(*self.tau_ca_decay))
        return npil

    def construct(self, seq, cells, npil):
        for cell in cells:
            cell_fluo = np.array([cell.fluo]*np.sum(cell.mask)).transpose()
            m = np.mean(cell.fluo)
            noise = rand.normal(m, self.incell_ca_dist_noise[1]*m, size=cell_fluo.shape[1])
            cell_fluo = cell_fluo+noise
            cell.fluo_by_pixel = cell_fluo
            seq[:, cell.mask] += cell_fluo

        npil_fluo = np.array([npil.fluo]*np.sum(npil.mask)).transpose()
        m = np.mean(npil.fluo)
        noise = rand.normal(m, self.npil_ca_dist_noise[1]*m, size=npil_fluo.shape[1])
        npil_fluo += noise
        npil.fluo_by_pixel = npil_fluo
        seq[:, npil.mask] += npil_fluo #add neuropil
        
        return seq
    def normalize(self, seq):
        if 0==1 and len(seq.shape) == 3: #normalizes each frame to itself
            mins = np.apply_over_axes(np.min, seq, [1,2])
            seq = seq + abs(mins)
            maxs = np.apply_over_axes(np.max, seq, [1,2])
        elif len(seq.shape) == 3: #normalizes entire 3d matrix to its highest value
            minval = np.min(seq)
            seq = seq + abs(minval)
            maxval = np.max(seq)
            maxs = maxval
        elif len(seq.shape) < 3:
            mins = np.min(seq, axis=0)
            seq = seq.mins
            maxs = np.max(seq, axis=0)
        seq = seq / maxs
        return seq

    def save_mov(self, fmt='avi', dest=''):
        mov = self.mov
        fname = os.path.join(dest,self.fname + '.' + fmt)
        if fmt=='avi':
            vw = cv2.VideoWriter(fname, fourcc=cv2.cv.CV_FOURCC('m', 'p', '4', 'v'), fps=int(1./self.Ts), frameSize=tuple(self.image_size_final[::-1]))
            for fr in mov:
                vw.write(cv2.cvtColor(fr,cv2.cv.CV_GRAY2RGB))
            vw.release()
        elif fmt=='tiff' or fmt=='tif':
            try:
                #mov = np.rollaxis(mov,0,3) #change time axis to last
                timsave(fname, mov)
            except:
                raise Exception('No working module for tiff saving.')

    def save_data(self, dest=''):
        fname = os.path.join(dest, self.fname + '.data')
        pickle.dump(self, open(fname,'wb'))

    def image(self, seq):
        t = self.t
        cells = self.cells
        mov_nojit = seq
        idx0 = [np.floor(j/2.) for j in self.jitter_pad] 
        idx1 = [isz-np.ceil(j/2.) for isz,j in zip(self.image_size,self.jitter_pad)]
        mov = np.empty((len(mov_nojit),self.image_size_final[0],self.image_size_final[1]))
        for fidx,frame in enumerate(mov_nojit):
            sn = rand.choice([-1., 1.], size=2)
            jmag = rand.choice([0.,1.],size=2)# rand.poisson(jitter_lambda, size=2)
            jity,jitx = sn*jmag
            mov[fidx,:,:] = frame[idx0[0]+jity:idx1[0]+jity, idx0[1]+jitx:idx1[1]+jitx]
        self.mov_nofilter = mov
        self.mov_filtered = gaussian_filter(mov, self.imaging_filter_sigma)
        noise = rand.poisson(self.imaging_noise_lam, size=mov.shape)
        noise = self.imaging_noise_mag * noise/np.max(noise)
        self.mov = self.mov_filtered + noise

        for cell in cells:
            cell.fluo_with_noise = np.mean(mov[:,cell.mask_im],axis=1)
            cell.fluo_with_noise_with_nucleus = np.mean(mov[:,cell.mask_im_with_nucleus],axis=1)

        return self.mov

    def generate_movie(self):
        self.t = np.arange(0., self.duration, self.Ts)
        seq = self.imaging_background * np.ones((len(self.t), self.image_size[self.Y], self.image_size[self.X]))
        self.cells = self.generate_cells()
        self.neuropil = self.generate_neuropil()
        
        for cell in self.cells:
            stim = self.generate_stim(cell.offset)
            cell.ca = self.generate_ca(stim, cell.tau_cdecay, cell.mag, cell.baseline)
            cell.fluo = self.generate_fluo(cell.ca, cell.gain, cell.expression)

        self.stim = self.generate_stim(0.)
        self.neuropil.ca = self.generate_ca(self.stim, self.neuropil.tau_cdecay, self.neuropil_mag, self.neuropil_baseline)
        self.neuropil.fluo = self.generate_fluo(self.neuropil.ca)

        seq = self.construct(seq,self.cells,self.neuropil)

        seq = self.normalize(seq)
        mov = self.image(seq)
        mov = np.rint(self.normalize(mov)*255.).astype(np.uint8)
        self.mov = mov
        return mov

if __name__ == '__main__':
    sim = Simulation('test_mov_1435')
    sim.generate_movie()
    sim.save_mov(fmt='tif',dest='')
    sim.save_data(dest='')
