"""
Notes.

TODO: add gaussian noise on top of the biology before imaging noise

-rise time of gcamp is considered constant as of now
-when the stochastic parameters are implemented, often abs() is used, meaning they are no longer truly gaussian but half gaussian
-clusters centers are constrained to within the final FOV, though cells can still end up outside. importantly, soma density is only a rough parameter. number of clusters inside FOV will be on average the number specified
"""

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


class Simulation(object):
    MEAN,STD = 0,1
    Y,X = 0,1

    def __init__(self, fname='noname'):
        self.fname = fname
        self.has_saved = False

        # time and space
        self.Ds = 1.1 #micrometers/pixel
        self.image_size_final = [64, 128] #pixels
        self.jitter_pad = [20, 20] #pixels
        self.jitter_lambda = 1.0 #poisson
        self.image_size = [i+j for i,j in zip(self.image_size_final, self.jitter_pad)]
        self.field_size = [self.Ds*i for i in self.image_size] #micrometers
        self.field_size_final = [self.Ds*i for i in self.image_size_final] #micrometers
        self.image_placement = [[np.floor(j/2.) for j in self.jitter_pad], [isz-np.ceil(j/2.) for isz,j in zip(self.image_size,self.jitter_pad)]] #[first, last]
        self.Ts = 0.064 #s/frame

        # the biological tissue
        self.soma_radius = [3., 0.2] #micrometers
        self.soma_circularity_noise_world = [0., 0.15] #micrometers
        self.soma_circularity_noise = [ss/self.Ds for ss in self.soma_circularity_noise_world] #pixels
        self.nucleus_radius = [0.45, 0.05] #as proportion of soma radius. in application, constrains std to only decrease but not increase size
        self.soma_density_field = 100 #cells per *final* frame area
        self.soma_clusters_density_field = 6 #*expected* cluster per *final* frame area
        self.soma_clusters_density = self.soma_clusters_density_field / np.product(self.field_size_final) #clusters/micrometer_squared
        self.soma_cluster_spread = [5., 10.] #distance from cluster center, as multiple of mean expected soma radius
        self.ca_rest = 0.050 #micromolar
        self.neuropil_density = 1.0 #neuropil probability at any given point

        # the imaging equipment
        self.imaging_background = 0.1
        self.imaging_noise_lam = 3.0 #if shot
        self.imaging_noise_mag = 1.05 #if shot, when movie is 0-1.0
        self.imaging_noise = [0.0, 0.2] #if gaussian, when movie is 0-1.0
        self.imaging_filter_sigma = [0., 0.2, 0.2]
        self.noise_type = 'gauss' #shot or gauss

        # indicator
        self.tau_gcamp_rise = 0.084 #s (58ms t1/2 rise dvidied by ln2)
        self.tau_gcamp_decay = 0.102 #s (71ms t1/2 decay divided by ln2)
        self.gcamp_kd = 0.29 #micromolar
        self.gcamp_nh = 2.7 #hill coef
        self.gcamp_rf = 29.

        # the biological event
        self.tau_ca_decay = [0.250, 0.050] #s
        self.ca_per_ap = 0.010 #micromolar
        self.stim_onset = 0.5 #s
        self.stim_f = 200. #spikes/s
        self.stim_dur = [0.100, 0.100] #s
        self.stim_gap = 1.5 #s
        self.stim_n = 30
        self.stim_durs = np.array([self.stim_dur[0] for _ in xrange(self.stim_n)])#rand.normal(*self.stim_dur, size=self.stim_n) #this is for random durations
        self.duration = (self.stim_onset + self.stim_gap) * self.stim_n + np.sum(self.stim_durs) #s
        self.cell_timing_offset = [0.050, 0.030] #seconds
        # these are working on values from 0-1:
        self.cell_magnitude = [1.0, 0.01] #magnitude of cells' *ca* response amplitudes relative to each other
        self.cell_baseline = [1.0, 0.01] #magnitude of cells' *ca* baseline values relative to each other. This is a multiplier to the bseline Ca
        self.cell_expression = [1.0, 5.00] #note that this refers to indicator while magnitude and baseline refer to calcium. it's the baseline *and* magnitude of a cell's fluorescent response relative to other cells (i.e. a multiplying factor for converting ca to f). interpreted as expression
        self.cell_f_strength = [0.0, 0.000001] #this is an additive offset for fluorescence, for instance, if a cell is just generally brighter because it's closer to the surface
        self.neuropil_mag = 1.0 #as a fraction of average cell magnitude
        self.neuropil_baseline = 0.9 #as a fraction of average cell baseline
        self.neuropil_expression = 2.0 #as a multiple of the "avg" (but not truly avg) cell_expression
        self.incell_ca_dist_noise = [-1, 0.1] #distribution of ca/fluo within cell, mean is mean of cell signal, 2nd value is fraction of that to make std
        self.npil_ca_dist_noise = [-1, 2.5]
    @property
    def soma_density_field(self):
        return self._soma_density_field
    @soma_density_field.setter
    def soma_density_field(self,val):
        self._soma_density_field = val
        self.soma_density = self._soma_density_field / np.product(self.field_size_final) #cells/micrometer_squared
    def ca2f(self, ca):
        def response_curve(c): 
            fmin = self.gcamp_rf / (self.gcamp_rf - 1)
            return fmin + self.gcamp_rf*(c**self.gcamp_nh)/(self.gcamp_kd + c**self.gcamp_nh)
        #convert a calcium concentration to an ideal fluorescence maximal value
        f = response_curve(ca)
        return f

    def generate_stim(self, shift):
        #time array, onset time, frequency of APs, duration of stim
        #assumes t has constant sampling rate
        t = self.t
        onsets = [(self.stim_onset+self.stim_gap)*i + np.sum(self.stim_durs[:i]) + self.stim_onset for i in xrange(self.stim_n)]
        onsets = [o + abs(shift) for o in onsets]
        self.dur_idxs = np.round([sd/self.Ts for sd in self.stim_durs])

        stim = np.zeros_like(t)
        self.idxs_start = np.round(np.array(onsets)/self.Ts)
        self.idxs_end = self.idxs_start + self.dur_idxs
        if self.stim_f*self.Ts > 1.0: #more than one spike per sample
            self.stim_f_use = 1/self.Ts
            self.sps = self.stim_f/self.stim_f_use
        else:
            self.sps = 1.0
            self.stim_f_use = self.stim_f
        idxs = np.concatenate([np.arange(idx_start, idx_end, np.rint(1./(self.stim_f_use*self.Ts)), dtype=int) for idx_start,idx_end in zip(self.idxs_start, self.idxs_end)])
        idxs = [i for i in idxs if i<len(stim)]
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
        return fbl + f * gain

    def generate_cells(self):
        n_clusters = np.rint(self.soma_clusters_density * np.product(self.field_size))
        n_cells = np.rint(self.soma_density * np.product(self.field_size))
        
        self.cluster_centers = [[rand.uniform(0,i) for i in self.field_size] for cl in np.arange(n_clusters)] #in Ds units
        self.cluster_centers_im = [list(np.array(cs)-np.array(self.image_placement[0])) for cs in self.cluster_centers] #also in Ds units
        self.cluster_centers_im_,self.cluster_centers_ = [np.rint(np.array(i)/self.Ds) for i in [self.cluster_centers_im, self.cluster_centers]] #in pixels
        cells = []
        for cl in self.cluster_centers:
            for c in np.arange(np.rint(n_cells/n_clusters)):
                cell = Cell(self)
                miny,minx,maxy,maxx = 1,1,0,0
                while miny>=maxy:
                    miny = max(0, cl[self.Y]-abs(rand.normal(*self.soma_cluster_spread)*self.soma_radius[0]))
                    maxy = min(self.field_size[self.Y], cl[self.Y]+abs(rand.normal(*self.soma_cluster_spread)*self.soma_radius[0]))
                    miny = round(miny)
                    maxy = round(maxy)
                while minx>=maxx:
                    minx = max(0, cl[self.X]-abs(rand.normal(*self.soma_cluster_spread)*self.soma_radius[0]))
                    maxx = min(self.field_size[self.X], cl[self.X]+abs(rand.normal(*self.soma_cluster_spread)*self.soma_radius[0]))
                    minx = round(minx)
                    maxx = round(maxx)
                cell.center = [rand.randint(qmin,qmax) for qmin,qmax in [(miny,maxy),(minx,maxx)]]
                cell.radius = rand.normal(*self.soma_radius)
                cell.nuc_radius = min(self.nucleus_radius[0]*cell.radius, rand.normal(*(np.array(self.nucleus_radius)*cell.radius)))
                cell.mag = abs(rand.normal(*self.cell_magnitude))
                cell.baseline = abs(rand.normal(*self.cell_baseline)) #will be the multiplying factor to the standard ca baseline
                cell.tau_cdecay = max(1.e-10, rand.normal(*self.tau_ca_decay))
                cell.expression = abs(rand.normal(*self.cell_expression))
                cell.f_strength = abs(rand.normal(*self.cell_f_strength))
                cell.offset = rand.normal(*self.cell_timing_offset)
                cell.compute_mask()
                cells.append(cell)
        self.n_cells_in_fov = sum([c.was_in_fov for c in cells])
        return cells
    def generate_neuropil(self):
        npil = Cell(self)
        available_pts = np.argwhere(np.logical_not(np.sum([c.mask_with_nucleus for c in self.cells],axis=0)))
        npil.mask = np.zeros(self.image_size, dtype=bool)
        for ap in available_pts:
            npil.mask[ap[0],ap[1]] = bool(rand.random()<self.neuropil_density)
        idx0,idx1 = self.image_placement
        npil.mask_im = npil.mask[idx0[0]:idx1[0], idx0[1]:idx1[1]] #mask once movie has been cropped
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

    def save_mov(self, fmt='avi', dest='.'):
        if not self.has_saved:
            #must specify a unique new directory name for this output
            if not os.path.exists(dest):
                os.makedirs(dest)
            sim_dir = os.path.join(dest,self.fname)
            i = 1
            while os.path.exists(sim_dir):
                sim_dir = os.path.join(dest,self.fname+'_%i'%i)
                print sim_dir
                i += 1
            os.mkdir(sim_dir)
            self.sim_dir = sim_dir
            self.has_saved = True
        mov = self.mov
        fname = os.path.join(self.sim_dir,self.fname + '.' + fmt)
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

    def save_data(self, fmt='npy', dest='.'):
        if not self.has_saved:
            #must specify a unique new directory name for this output
            if not os.path.exists(dest):
                os.makedirs(dest)
            sim_dir = os.path.join(dest,self.fname)
            i = 1
            while os.path.exists(sim_dir):
                sim_dir = os.path.join(dest,self.fname,'-%i'%i)
                i += 1
            os.mkdir(sim_dir)
            self.sim_dir = sim_dir
            self.has_saved = True
        fname = os.path.join(self.sim_dir, self.fname)
        params = {k:self.__dict__[k] for k in self.__dict__ if k not in ['cells','neuropil','t','mov','mov_nofilter','stim','mov_nojit','mov_filtered']}
        cells = [cell.get_dict() for cell in self.cells]
        npil = np.array([self.neuropil.get_dict()])
        t = self.__dict__['t']
        stim = self.__dict__['stim']
        movie = np.array([{k:self.__dict__[k] for k in ['mov','mov_nofilter','mov_nojit','mov_filtered']}])
        if fmt in ['npy','npz','numpy','n']:
            np.savez(fname, params=params, cells=cells, neuropil=npil, time=t, stim=stim, movie=movie)
        elif fmt in ['mat','matlab','m']:
            matdic = {'params':params, 'cells':cells, 'neuropil':npil, 'time':t, 'stim':stim, 'movie':movie}
            savemat(fname, matdic)

    def image(self, seq):
        t = self.t
        cells = self.cells
        self.mov_nojit = seq
        idx0,idx1 = self.image_placement
        mov = np.empty((len(self.mov_nojit),self.image_size_final[0],self.image_size_final[1]))
        for fidx,frame in enumerate(self.mov_nojit):
            sn = rand.choice([-1., 1.], size=2)
            jmag = rand.choice([0.,1.],size=2)# rand.poisson(jitter_lambda, size=2)
            jity,jitx = 0,0 #sn*jmag
            mov[fidx,:,:] = frame[idx0[0]+jity:idx1[0]+jity, idx0[1]+jitx:idx1[1]+jitx]
        self.mov_nofilter = mov
        self.mov_filtered = self.normalize(gaussian_filter(mov, self.imaging_filter_sigma))
        shot_noise = rand.poisson(self.imaging_noise_lam, size=mov.shape) #shot
        shot_noise = self.imaging_noise_mag * shot_noise/np.max(shot_noise) #shot
        gauss_noise = rand.normal(*self.imaging_noise, size=mov.shape) # gaussian
        if self.noise_type == 'shot':
            noise = shot_noise
        elif self.noise_type == 'gauss':
            noise = gauss_noise
        self.mov = self.mov_filtered + noise

        return self.mov
    def store_noisy(self):
        for cell in self.cells:
            if np.any(cell.mask_im) and np.any(cell.mask_im_with_nucleus):
                cell.fluo_with_noise = np.mean(self.mov[:,cell.mask_im],axis=1)
                cell.fluo_with_noise_with_nucleus = np.mean(self.mov[:,cell.mask_im_with_nucleus],axis=1)
            else:
                cell.fluo_with_noise = np.array([])
                cell.fluo_with_noise_with_nucleus = np.array([])
        self.neuropil.fluo_with_noise = np.mean(self.mov[:,self.neuropil.mask_im],axis=1)
    def generate_movie(self):
        self.t = np.arange(0., self.duration, self.Ts)
        seq = self.imaging_background * np.ones((len(self.t), self.image_size[self.Y], self.image_size[self.X]))
        self.cells = self.generate_cells()
        self.neuropil = self.generate_neuropil()
        
        for cell in self.cells:
            cell.stim = self.generate_stim(cell.offset)
            cell.ca = self.generate_ca(cell.stim, cell.tau_cdecay, cell.mag, cell.baseline)
            cell.fluo = self.generate_fluo(cell.ca, cell.expression, cell.f_strength)
        
        self.stim = self.generate_stim(0.)
        self.neuropil.stim = self.stim
        self.neuropil.ca = self.generate_ca(self.stim, self.neuropil.tau_cdecay, self.neuropil_mag, self.neuropil_baseline)
        self.neuropil.fluo = self.generate_fluo(self.neuropil.ca, self.neuropil_expression*self.cell_expression[0])
        
        seq = self.construct(seq,self.cells,self.neuropil)
        
        seq = self.normalize(seq)
        mov = self.image(seq)
        mov = np.rint(self.normalize(mov)*255.).astype(np.uint8)
        self.mov = mov
        self.store_noisy()

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
        self.center_im_ = np.array(self.center_)-np.array(self.sim.image_placement[0])
        self.center_im = self.center_im_ * self.sim.Ds
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
        y,x = self.sim.image_size
        cy,cx = self.center_
        r = self.radius_
        nr = self.nuc_radius_
        dy,dx = np.ogrid[-cy:y-cy, -cx:x-cx]
        pos_array = dy*dy + dx*dx
        pos_array += rand.normal(*self.sim.soma_circularity_noise, size=pos_array.shape)
        pos_array = np.rint(pos_array)
        self.mask = pos_array <= r*r
        self.mask_with_nucleus = np.copy(self.mask)
        self.mask[pos_array <= nr*nr] = False
        
        idx0,idx1 = self.sim.image_placement
        self.mask_im = self.mask[idx0[0]:idx1[0], idx0[1]:idx1[1]] #mask once movie has been cropped
        self.mask_im_with_nucleus = self.mask_with_nucleus[idx0[0]:idx1[0], idx0[1]:idx1[1]]
        self.was_in_fov = bool(np.sum(self.mask_im_with_nucleus))
    def get_dict(self):
        return {k:self.__dict__[k] for k in self.__dict__ if k not in ['sim']}

if __name__ == '__main__':
    sim = Simulation('test_mov')
    sim.generate_movie()
    sim.save_mov(fmt='tif')
    sim.save_data(fmt='npy', dest='.')
    sim.save_data(fmt='mat', dest='.')
