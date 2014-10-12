"""
Notes.

-rise time of gcamp is considered constant as of now
-when the stochastic parameters are implemented, often abs() is used, meaning they are no longer truly gaussian
"""

import pylab as pl
import cv2
import numpy as np
import os
rand = np.random
from scipy.io import savemat
try:
    from tifffile import imsave as timsave
except:
    pass

MEAN,STD = 0,1
Y,X = 0,1

# time and space
Ds = 0.9 #micrometers/pixel
image_size = [32, 128] #pixels
field_size = [Ds*i for i in image_size] #micrometers
Ts_world = 1. * 10.**-3 #s/sample
Ts_microscope_pixels = 10.31*10.**-6. #s/pixel
Ts_microscope = Ts_microscope_pixels * np.product(image_size) #s/sample

# the biological tissue
soma_radius = [3., 0.3] #micrometers
soma_circularity_noise_world = [0., 2.] #micrometers
soma_circularity_noise = [ss/Ds for ss in soma_circularity_noise_world] #pixels
soma_density_field = 8#cells per frame area
soma_density = soma_density_field / np.product(field_size) #cells/micrometer_squared
ca_rest = 0.050 #micromolar
neuropil_density = 1.0 #neuropil probability at any given point

# the imaging equipment
imaging_background = 0.1
imaging_noise = [0., 0.08] #when movie is 0-1.0

# indicator
tau_gcamp_rise = 0.084 #s (58ms t1/2 rise dvidied by ln2)
tau_gcamp_decay = 0.102 #s (71ms t1/2 decay divided by ln2)
gcamp_kd = 0.29 #micromolar
gcamp_nh = 2.46 #hill coef
#gcamp_response_ca = [1.0*i for i in [0.0, 0.0002, 0.0005, 0.0009, 0.0014, 0.0019, 0.0027, 0.0036, 0.0047, 0.0061, 0.0079, 0.0101, 0.0128, 0.0163, 0.0206, 0.2394]]
#gcamp_response_dff = [i/3.2844 for i in [0.1268, 0.1224, 0.1602, 0.2596, 0.5133, 0.9588, 1.5930, 2.1132, 2.5762, 2.8900, 3.1594, 3.2259, 3.2941, 3.3160, 3.2729, 3.2844]]

# the biological event
tau_ca_decay = 0.100 #s
ca_per_ap = 0.200 #micromolar
stim_onset = 0.5 #s
stim_f = 250. #spikes/s
stim_dur = 0.100 #s
stim_gap = 1.5 #s
stim_n = 8
duration = (stim_onset + stim_dur + stim_gap) * stim_n #s
cell_timing_offset = [0.050, 0.025] #seconds
# these are working on values from 0-1:
cell_magnitude = [1.0, 0.01] #magnitude of cells' *ca* response amplitudes relative to each other
cell_baseline = [1.0, 0.01] #magnitude of cells' *ca* baseline values relative to each other. This is a multiplier to the bseline Ca
cell_expression = [1.0, 0.01] #note that this refers to indicator while magnitude and baseline refer to calcium. it's the baseline *and* magnitude of a cell's fluorescent response relative to other cells (i.e. a multiplying factor for converting ca to f)
cell_fluo_baseline = [0.0, 100.0] #this value also refers to indicator and not ca. Importantly, it represents the proportion of the global resting *fluorescence* level that will be *added* to the fluorescence
neuropil_mag = 0.9 #as a fraction of average cell magnitude
neuropil_baseline = 0.9 #as a fraction of average cell baseline
incell_ca_dist_noise = [-1, 0.1] #distribution of ca/fluo within cell, mean is mean of cell signal, 2nd value is fraction of that to make std
npil_ca_dist_noise = [-1, 1.0]

def ca2f(ca):
    def response_curve(c): 
        return (c**gcamp_nh)/(gcamp_kd + c**gcamp_nh)
    #convert a calcium concentration to an ideal fluorescence maximal value
    f = response_curve(ca)
    return f

def generate_stim(t, shift):
    #time array, onset time, frequency of APs, duration of stim
    #assumes t has constant sampling rate
    onsets = [(stim_onset+stim_dur+stim_gap)*i + stim_onset for i in xrange(stim_n)]
    onsets = [o + abs(shift) for o in onsets]

    stim = np.zeros_like(t)
    idxs_start = [np.argmin(np.abs(onset-t)) for onset in onsets]
    idxs_end = [np.argmin(np.abs((onset+stim_dur)-t)) for onset in onsets]
    idxs = np.concatenate([np.arange(idx_start, idx_end, np.rint(1./(stim_f*Ts_world)), dtype=int) for idx_start,idx_end in zip(idxs_start, idxs_end)])
    stim[idxs] = 1.
    return stim

def generate_ca(t, stim, mag, bl):
    ca = np.zeros_like(t)
    bsl = bl*ca_rest
    ca[0] = bsl
    for idx,dt in zip(xrange(1,len(ca)), t[1:]-t[:-1]):
        ca[idx] = bsl + (ca[idx-1] - bsl) * np.exp(-dt/tau_ca_decay) + stim[idx-1]*ca_per_ap*mag
    return ca

def generate_fluo(t, ca, expression=1.0, fbl=0.0):
    f_ideal = np.zeros_like(ca)
    fbsl = fbl*ca2f(ca_rest)
    f_ideal[0] = ca2f(ca[0])
    f = np.zeros_like(f_ideal)
    f[0] = f_ideal[0]

    for idx,dt in zip(xrange(1,len(ca)), t[1:]-t[:-1]):
        f_ideal[idx] = ca2f(ca[idx])
        if f_ideal[idx-1] > f[idx-1]: #on the rise
            f[idx] = min(f[idx-1] + dt/tau_gcamp_rise * (f_ideal[idx-1] - f[idx-1]), f_ideal[idx-1])
        elif f_ideal[idx-1] < f[idx-1]: #on the decay
            f[idx] = max(f[idx-1] + dt/tau_gcamp_decay * (f_ideal[idx-1] - f[idx-1]), f_ideal[idx-1])
        else: #not rising or decaying
            f[idx] = f[idx-1]

    return fbsl + f * expression

class Cell(object):
    #trailing underscores refer to pixels, otherwise units of Ds (ex. micrometers)
    def __init__(self):
        self._center = 0.0
        self._radius = 1.0
    @property
    def center(self):
        return self._center
    @center.setter
    def center(self, val):
        val = np.array(val)
        self._center = val
        self.center_ = np.rint(val / Ds).astype(int)
    @property
    def radius(self):
        return self._radius
    @radius.setter
    def radius(self, val):
        self._radius = val
        self.radius_ = np.rint(val / Ds).astype(int)
    def compute_mask(self):
        y,x = image_size
        cy,cx = self.center_
        r = self.radius_
        dy,dx = np.ogrid[-cy:y-cy, -cx:x-cx]
        pos_array = dy*dy + dx*dx
        pos_array += rand.normal(*soma_circularity_noise, size=pos_array.shape)
        pos_array = np.rint(pos_array)
        self.mask = pos_array <= r*r
def generate_cells():
    n_cells = np.rint(soma_density * np.product(field_size))
    cells = []
    for c in np.arange(n_cells):
        cell = Cell()
        cell.center = [rand.randint(0,i) for i in field_size]
        cell.radius = rand.normal(*soma_radius)
        cell.mag = abs(rand.normal(*cell_magnitude))
        cell.baseline = abs(rand.normal(*cell_baseline)) #will be the multiplying factor to the standard ca baseline
        cell.expression = abs(rand.normal(*cell_expression))
        cell.fluo_baseline = abs(rand.normal(*cell_fluo_baseline))
        cell.offset = rand.normal(*cell_timing_offset)
        cell.compute_mask()
        cells.append(cell)
    return cells
def generate_neuropil():
    npil = Cell()
    npil.mask = rand.random(image_size)<neuropil_density
    return npil

def construct(seq, cells, npil):
    for cell in cells:
        cell_fluo = np.array([cell.fluo]*np.sum(cell.mask)).transpose()
        m = np.mean(cell.fluo)
        noise = rand.normal(m, incell_ca_dist_noise[1]*m, size=cell_fluo.shape[1])
        cell_fluo = cell_fluo+noise
        seq[:, cell.mask] += cell_fluo

    npil_fluo = np.array([npil.fluo]*np.sum(npil.mask)).transpose()
    m = np.mean(npil.fluo)
    noise = rand.normal(m, npil_ca_dist_noise[1]*m, size=npil_fluo.shape[1])
    npil_fluo += noise
    seq[:, npil.mask] += npil_fluo #add neuropil
    
    return seq
def normalize(seq):
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

def save_mov(mov, fname, fmt='avi'):
    fname = fname + '.' + fmt
    if fmt=='avi':
        vw = cv2.VideoWriter(fname, fourcc=cv2.cv.CV_FOURCC('m', 'p', '4', 'v'), fps=int(1./Ts_microscope), frameSize=tuple(image_size[::-1]))
        for fr in mov:
            vw.write(cv2.cvtColor(fr,cv2.cv.CV_GRAY2RGB))
        vw.release()
    elif fmt=='tiff' or fmt=='tif':
        try:
            #mov = np.rollaxis(mov,0,3) #change time axis to last
            timsave(fname, mov)
        except:
            raise Exception('No working module for tiff saving.')

def save_data(fname, cells,neuropil,stim,t):
    data = {}
    data['cells'] = [cell.__dict__ for cell in cells]
    data['neuropil'] = neuropil.__dict__
    data['stim'] = stim
    pnames = ['Ds','image_size','field_size','duration','Ts_world','Ts_microscope_pixels', 'Ts_microscope', 'soma_radius', 'soma_circularity_noise_world', 'soma_circularity_noise', 'soma_density_field', 'soma_density', 'ca_rest', 'neuropil_density', 'imaging_background', 'imaging_noise', 'tau_gcamp_rise', 'tau_gcamp_decay', 'gcamp_kd', 'gcamp_nh', 'tau_ca_decay', 'ca_per_ap', 'stim_onset', 'stim_f', 'stim_dur', 'stim_gap', 'stim_n', 'cell_timing_offset', 'cell_magnitude', 'cell_baseline', 'cell_expression', 'cell_fluo_baseline', 'neuropil_mag', 'neuropil_baseline', 'incell_ca_dist_noise', 'npil_ca_dist_noise']
    data['params'] = {i:eval(i) for i in pnames}
    data['time'] = t

    np.save(fname, np.array([data]))
    savemat(fname, data)

def generate_movie():
    t = np.arange(0., duration, Ts_world)
    seq = imaging_background * np.ones((len(t), image_size[Y], image_size[X]))
    cells = generate_cells()
    neuropil = generate_neuropil()
    
    for cell in cells:
        stim = generate_stim(t, cell.offset)
        ca = generate_ca(t, stim, cell.mag, cell.baseline)
        fluo = generate_fluo(t, ca, cell.expression, cell.fluo_baseline)
        cell.ca = ca
        cell.fluo = fluo

    stim = generate_stim(t, 0.)
    ca = generate_ca(t, stim, neuropil_mag, neuropil_baseline)
    fluo = generate_fluo(t, ca)
    neuropil.fluo = fluo

    seq = construct(seq,cells,neuropil)

    seq = normalize(seq)
    samp_int = np.rint(Ts_microscope / Ts_world)
    mov = seq[0:len(seq):samp_int, :, :]
    noise = rand.normal(*imaging_noise, size=mov.shape)
    mov = mov + noise
    mov = np.rint(normalize(mov)*255.).astype(np.uint8)
    return mov,cells,neuropil,stim,t

if __name__ == '__main__':
    n = 1
    fname_glob = 'mov'
    os.mkdir(fname_glob)
    for i in xrange(n):
        fname = fname_glob + '_' + str(i)
        fname = os.path.join(fname_glob, fname)
        mov,cells,neuropil,stim,t = generate_movie()
        save_mov(mov, fname, fmt='tif')
        save_data(fname, cells,neuropil,stim,t)
