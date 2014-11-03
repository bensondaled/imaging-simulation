import pylab as pl
import time
import re
import csv
import numpy as np
import os
import sys
from scipy.io import savemat,loadmat
rand = np.random
pjoin = os.path.join

sim_dir = '/jukebox/wang/deverett/simulations/set_1'
out_dir = '/jukebox/wang/abadura/FOR_PAPER_GRAN_CELL/simulations/AFTER_CLUSTER_AND_POSTPROCESSED/'

def overlap(mask1,mask2,percent_of=False):
    if not np.sum(mask1*mask2):
        return 0.
    ol = float(len(np.argwhere(mask1==mask2)))
    if percent_of==1:
        ol /= np.sum(mask1)
    elif percent_of==2:
        ol /= np.sum(mask2)
    return ol

def corr(sig1,sig2,norm=True):
    if norm:
        sig1 = (sig1-np.mean(sig1))/np.std(sig1)
        sig2 = (sig2-np.mean(sig2))/np.std(sig2)
    return np.max(np.correlate(sig1,sig2,'full'))

class Result(object):
    OUT = 0 # idx, not object
    IN = 1 # idx, not object
    PERC_OUT = 2
    PERC_IN = 3
    def __init__(self, in_path, out_path):
        self.in_path = in_path
        self.out_path = out_path
        
        self.out = loadmat(self.out_path)
        self.inn = np.load(self.in_path)

        self.inn_params = np.atleast_1d(self.inn['params'])[0]
        self.in_cells = self.inn['cells']
        self.in_masks = [c['mask_im'] for c in self.in_cells]
        self.out_masks = np.rollaxis(self.out['dendmaskOrig'],2,0)

        self.matches = self.match_cells()

        # cell match stats
        self.n_in_cells_matched = len(np.unique(self.matches[:,self.IN])) #percentage of input cells 'found'
        self.percent_in_cells_matched = float(self.n_in_cells_matched)/float(len(self.in_cells))
        self.avg_percent_in_cell_overlap = np.mean(self.matches[:,self.PERC_IN]) #note: this considers multiple output cells mapped onto one input cell. may want to constrain

        # trace match stats
        #self.corrs_fluo = [corr(self.out_cells[m[self.OUT]]['fluo??'], self.in_cells[m[self.IN]]['fluo']) for m in self.matches] #Fake
        #self.corrs_ca = [corr(self.out_cells[m[self.OUT]]['ca??'], self.in_cells[m[self.IN]]['ca']) for m in self.matches] #Fake
        #self.corrs_stim = [corr(self.out_cells[m[self.OUT]]['stim??'], self.in_cells[m[self.IN]]['stim']) for m in self.matches] #Fake

    def match_cells(self):
        """
        Iterates through the deconv output cells, asking which input cell does it match.
        Maximizes overlap by percent of *output* cell.
        """
        matches = [] #each item is a list, [output cell idx, input cell idx]
        for oi,om in enumerate(self.out_masks):
            in_match_idx = np.argmax([overlap(om,im,percent_of=1) for im in self.in_masks])
            perc_out = overlap(om, self.in_masks[in_match_idx], percent_of=1)
            perc_in = overlap(om, self.in_masks[in_match_idx], percent_of=2)
            matches.append([oi, in_match_idx, perc_out, perc_in])
        return np.array(matches)

def parse_results(taskn):
    sims_dirs = [pjoin(sim_dir,d) for d in os.listdir(sim_dir) if os.path.isdir(pjoin(sim_dir,d))]
    in_paths = [pjoin(sd,f) for sd in sims_dirs for f in  os.listdir(sd) if '.npz' in f]
    out_paths = [pjoin(out_dir,subdir,simdir,simdir,'00','FilesSummary_.mat') for subdir in os.listdir(out_dir) for simdir in os.listdir(pjoin(out_dir,subdir)) if re.match('^[0-9]{2}_[0-9]{3}$',simdir)]
    out_nums = [op.split('/')[-3] for op in out_paths]
    inout_paths = []
    for ip in sorted(in_paths):
        num = os.path.splitext(os.path.split(ip)[-1])[0]
        if num in out_nums:
            inout_paths.append([ip, out_paths[out_nums.index(num)]])

    if taskn>=len(inout_paths):
        return
    inp,outp = inout_paths[taskn]

    if not os.path.exists('temp'):
        os.mkdir('temp')
    dw = csv.DictWriter(open('./temp/temp_%i.csv'%taskn,'w'),fieldnames=['input','output','pct_matched','n_cells','noise_mag','timing_offset_std'])
    dw.writeheader()

    res = Result(inp, outp)
    dic = {}
    dic['input'] = inp
    dic['output'] = outp
    dic['pct_matched'] = res.percent_in_cells_matched
    dic['n_cells'] = res.inn_params['n_cells_in_fov']
    dic['noise_mag'] = res.inn_params['imaging_noise_mag']
    dic['timing_offset_std'] = res.inn_params['cell_timing_offset'][1]
    dw.writerow(dic)

def merge_results():
    dw = csv.DictWriter(open('./results.csv','w'),fieldnames=['input','output','pct_matched','n_cells','noise_mag','timing_offset_std'])
    dw.writeheader()

    for f in sorted(os.listdir('temp')):
        dr = csv.DictReader(open(pjoin('temp',f),'r'))
        for rec in dr:
            dw.writerow(rec)
        os.remove(pjoin('temp',f))
    time.sleep(0.1)
    os.rmdir('temp')

def count_available():
    sims_dirs = [pjoin(sim_dir,d) for d in os.listdir(sim_dir) if os.path.isdir(pjoin(sim_dir,d))]
    in_paths = [pjoin(sd,f) for sd in sims_dirs for f in  os.listdir(sd) if '.npz' in f]
    out_paths = [pjoin(out_dir,subdir,simdir,simdir,'00','FilesSummary_.mat') for subdir in os.listdir(out_dir) for simdir in os.listdir(pjoin(out_dir,subdir)) if re.match('^[0-9]{2}_[0-9]{3}$',simdir)]
    out_nums = [op.split('/')[-3] for op in out_paths]
    inout_paths = []
    for ip in sorted(in_paths):
        num = os.path.splitext(os.path.split(ip)[-1])[0]
        if num in out_nums:
            inout_paths.append([ip, out_paths[out_nums.index(num)]])

    print "Available input (simulation) files:\t\t\t%i"%(len(in_paths)) 
    print "Available output (deconvolution) files:\t\t\t%i"%(len(out_paths)) 
    print "Available pairs to analyze:\t\t\t\t%i"%(len(inout_paths)) 


if __name__ == '__main__':
    option = sys.argv[1]
    if option == 'merge':
        merge_results()
    elif option == 'count':
        count_available()
    elif option == 'parse':
        taskn = int(sys.argv[2])-1
        parse_results(taskn)
