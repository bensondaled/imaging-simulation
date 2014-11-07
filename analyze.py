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

sim_dir = '/jukebox/wang/deverett/simulations/batch_3'
out_dir = '/jukebox/wang/abadura/FOR_PAPER_GRAN_CELL/simulations/AFTER_CLUSTER_AND_POSTPROCESSED/batch_3_ANALYZED'
temp_dir = '/jukebox/wang/deverett/tmp'



def cat(a,ty):
    if type(a[0]) in [np.ndarray, list]:
        return np.concatenate(a).astype(ty)
    else:
        return np.array(a).astype(ty)

def overlap(mask1,mask2,percent_of=False):
    if not np.sum(mask1*mask2):
        return 0.
    ol = float(len(np.argwhere(mask1*mask2==1)))
    if percent_of==1:
        ol /= np.sum(mask1)
    elif percent_of==2:
        ol /= np.sum(mask2)
    return ol

def distance(mask1, mask2):
    m1c = np.mean(np.argwhere(mask1), axis=0)
    m2c = np.mean(np.argwhere(mask2), axis=0)
    return np.sqrt(np.sum((m1c-m2c)**2))

def corr(sig1,sig2,norm=True):
    if len(sig1) / len(sig2) == 2:
        if len(sig1)%len(sig2):
            sig1 = sig1[1::2]
        else:
            sig1 = sig1[::2]
    elif len(sig2) / len(sig1) == 2:
        if len(sig2)%len(sig1):
            sig2 = sig2[1::2]
        else:
            sig2 = sig2[::2]
    if norm:
        sig1 = (sig1-np.mean(sig1))/np.std(sig1)
        sig2 = (sig2-np.mean(sig2))/np.std(sig2)
    return np.correlate(sig1,sig2,'full')

def variance(sig1,sig2,norm=True):
    if len(sig1) / len(sig2) == 2:
        if len(sig1)%len(sig2):
            sig1 = sig1[1::2]
        else:
            sig1 = sig1[::2]
    elif len(sig2) / len(sig1) == 2:
        if len(sig2)%len(sig1):
            sig2 = sig2[1::2]
        else:
            sig2 = sig2[::2]
    if norm:
        sig1 = (sig1-np.mean(sig1))/np.std(sig1)
        sig2 = (sig2-np.mean(sig2))/np.std(sig2)
    return ((sig1-sig2)**2)/len(sig1)

class Result(object):
    OUT = 0 # idx, not object
    IN = 1 # idx, not object
    PERC_OUT = 2
    PERC_IN = 3
    DIST = 4
    def __init__(self, in_path, out_path):
        self.in_path = in_path
        self.out_path_dir = out_path
        self.out_path = pjoin(self.out_path_dir,'resultsFactorizationWhole.mat')
        
        self.out = loadmat(self.out_path)
        self.inn = np.load(self.in_path)

        self.inn_params = np.atleast_1d(self.inn['params'])[0]
        self.in_cells = self.inn['cells']
        self.in_masks = np.array([c['mask_im'] for c in self.in_cells])
        self.in_masks_infov = np.array([c['mask_im'] for c in self.in_cells if c['was_in_fov']])
        self.out_masks = self.out['A'].toarray()
        n_outmasks = self.out_masks.shape[-1]
        self.out_masks = np.reshape(self.out_masks, np.append(self.in_masks.shape[1:],n_outmasks), order='F')
        self.out_masks = np.rollaxis(self.out_masks,2,0)
        self.out_masks[self.out_masks>0] = 1.0

        self.matches_all = self.match_cells()
        self.matches = np.array([m for m in self.matches_all if m[self.PERC_IN]>0])
        
        self.in_cell_neighbors=np.array([self.n_neighbors(c) for c in self.in_cells] )
        self.in_cell_match_quality = [self.match_quality(idx) for idx,c in enumerate(self.in_cells)]
        self.out_cell_max_ca = np.max(self.out['C'],axis=1)
        self.out_max_npil = np.max(self.out['f'])
        self.out_psn = self.out['P'][0][0][0].reshape(self.in_masks.shape[1:])

        self.n_in_cells_matched = len(np.unique(self.matches[:,self.IN])) 
        self.percent_in_cells_matched = float(self.n_in_cells_matched)/float(len(self.in_masks_infov))
        self.variances = [variance(self.out['C'][m[self.OUT]], self.in_cells[m[self.IN]]['ca']) for m in self.matches]
        
    def n_neighbors(self, c):
        th = 9
        n = 0
        for cp in self.in_cells:
            if distance(c['mask_im'], cp['mask_im']) < th:
                n+=1
        return n
    
    def match_quality(self, cidx):
        maxdist = np.max(self.matches[:,self.DIST])
        matches = [m for m in self.matches if m[self.IN]==cidx]
        match_quals = [m[self.PERC_OUT]*m[self.PERC_IN]*(1-(m[self.DIST]/maxdist)) for m in matches]
        if not match_quals:
            return 0.0
        else:
            return match_quals[np.argmax(match_quals)]

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
            dist = distance(om, self.in_masks[in_match_idx])
            matches.append([oi, in_match_idx, perc_out, perc_in, dist])
        return np.array(matches)

    def display_cells(self):
        pl.imshow(np.sum(self.in_masks, axis=0))
        p = True
        while True:
            p = pl.ginput(1,timeout=0)[0]
            p = np.array(p).astype(int)
            if not np.any(p):
                break
            for cidx,c in enumerate(self.inn['cells']):
                mask = c['mask_im']
                if mask[p[1],p[0]]:
                    print
                    print "Cell: %i"%cidx
                    print "Offset: %0.3f"%c['offset']
                    print "Expression: %0.3f"%c['expression']
                    print "Cell Magnitude: %0.3f"%c['mag']
                    print "Baseline: %0.3f"%c['baseline']
                    print "Center: " + str(c['center_im_'])
                    print "Neighbors: %i" % self.in_cell_neighbors[cidx]
                    print "Match Quality: %0.3f" % self.in_cell_match_quality[cidx]
                    print "Found: " + str(cidx in self.matches[:,self.IN])

def parse_results(taskn):
    sims_dirs = [pjoin(sim_dir,d) for d in os.listdir(sim_dir) if os.path.isdir(pjoin(sim_dir,d))]
    in_paths = [pjoin(sd,f) for sd in sims_dirs for f in  os.listdir(sd) if '.npz' in f]
    out_paths = get_outpaths()
    out_nums = [op.split('/')[-3] for op in out_paths]
    inout_paths = []
    for ip in sorted(in_paths):
        num = os.path.splitext(os.path.split(ip)[-1])[0]
        if num in out_nums:
            inout_paths.append([ip, out_paths[out_nums.index(num)]])

    if taskn>=len(inout_paths):
        return
    inp,outp = inout_paths[taskn]

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    outname = pjoin('/jukebox/wang/deverett/tmp','temp_%i'%taskn)

    res = Result(inp, outp)
    dic = {}
    dic['input'] = inp
    dic['output'] = outp
    dic['pct_matched'] = res.percent_in_cells_matched
    dic['avg_variance'] = np.mean(res.variances)
    dic['n_cells'] = res.inn_params['n_cells_in_fov']
    dic['noise_mag'] = res.inn_params['imaging_noise_mag']
    dic['noise'] = res.inn_params['imaging_noise']
    dic['timing_offset_std'] = res.inn_params['cell_timing_offset'][1]
    dic['neuropil_mag'] = res.inn_params['neuropil_mag']
    dic['in_cell_expressions'] = [c['expression'] for c in res.in_cells]
    dic['in_cell_matched'] = [cidx in res.matches[:,res.IN] for cidx in xrange(len(res.inn['cells']))]
    dic['in_cell_neighbors'] = res.in_cell_neighbors
    dic['in_cell_match_quality'] = res.in_cell_match_quality
    dic['out_cell_max_ca'] = res.out_cell_max_ca
    dic['out_max_npil'] = res.out_max_npil
    dic['avg_out_psn'] = np.mean(res.out_psn)
    dic['std_out_psn'] = np.std(res.out_psn)

    np.save(outname, np.array([dic]))

def merge_results():
    outname = pjoin(sim_dir,'comparison.npy')
    if os.path.exists(outname):
        os.remove(outname)
    out_all = []

    for f in sorted(os.listdir(temp_dir)):
        data = np.load(pjoin(temp_dir,f))
        for d in data:
            out_all.append(np.array([d]))
        os.remove(pjoin(temp_dir,f))
    for f in os.listdir(temp_dir):
        if f[0]=='.':
            os.remove(pjoin(temp_dir,f))
    os.rmdir(temp_dir)
    np.save(outname, out_all)

def get_outpaths():
    return [pjoin(out_dir,subdir,simdir,simdir[:2]) for subdir in [dd for dd in os.listdir(out_dir) if os.path.isdir(pjoin(out_dir,dd))] for simdir in os.listdir(pjoin(out_dir,subdir)) if re.match('^[0-9]{2}_[0-9]{3}$',simdir)]

def count_available():
    sims_dirs = [pjoin(sim_dir,d) for d in os.listdir(sim_dir) if os.path.isdir(pjoin(sim_dir,d))]
    sims_dirs = [pjoin(sim_dir,d) for d in os.listdir(sim_dir) if os.path.isdir(pjoin(sim_dir,d))]
    in_paths = [pjoin(sd,f) for sd in sims_dirs for f in  os.listdir(sd) if '.npz' in f]
    in_paths = [pjoin(sd,f) for sd in sims_dirs for f in  os.listdir(sd) if '.npz' in f]
    out_paths = get_outpaths()
    out_nums = [op.split('/')[-3] for op in out_paths]
    inout_paths = []
    for ip in sorted(in_paths):
        num = os.path.splitext(os.path.split(ip)[-1])[0]
        if num in out_nums:
            inout_paths.append([ip, out_paths[out_nums.index(num)]])

    print "Available input (simulation) files:\t\t\t%i"%(len(in_paths)) 
    print "Available output (deconvolution) files:\t\t\t%i"%(len(out_paths)) 
    print "Available pairs to analyze:\t\t\t\t%i"%(len(inout_paths)) 

def figure1():
    #neuropil
    data = np.load(pjoin(sim_dir,'comparison.npy'))
    data = np.array([d[0] for d in data])
    data = data[np.argsort([d['input'] for d in data])]
    
    MA = 0
    EX = 1
    NP = 2
    data_ma = [d['in_cell_matched'] for d in data]
    data_ex = [d['in_cell_expressions'] for d in data]
    data_np = [d['neuropil_mag'] for d in data]
    data_all = []
    for dm,de,dn in zip(data_ma,data_ex,data_np):
        assert len(dm) == len(de)
        for dmm,dee in zip(dm,de):
            data_all.append([int(dmm), float(dee), float(dn)])
    data_all = np.array(data_all)
    data_no = data_all[np.squeeze(np.argwhere(data_all[:,MA]==0)),:]
    data_yes = data_all[np.squeeze(np.argwhere(data_all[:,MA]==1)),:]

    #bin means
    np_vals = np.unique(data_all[:,NP])
    data_no = np.array([data_no[np.argwhere(data_no[:,NP]==nv)] for nv in np_vals])
    data_yes = np.array([data_yes[np.argwhere(data_yes[:,NP]==nv)] for nv in np_vals])
    data_no_mean = np.array([np.mean(i) for i in data_no])
    data_yes_mean = np.array([np.mean(i) for i in data_yes])
    data_no_std = np.array([np.std(i) for i in data_no])
    data_yes_std = np.array([np.std(i) for i in data_yes])
    data_no_sem = np.array([np.std(i)/np.sqrt(len(i)) for i in data_no])
    data_yes_sem = np.array([np.std(i)/np.sqrt(len(i)) for i in data_yes])
    mult = 180.
    s_no = np.array([len(nn)/float(len(nn)+len(yy)) for nn,yy in zip(data_no,data_yes)])
    s_yes = np.array([1-sn for sn in s_no])
    s_no,s_yes = mult*np.array([s_no,s_yes])

    pl.scatter(np_vals,data_no_mean,marker='o',label='Unfound Cells', color='r', s=s_no)
    pl.scatter(np_vals,data_yes_mean,marker='o',label='Found Cells', color='g',s=s_yes)
    pl.errorbar(np_vals,data_no_mean,yerr=data_no_sem,fmt=None,ecolor='k')
    pl.errorbar(np_vals,data_yes_mean, yerr=data_yes_sem,fmt=None,ecolor='k')
    pl.legend(loc='upper left',numpoints=1)
    
    #next
    data_np_full = cat(data_np, float)
    pct_matched = cat([d['pct_matched'] for d in data], float)
    
    pl.figure(2)
    idxs = np.argsort(data_np_full)
    pl.scatter(data_np_full[idxs],pct_matched[idxs])


    pl.figure(3)
    #make the curve of yes/no matched vs expression level
    data_ex = cat(data_ex, float)
    rang = np.linspace(data_ex.min(),data_ex.max(),50)
    data_ma = cat(data_ma, int)
    ma_binned = []
    for r1,r0 in zip(rang[1:],rang[:-1]):
        ma_binned.append(np.mean(data_ma[np.argwhere(np.logical_and(data_ex<r1, data_ex>=r0))]))
    ma_binned.append(np.mean(data_ma[np.argwhere(data_ex>rang[-1])]))
    pl.scatter(rang,ma_binned) 
    
    #from sims:
    CA = 0
    NP = 1
    data_out_ca = [d['out_cell_max_ca'] for d in data]
    data_out_npil = [d['out_max_npil'] for d in data]
    data_out = []
    for dc,dn in zip(data_out_ca, data_out_npil):
        for dcc in dc:
            data_out.append([float(dcc), float(dn)])
    data_out = np.array(data_out)
    np_vals = np.unique(data_out[:,NP])
    data_out = np.array([data_out[np.argwhere(data_out[:,NP]==nv)] for nv in np_vals])
    data_out_mean = np.array([np.mean(i) for i in data_out])
    data_out_sem = np.array([np.std(i)/np.sqrt(len(i)) for i in data_out])
    pl.figure(4)
    pl.plot(np_vals,data_out_mean,marker='o',markersize=7,linestyle='None', color='k')
    pl.errorbar(np_vals,data_out_mean,yerr=data_out_sem,fmt=None,ecolor='gray')

    #noise p.sn
    dd = cat([d['avg_out_psn'] for d in data],float)
    pl.figure(5)
    pl.plot(dd,'b*')
    
    return data_all

if __name__ == '__main__':
    option = sys.argv[1]
    if option == 'merge':
        merge_results()
    elif option == 'count':
        count_available()
    elif option == 'parse':
        taskn = int(sys.argv[2])-1
        parse_results(taskn)
    elif option == 'figure1':
        data_all=figure1()
    elif option == 'play':
        data = np.load(pjoin(sim_dir,'comparison.npy'))
        data = np.array([d[0] for d in data])
        data = data[np.argsort([d['input'] for d in data])]
        
        #batch4
        data_ex = cat([d['in_cell_expressions'] for d in data],float)
        idxs_filt = np.argwhere(data_ex>-1.)
        data_ex_filt = data_ex[idxs_filt]
        data_ma = cat([d['in_cell_matched'] for d in data],int)
        data_ma_filt = np.squeeze(data_ma[idxs_filt])
        data_noise = np.array([d['noise'][1] for d in data])
        data_noise = np.repeat(data_noise, float(len(data_ma))/len(data_noise))
        data_noise_filt = np.squeeze(data_noise[idxs_filt])
        idxs = np.argsort(data_noise_filt) 
        lims = [(i1,i2) for i1,i2 in zip(np.arange(0,len(idxs),100)[:-1],np.arange(0,len(idxs),100)[1:])]
        data_ma_bin = [np.mean(data_ma_filt[idxs][i1:i2]) for i1,i2 in lims]
        data_noise_bin = [np.mean(data_noise_filt[idxs][i1:i2]) for i1,i2 in lims]
        pl.scatter(data_noise_bin, data_ma_bin)
        
        sys.exit(0)
        xaxis = 'noise'
        xtype = float
        yaxis = 'pct_matched'
        ytype = float
        
        datax = cat([d[xaxis][1] for d in data], xtype)
        datay = cat([d[yaxis] for d in data], ytype)
        idxs = np.argsort(datax)
        pl.scatter(datax[idxs],datay[idxs])
