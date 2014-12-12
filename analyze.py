import pylab as pl
import time
import re
import csv
import numpy as np
import os
import sys
from scipy.sparse import spdiags
from scipy.io import savemat,loadmat
rand = np.random
pjoin = os.path.join

batch = 3
sim_dir = '/jukebox/wang/deverett/simulations/batch_%i'%(batch)
out_dir = '/jukebox/wang/abadura/FOR_PAPER_GRAN_CELL/simulations/AFTER_CLUSTER_AND_POSTPROCESSED/batch_%i_ANALYZED'%(batch)
temp_dir = '/jukebox/wang/deverett/tmp'

def cat(a,ty):
    if type(a[0]) in [np.ndarray, list]:
        return np.concatenate(a).astype(ty)
    else:
        return np.array(a).astype(ty)

def normalize(s):
    s = np.array(s)
    s = s-s.min()
    s = s/s.max()
    return s

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

def corr(sig1,sig2,norm=True,method=np.correlate):
    sig1 = np.squeeze(sig1)
    sig2 = np.squeeze(sig2)
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
    if method == np.correlate:
        return method(sig1,sig2,'full')
    elif method == np.corrcoef:
        return method(sig1,sig2)[0,1]

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
    CORR = 5
    def __init__(self, in_path, out_path):
        self.in_path = in_path
        self.out_path_dir = out_path
        self.out_path = pjoin(self.out_path_dir,'resultsFactorizationWhole.mat')
        
        self.out = loadmat(self.out_path)
        self.inn = np.load(self.in_path)

        self.inn_params = np.atleast_1d(self.inn['params'])[0]
        self.in_cells = self.inn['cells']
        self.in_masks = np.array([c['mask_im'] for c in self.in_cells])
        self.mask_shape = self.in_masks[0].shape
        self.in_masks_infov = np.array([c['mask_im'] for c in self.in_cells if c['was_in_fov']])
        self.out_masks = self.out['A'].toarray()
        n_outmasks = self.out_masks.shape[-1]
        self.out_masks = np.reshape(self.out_masks, np.append(self.in_masks.shape[1:],n_outmasks), order='F')
        self.out_masks = np.rollaxis(self.out_masks,2,0)
        self.out_energies = np.array([np.sum(m**2) for m in self.out_masks])
        self.out_masks[self.out_masks>0] = 1.0

        self.matches_all = np.array(self.match_cells())
        self.matches = np.array([m for m in self.matches_all if m[self.PERC_IN]>0])
        self.best_match_in_perc = [[m[self.PERC_IN] for m in self.matches_all if m[self.IN]==ci] for ci in xrange(len(self.in_cells))]
        self.best_match_in_perc = [np.max(m) if len(m) else 0 for m in self.best_match_in_perc]
        self.best_match_in = [np.array([(self.out_energies[m[self.OUT]],m[self.OUT]) for m in self.matches_all if m[self.IN]==ci]) for ci in xrange(len(self.in_cells))]
        self.best_match_in = [m[np.argmax(m[:,0])][1] if len(m) else -1 for m in self.best_match_in]
        
        self.in_cell_neighbors=np.array([self.n_neighbors(c) for c in self.in_cells] )
        self.in_cell_match_quality = [self.match_quality(idx) for idx,c in enumerate(self.in_cells)]
        self.out_cell_max_ca = np.max(self.out['C'],axis=1)
        self.deconv_f = self.out['f']
        self.out_max_npil = np.max(self.out['f'])
        self.out_psn = self.out['P'][0][0][0].reshape(self.in_masks.shape[1:])
        self.deconv_psn = self.out['P'][0][0][0].reshape(self.in_masks.shape[1:])
        
        s = len(np.squeeze(self.deconv_f))
        self.pg = np.squeeze(self.out['P']['g'][0][0])
        self.out_G = spdiags(np.ones(s)*self.pg, np.array([-1]), s, s).toarray() #removed negative from self.pg
        self.out_G[0,0] = 0
        self.deconv_matrix = self.out['C'].dot(self.out_G.T)

        self.n_in_cells_matched = len(np.unique(self.matches[:,self.IN])) 
        self.percent_in_cells_matched = float(self.n_in_cells_matched)/float(len(self.in_masks_infov))
        self.variances = [variance(self.out['C'][m[self.OUT]], self.in_cells[m[self.IN]]['ca']) for m in self.matches]

    def get_inmask(self):
        return np.sum(self.in_masks, axis=0)
    def get_outmask(self):
        return np.sum(self.out_masks, axis=0)
    def get_out_npil(self):
        npil = self.out['b'].toarray()#.reshape(self.in_masks.shape[1:])
        return np.array([np.min(npil), np.max(npil), np.mean(npil), np.std(npil)])


    def get_input_cell_summaries(self):
        cell_dtype = np.dtype([ ('input_path', 'a%i'%(len(self.in_path))),\
                                ('output_path', 'a%i'%(len(self.out_path_dir))),\
                                ('idx', np.int32),\
                                ('expression', np.float32),\
                                ('offset', np.float32),\
                                ('matched', np.bool),\
                                ('best_match_in_perc', np.float32),\
                                ('best_match', np.float32),\
                                ('ca', np.float64, len(self.inn['cells'][0]['ca'])),\
                                ('f', np.float64, len(self.inn['cells'][0]['fluo'])),\
                                ('sim_npil_mag', np.float32),\
                                ('sim_noise_g_std', np.float32),\
                                ('sim_noise_sh_mag', np.float32),\
                                ('sim_cell_timing_offset', np.float32),\
                                ('sim_n_cells', np.int32),\
                              ])
        cells = np.zeros(len(self.inn['cells']), dtype=cell_dtype)
        for ci,c in enumerate(self.inn['cells']):
            cells[ci]['input_path'] = self.in_path
            cells[ci]['output_path'] = self.out_path_dir
            cells[ci]['idx'] = ci
            cells[ci]['expression'] = float(c['expression'])
            cells[ci]['offset'] = float(c['offset'])
            cells[ci]['matched'] = ci in np.unique(self.matches[:,self.IN])
            cells[ci]['best_match_in_perc'] = self.best_match_in_perc[ci] #of all deconv cells mapped to this cell, what was the overlap of the one that overlapped most
            cells[ci]['best_match'] = self.best_match_in[ci] #of all deconv cells mapped to this cell, which one had highest energy?
            cells[ci]['ca'][:] = c['ca']
            cells[ci]['f'][:] = c['fluo']
            cells[ci]['sim_npil_mag'] = float(self.inn_params['neuropil_mag'])
            cells[ci]['sim_noise_g_std'] = float(self.inn_params['imaging_noise'][1])
            cells[ci]['sim_noise_sh_mag'] = float(self.inn_params['imaging_noise_mag'])
            cells[ci]['sim_cell_timing_offset'] = float(self.inn_params['cell_timing_offset'][1])
            cells[ci]['sim_n_cells'] = int(self.inn_params['n_cells_in_fov'])
        return cells
    def get_output_cell_summaries(self):
        cell_dtype = np.dtype([ ('input_path', 'a%i'%(len(self.in_path))),\
                                ('output_path', 'a%i'%(len(self.out_path_dir))),\
                                ('idx', np.int32),\
                                ('input_match_idx', np.int8),\
                                ('match_perc_in', np.float32),\
                                ('match_perc_out', np.float32),\
                                ('match_corrcoef', np.float32),\
                                ('C', np.float32, self.out['C'].shape[1]),\
                                ('deconv_f', np.float32, self.deconv_f.shape),\
                                ('deconv_mean_psn', np.float32),\
                                ('out_npil', np.float32, 4),\
                                ('sim_npil_mag', np.float32),\
                                ('sim_noise_g_std', np.float32),\
                                ('sim_noise_sh_mag', np.float32),\
                                ('sim_cell_timing_offset', np.float32),\
                                ('sim_n_cells', np.int32),\
                              ])
        cells = np.zeros(len(self.out_masks), dtype=cell_dtype)
        for ci in xrange(len(self.out_masks)):
            cells[ci]['input_path'] = self.in_path
            cells[ci]['output_path'] = self.out_path_dir
            cells[ci]['idx'] = ci
            cells[ci]['input_match_idx'] = self.matches_all[ci,self.IN]
            cells[ci]['match_perc_in'] = self.matches_all[ci,self.PERC_IN]
            cells[ci]['match_perc_out'] = self.matches_all[ci,self.PERC_OUT]
            cells[ci]['match_corrcoef'] = self.matches_all[ci,self.CORR]
            cells[ci]['C'][:] = self.out['C'][ci]
            cells[ci]['deconv_f'][:] = self.deconv_f
            cells[ci]['deconv_mean_psn'] = np.mean(self.deconv_psn)
            cells[ci]['out_npil'][:] = self.get_out_npil()
            cells[ci]['sim_npil_mag'] = float(self.inn_params['neuropil_mag'])
            cells[ci]['sim_noise_g_std'] = float(self.inn_params['imaging_noise'][1])
            cells[ci]['sim_noise_sh_mag'] = float(self.inn_params['imaging_noise_mag'])
            cells[ci]['sim_cell_timing_offset'] = float(self.inn_params['cell_timing_offset'][1])
            cells[ci]['sim_n_cells'] = int(self.inn_params['n_cells_in_fov'])
        return cells

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
            inca = self.in_cells[in_match_idx]['ca']
            outca = self.out['C'][oi]
            cc = corr(inca,outca,method=np.corrcoef)
            matches.append([oi, in_match_idx, perc_out, perc_in, dist, cc])
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
    np.savez(outname, inn=res.get_input_cell_summaries(), out=res.get_output_cell_summaries(), in_mask=res.get_inmask(), out_mask=res.get_outmask())

def merge_results():
    outname = pjoin(sim_dir,'comparison.npz')
    if os.path.exists(outname):
        os.remove(outname)

    inn = []
    out = []
    for f in sorted(os.listdir(temp_dir)):
        data = np.load(pjoin(temp_dir,f))
        inn.append(data['inn'])
        out.append(data['out'])
        os.remove(pjoin(temp_dir,f))
    inn = np.concatenate(np.array(inn))
    out = np.concatenate(np.array(out))

    for f in os.listdir(temp_dir):
        if f[0]=='.':
            os.remove(pjoin(temp_dir,f))
    os.rmdir(temp_dir)
    np.savez(outname, inn=inn, out=out)

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

if __name__ == '__main__':
    option = sys.argv[1]
    if option == 'merge':
        merge_results()
    elif option == 'count':
        count_available()
    elif option == 'parse':
        taskn = int(sys.argv[2])-1
        parse_results(taskn)
    elif option == 'figure':
        data = np.load(pjoin(sim_dir,'comparison.npz'))
        inn,out = data['inn'],data['out']
        inn = inn[np.argsort(inn['input_path'])]
        out = out[np.argsort(out['input_path'])]
        
        figidx = int(sys.argv[2])

        if figidx == 1:
            #expression vs neuropil magnitude, splitting matches and non-matches
            unique_npil_mags = np.unique(inn['sim_npil_mag'])
            yes_idxs = [np.argwhere(np.logical_and(inn[:]['matched']==True, inn[:]['sim_npil_mag']==nm)) for nm in unique_npil_mags]
            no_idxs = [np.argwhere(np.logical_and(inn[:]['matched']==False, inn[:]['sim_npil_mag']==nm)) for nm in unique_npil_mags]
            yes_expr,no_expr = zip(*[[inn['expression'][yi],inn['expression'][ni]] for yi,ni in zip(yes_idxs,no_idxs)])
            yes_ns,no_ns = map(len,yes_expr),map(len,no_expr)
            yes_means,no_means = map(np.mean, yes_expr),map(np.mean, no_expr)
            yes_stds,no_stds = map(np.std, yes_expr),map(np.std, no_expr)
            yes_sems, no_sems = zip(*[[ys/np.sqrt(yn), ns/np.sqrt(nn)] for ys,yn,ns,nn in zip(yes_stds,yes_ns,no_stds,no_ns)])
            mult = 180.
            yes_szs,no_szs = zip(*[[mult*yn/(yn+nn), mult*nn/(yn+nn)] for yn,nn in zip(yes_ns,no_ns)])

            pl.scatter(unique_npil_mags,no_means,marker='o',label='Unfound Cells', color='r', s=no_szs)
            pl.scatter(unique_npil_mags,yes_means,marker='o',label='Found Cells', color='g', s=yes_szs)
            #pl.errorbar(unique_npil_mags,no_means,yerr=no_sems,fmt=None,ecolor='k')
            #pl.errorbar(unique_npil_mags,yes_means,yerr=yes_sems,fmt=None,ecolor='k')
            #pl.legend(loc='upper left',numpoints=1)
            pl.xlabel('Neuropil Magnitude', fontsize=20)
            pl.ylabel('Cell Expression', fontsize=20)

        elif figidx == 2:
            #used when variable differs among cells
            #variable value along bottom, pct matched along y
            vname = 'expression'
            if 'batch_5' in sim_dir or 'batch_6' in sim_dir:
                vname = 'offset'
            try:
                thresh_expression, thresh_match = float(sys.argv[3]),float(sys.argv[4])
            except:
                print "figure x <expr_thresh> <match_thresh>"
                sys.exit(0)
            inn_filt = inn[np.argwhere(inn['expression']>thresh_expression)]
            inn_matched = inn_filt['best_match_in_perc'] > thresh_match

            idxs = np.squeeze(np.argsort(np.squeeze(inn_filt[vname])))
            bins = np.floor(np.linspace(0,len(idxs)+1,30))
            lims = [(i1,i2) for i1,i2 in zip(bins[:-1], bins[1:])]
            var_mags = [np.mean(inn_filt[vname][idxs][i1:i2]) for i1,i2 in lims]
            pct_matched = [np.mean(inn_matched[idxs][i1:i2]) for i1,i2 in lims]
            pl.scatter(var_mags, pct_matched)
            pl.xlabel(vname, fontsize=20)
            pl.ylabel('Fraction of Cells Matched', fontsize=20)
        
        elif figidx == 3:
            #used when variable does not differ among cells
            #variable value along bottom, pct matched or other var along y
            yvar = 'pct' #pct or psn or corrcoef or out_npil
            if 'batch_3' in sim_dir:
                vname = 'sim_npil_mag'
            if 'batch_4' in sim_dir:
                vname = 'sim_noise_g_std'
            elif 'batch_5' in sim_dir or 'batch_6' in sim_dir: 
                vname = 'sim_cell_timing_offset'
            elif 'batch_7' in sim_dir: 
                vname = 'sim_n_cells'
            try:
                thresh_expression, thresh_match = float(sys.argv[3]),float(sys.argv[4])
            except:
                print "figure x <expr_thresh> <match_thresh>"
                sys.exit(0)
            unique_var_mags = np.unique(inn[vname])
            inn_filt = inn[np.argwhere(inn['expression']>thresh_expression)]
            inn_matched = inn_filt['best_match_in_perc'] > thresh_match
            inn_matched_objs = inn_filt[inn_matched]
            if yvar == 'corrcoef' or yvar == 'psn':
                out_matches = []
                for imo in inn_matched_objs:
                    good = False
                    for oo in out[np.argwhere(out['input_path']==imo['input_path'])]:
                        if oo['input_path'] == imo['input_path'] and oo['idx'] == imo['best_match']:
                            out_matches.append(oo)
                            good = True
                out_matches = np.concatenate(out_matches)


            if yvar == 'pct':
                outcome = [np.mean(inn_matched[np.argwhere(inn_filt[vname]==vm)]) for vm in unique_var_mags]
            elif yvar == 'psn':
                outcome = [np.mean(out_matches['deconv_mean_psn'][np.argwhere(out_matches[vname]==vm)]) for vm in unique_var_mags]
            elif yvar == 'corrcoef':
                outcome = [np.mean(out_matches['match_corrcoef'][np.argwhere(out_matches[vname]==vm)]) for vm in unique_var_mags]
            elif yvar == 'out_npil':
                outcome_max = [np.mean(out['out_npil'][np.squeeze(np.argwhere(out[vname]==vm))][:,1]) for vm in unique_var_mags]
                outcome_min = [np.mean(out['out_npil'][np.squeeze(np.argwhere(out[vname]==vm))][:,0]) for vm in unique_var_mags]
                outcome_std = [np.mean(out['out_npil'][np.squeeze(np.argwhere(out[vname]==vm))][:,3]) for vm in unique_var_mags]
                outcome_mean = [np.mean(out['out_npil'][np.squeeze(np.argwhere(out[vname]==vm))][:,2]) for vm in unique_var_mags]
                #outcome = outcome_mean
                outcome = [m-mm for m,mm in zip(outcome_max,outcome_min)]
            pl.scatter(unique_var_mags, outcome)
            pl.xlabel(vname, fontsize=20)
            pl.ylabel(yvar, fontsize=20)

    elif option == 'example':
        #use batch 3
        data = np.load(pjoin(sim_dir,'comparison.npz'))
        inn,out = data['inn'],data['out']
        inn = inn[np.argsort(inn['input_path'])]
        out = out[np.argsort(out['input_path'])]

        n = '01_007'
        cidx = 21
        rec_idx = np.argwhere([n==ip.split('/')[-2] for ip in inn['input_path']])[0]
        rec = inn[rec_idx[0]]
        res = Result(rec['input_path'], rec['output_path'])
        mov = res.inn['movie'][0]['mov']

        #pl.figure(1)
        #zpro = np.std(mov, axis=0)
        #pl.imshow(zpro, cmap=pl.cm.Greys_r)

        #pl.figure(2)
        #pl.imshow(res.get_inmask(), cmap=pl.cm.Greys_r)
        #pl.figure(3)
        #pl.imshow(res.get_outmask(), cmap=pl.cm.Greys_r)

        pl.figure(4)
        outcellidx = np.argwhere(out['input_match_idx']==cidx)[0]
        outcell = out[outcellidx]
        incell = inn[cidx]
        t = res.inn['time']
        inca = incell['f']
        outca = outcell['C'][0]
        pl.plot(t, normalize(inca))
        pl.plot(t[1::2],normalize(outca)+1.1)
        pl.savefig('cell21_ca.eps')
        
        pl.figure(5)
        outcellidx = inn['best_match'][cidx]
        outstim = res.deconv_matrix[outcellidx]
        t = res.inn['time']
        instim = res.inn['cells'][cidx]['stim']
        pl.plot(t, normalize(instim))
        pl.plot(t[1::2],normalize(outstim)+1.1)
        pl.savefig('cell22_deconv.eps')
        
    elif option == 'play':
        data = np.load(pjoin(sim_dir,'comparison.npz'))
        inn,out = data['inn'],data['out']
        inn = inn[np.argsort(inn['input_path'])]
        out = out[np.argsort(out['input_path'])]
