import sys
import os
pjoin = os.path.join
import re

batch = 8
sim_dir = '/jukebox/wang/deverett/simulations/batch_%i'%(batch)
out_dir = '/jukebox/wang/abadura/FOR_PAPER_GRAN_CELL/simulations/AFTER_CLUSTER_AND_POSTPROCESSED/batch_%i_ANALYZED'%(batch)

def get_outpaths():
    return [pjoin(out_dir,subdir,simdir,simdir[:2]) for subdir in [dd for dd in os.listdir(out_dir) if os.path.isdir(pjoin(out_dir,dd))] for simdir in os.listdir(pjoin(out_dir,subdir)) if re.match('^[0-9]{2}_[0-9]{3}$',simdir)]

for p in get_outpaths():
    os.system('submit convert_ag.m %s'%p)
