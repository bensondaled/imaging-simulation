from simulations import Simulation
import json
import sys
import numpy as np
import os

task_id = sys.argv[1]
task_id = int(task_id)-1

dest = '/jukebox/wang/deverett/simulations/batch_7'

with open('generation_params.json','r') as f:
    params = json.loads(f.read())
vars = params['variables']
vals = params['values']

var_n = int(np.floor(float(task_id)/float(len(vals[0]))))
val_n = task_id % len(vals[0])

varr = vars[var_n]
val_list = vals[var_n]
val = val_list[val_n]
    
name = "%02d_%03d"%(var_n+1,val_n+1)
sim = Simulation(name)
setattr(sim,varr,val)
sim.generate_movie()
sim.save_mov(fmt='tif',dest=dest)
sim.save_data(fmt='npy', dest=dest)
sim.save_data(fmt='mat', dest=dest)
