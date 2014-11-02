
import json
import numpy as np

n_per_variable = 40 #before replicates
replicates = 3
variables = ['soma_density_field','imaging_noise_mag','cell_timing_offset']
values = [np.linspace(10,200,n_per_variable).astype(int), np.linspace(0.01,4.0,n_per_variable), np.linspace(0.00001,0.800,n_per_variable)]
values = [list(np.repeat(i,replicates)) for i in values]
values[2] = [[0.050,i] for i in values[2]]

data = {}
data['variables'] = variables
data['values'] = values

f = open('generation_params.json','w')
f.write("%s"%json.dumps(data))
f.close()

print "Prepared params for %i simulations."%(len(values))
print "runjob should be run %i times."%(len(values)*n_per_variable*replicates)

