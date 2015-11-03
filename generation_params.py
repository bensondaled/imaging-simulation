import json
import numpy as np

n_per_variable = 40 #before replicates
replicates = 3

#batch1:
#variables = ['soma_density_field','imaging_noise_mag','cell_timing_offset']
#values = [np.linspace(10,200,n_per_variable).astype(int), np.linspace(0.01,4.0,n_per_variable), np.linspace(0.00001,0.800,n_per_variable)]
#values[2] = [[0.050,i] for i in values[2]] #comes after repeat

#batch2,3:
#variables = ['neuropil_mag']
#values = [np.linspace(0.01,2.2,n_per_variable)]

#batch4:
#variables = ['imaging_noise']
#values = [np.linspace(0.001,2.0,n_per_variable)]
#batch4: (more)
#values[0] = [[0.0,i] for i in values[0]] #comes after repeat

#batch5,6:
#variables = ['cell_timing_offset']
#values = [np.linspace(0.0005,0.950)]
#batch5 (more):
#values[0] = [[0.050,i] for i in values[0]] #comes after repeat

#batch7:
#variables = ['soma_density_field']
#values = [np.linspace(10,200,n_per_variable).astype(int)]

#batch8:
#variables = ['imaging_noise']
#values = [np.linspace(0.001,0.5,n_per_variable)]
#values = [list(np.repeat(i,replicates)) for i in values]
#batch8: (more)
#values[0] = [[0.0,i] for i in values[0]] #comes after repeat

#batch9:
variables = ['neuropil_mag']
values = [list(np.linspace(0.01,2.2,n_per_variable))]

data = {}
data['variables'] = variables
data['values'] = values

f = open('generation_params.json','w')
f.write("%s"%json.dumps(data))
f.close()

print "Prepared params for %i simulation set(s)."%(len(values))
print "job_generate should be run %i times."%(len(values)*n_per_variable*replicates)
print "EX: >> submit -tc 50 %i job_generate"%(len(values)*n_per_variable*replicates)
print "Do not forget to first specify the output location in generate_simulations.py"


