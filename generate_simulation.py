from simulations import Simulation
# this is for manually generating single simulations. for batches, use "generate_simulation*s*.py"

dest = './output'
name = 'high_npil_22'

sim = Simulation(name)
sim.imaging_noise = [0.,0.18] #0.1, 0.4 [making normal=0.18]
sim.neuropil_mag = 2.2 #0.05, 2.2 [normal=1.0]
sim.generate_movie()
sim.save_mov(fmt='tif',dest=dest)
sim.save_data(fmt='npy', dest=dest)
