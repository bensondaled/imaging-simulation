from simulations import Simulation

n = 3
dest = 'output_folder'

for i in xrange(n):
    sim = Simulation('sim_%03d'%i)
    sim.generate_movie()
    sim.save_mov(fmt='tif',dest=dest)
    sim.save_data(fmt='npy', dest=dest)
    sim.save_data(fmt='mat', dest=dest)
