imaging-simulation
==================

This goal of this project is to generate simulations of Ca imaging in granule cells. We then run deconvolution on the simulations, and compare the discovered results to the true data from the simulations.

File Index:

**Simulation Generation**
    * simulations.py: the module storing the Simulation object, which generates the simulations
    * generation_params.py: a script used just before generating simulations, to specify the number of simulations to create and their parameters
    * generation_params.json: the result of running generation_params.py, containing parameters for generation
    * generate_simulations.py: imports the necessary objects and parameters from simulations.py and generation_params.json, and creates the simulations
    * job_generate: a simple bash script to run generate_simulations.py on the cluster
    * tifffile.pyc: a necessary module for saving tiffs

**Results Analysis**
    * analyze.py: reads in simulation data & deconvolution results, and synthesizes them into simpler outputs. arguments:
        'count': will display the number of available comparisons for parsing
        'parse n': will run the comparison for the nth available file and save it in temp 
        'merge': will merge the results of multiple parses located in temp
    * job_analysis: a simple bash script to run analyze.py on the cluster
