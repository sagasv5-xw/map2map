import numpy as np
from nbodykit.source.catalog import BigFileCatalog
import os

# replace the path with your own path to the data
path = '/hildafs/home/xzhangn/xzhangn/sim_output/dmo-100MPC/15_0/dmo-64/set{}/output/PART_'


num_sims = 2 # number of different simulations
num_snapshots = 10 # number of snapshots per simulation

# list contain all scale factors
allTime = []
for i in range(num_sims):
    temp = []
    for j in range(num_snapshots):
        path2sim = path.format(i) + str(j).rjust(3, '0')
        print(path2sim)
        f = BigFileCatalog(path2sim, header = 'Header', dataset = '1/')
        a = f.attrs['Time']
        temp.append(a)
    allTime.append(temp)

print(len(allTime), len(allTime[0]))
assert len(allTime) == num_sims and len(allTime[0]) == num_snapshots , 'Wrong number of simulations or snapshots'


# replace the path with your own path to the style data    
style_savingpath = '/hildafs/home/xzhangn/xzhangn/sr_pipeline/3.5-training/15_0/map2map/scripts/style/set{}/'

for i in range(num_sims):
    style_folder = style_savingpath.format(i)
    if not os.path.exists(style_folder):
        os.makedirs(style_folder)
    for j in range(num_snapshots):
        path2style = style_folder + 'PART_' +  str(j).rjust(3, '0') + '.npy'
        a = np.array([allTime[i][j]])
        np.save(path2style, a)

