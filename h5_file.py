from km3pipe.dataclasses import Table

import h5py
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os.path
import km3io as ki
from glob import glob
import awkward as ak
from collections import deque

#Folder = '/pbs/home/a/ameskar/Amine_Acoustics/root_files'

Folder = ''

#file = 'myfile.txt'

# Creating a file at specified location
#with open(os.path.join(Folder, file), 'w') as fp:
#    pass

Fnames = glob(os.path.join('*.root')) # Finding the root files

Hits = ['channel_id', 'dom_id', 't', 'tdc', 'pos_x', 'pos_y', 'pos_z', 'dir_x', 'dir_y', 'dir_z', 'tot', 'a', 'trig']

out = {k:deque([]) for k in ['channel_id', 'dom_id', 't', 'tdc', 'pos_x', 'pos_y', 'pos_z', 'dir_x', 'dir_y', 'dir_z', 'tot', 'a', 'trig', 'E']} #Creating the list (dque is better than the lsit because of the adding or the rmoving items in the end)

outfile='Data_for_ML.h5'

for f in Fnames:

    if not os.path.isfile(outfile): #if the file exist this will prevent running the code again

        print("Opening ", f) # open the file

        r= ki.OfflineReader(f)

        if len(r.events) > 0 : #ensuring non-empty files

            for i in range(len(r.hits)):
                for field in Hits:

                    out[f'{field}'].append(np.average(getattr(r.hits[i], field, None))) # append it fast with deque

            out['E'].extend([element for row in ak.flatten(r[r.mc_tracks.status == 100].mc_tracks.E, axis=0) for element in row] )

with h5py.File(outfile, 'w') as hf:

    for key, value in out.items():

        hf.create_dataset(key, data= value )

print('Written:', outfile)
