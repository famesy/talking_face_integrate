import os
import h5py

# Create random data
# import numpy as np
# data_matrix = np.random.uniform(-1, 1, size=(1, 1, 12, 20))
#
# # Write data to HDF5
# with h5py.File('file.h5', 'w') as data_file:
#     data_file.create_dataset('group_name', data=data_matrix)
PATH = 'C:/Users/Ryuusei/PycharmProjects/XPRIZE/DANV-master/mfcc_test_h5/mfcc_python'
for filename in os.listdir(PATH):
    if filename.endswith('h5'):
        print(filename)

        with h5py.File(PATH + '/' + filename, 'r') as f:
            # List all groups
            print(f.keys())
            a_group_key = list(f.keys())[0]

            # Get the data
            data = list(f[a_group_key])
            print(data)
