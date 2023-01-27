import h5py

import os

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "\\validation\\smoothreward_s6_f5_d3_5000.h5"
print("1", dirname)

# with h5py.File("C:\\Users\\dangn\\Documents\\StandaloneAI\\validation\\smoothreward_s6_f5_d3_5000.h5", 'r') as f:
#     print("TEST1", "C:\\Users\\dangn\\Documents\\StandaloneAI\\validation\\smoothreward_s6_f5_d3_5000.h5")
#
#
# with h5py.File(dirname, 'r') as f:
#     print("TEST2")

print("TEST1", "C:\\Users\\dangn\\Documents\\StandaloneAI\\validation\\smoothreward_s6_f5_d3_5000.h5")
print("TEST2", dirname)
