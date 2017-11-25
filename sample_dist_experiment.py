import os


print("-------")
print(os.environ['PBS_TASKNUM'])
print(os.environ['PBS_NODENUM'])
print(os.environ['$PBS_NODENUM'])
print("-------")
