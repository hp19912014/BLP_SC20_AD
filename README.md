This is the Artifact for SC20
Each folder has a seperate makefile which requires CUDA libraries and NVIDIA GPUs.
Tested with cuda/9.0.176
All the code was tested on V100 with NVLink2 and V100 with PCI-E.
Currently, our BLP version is stored under directories with name "spin*" 
############################################
The bandwidth folder cover the code for bandwidth test present in the paper.
The prototype folder contains our prototype implementations discussed in the paper.
############################################
Most Macros value have been pre-defined. Some might have hints to change to fit different environments (NVLink or PCIE) to achieve best performance
