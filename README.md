# Exploring hidden flow structures from sparse data through deep-learning-strengthened proper orthogonal decomposition
This repository contains the source code for the research presented in the paper "Exploring hidden flow structures from sparse data through deep-learning-strengthened proper orthogonal decomposition", https://doi.org/10.1063/5.0138287.
The framework implements an improved POD framework for fluid dynamics problems by introducing Physics-Informed Neural Networks (PINNs) with parallel training.
## Overview
The main program, main.py, initiates multiple processes, each running an instance of sub_train.py, which corresponds to the training process of an individual PINN sub-network.
During the training, each sub-network writes out a Loss file and reconstructs the flow field, outputting it in a .h5 format at the end of each cosine annealing cycle.
The main.py monitors these outputs to calculate the Loss convergence criterion and performs Proper Orthogonal Decomposition (POD) for the modal convergence criterion.
Based on these criteria, main.py generates a signal file to guide whether sub_train.py should continue training.
## Key Components
`main.py`: The main script that orchestrates the parallel training of PINN sub-networks.  
`sub_train.py`: Handles the training process of each PINN sub-network.  
`utilities_ndm.py`: Defines the PINN network structure, inspired by the work available at https://github.com/maziarraissi/HFM.  
`PyPOD.py`: Contains utility functions used in the paper, such as post-processing, POD, and so on.  
Data Set: The data set used in the paper is available at this link http://gofile.me/5UAtB/NYzIc5sxJ.
## Getting Started
To use this framework, follow these steps:  
### 1. Clone the Repository  
```bash
git clone https://github.com/Panda000001/PINN-POD.git  
cd PINN-POD
```  
### 2. Install Dependencies  
Ensure you have Python 3.x and TensorFlow1.x installed.  
Install required Python packages (if any).  
### 3. Download the Data Set  
Download the data set from the provided link.  
Place the data set in the designated directory within the project.
### 4. Run the Main Program
```bash
python main.py
```
## Note
This part is also the preliminary work of our spatiotemporal parallel physics-informed neural networks, a more efficient framework of spatiotemporal parallel PINN, which is available at https://github.com/Shengfeng233/PINN-MPI.
## Citation
If you find this code useful in your research, please consider citing our paper:
```lua
@article{yan2023exploring,
  title={Exploring hidden flow structures from sparse data through deep-learning-strengthened proper orthogonal decomposition},
  author={Yan, Chang and Xu, Shengfeng and Sun, Zhenxu and Guo, Dilong and Ju, Shengjun and Huang, Renfang and Yang, Guowei},
  journal={Physics of Fluids},
  volume={35},
  number={3},
  year={2023},
  publisher={AIP Publishing}
}

@article{xu2023spatiotemporal,
  title={Spatiotemporal parallel physics-informed neural networks: A framework to solve inverse problems in fluid mechanics},
  author={Xu, Shengfeng and Yan, Chang and Zhang, Guangtao and Sun, Zhenxu and Huang, Renfang and Ju, Shengjun and Guo, Dilong and Yang, Guowei},
  journal={Physics of Fluids},
  volume={35},
  number={6},
  year={2023},
  publisher={AIP Publishing}
}
```
