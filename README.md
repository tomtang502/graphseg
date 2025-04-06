## GraphSeg
This repository is the official implementation of the paper [GraphSeg: Segmented 3D Representations via Graph Edge Addition and Contraction](https://github.com/tomtang502/graphseg). It contains a combination of real world experiments we ran, instruction to run GraphSeg on some standard dataset, along with the implementation of method itself. 
![Method Overview](/figures/gseg_overview.jpg)

## Installation

It's recommended to use a package manager like [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) to create a package environment to install and manage required dependencies, and the following installation guide assume a conda base environment is already initiated. Our code is tested on [Ubuntu 22.04.5 LTS (Jammy Jellyfish)](https://releases.ubuntu.com/jammy/).

```bash
git clone --recursive https://github.com/tomtang502/graphseg.git graphseg
cd graphseg

# create conda env
conda create -n graphseg python=3.10 cmake=3.14.0 -y
conda activate graphseg

# install graphseg
sudo chmod +x installation.sh 
./installation.sh

```

## Download Checkpoints and Data
Detailed instruction, chekcpoint download script, etc. coming soon!


## Grasping Experiments:
GraphSeg is designed for robot planning and manipulation on a tabletop, so we also conducted grasping experiments using Unitree Z1 Arm.
<table>
  <tr>
    <td><img src="/figures/gs_abs.png" alt="Method Abs" width="300"></td>
    <td><img src="/figures/grasp.gif" alt="Grasp exp" width="500"></td>
  </tr>
</table>
