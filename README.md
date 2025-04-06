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


## Abstract:

Robots operating in unstructured environments often require accurate and consistent object-level representations. This typically requires segmenting individual objects from the robot’s surroundings. While recent large models such as Segment Anything (SAM) offer strong performance in 2D image segmentation, these advances do not translate directly to performance in the physical 3D world, where they often over-segment objects and fail to produce consistent mask correspondences across views. In this paper, we present GraphSeg, a framework for generating consistent 3D object segmentations from a sparse set of 2D images of the environment without any depth information. GraphSeg adds edges to graphs and constructs dual correspondence graphs: one from 2D pixel-level similarities and one from inferred 3D structure. We formulate segmentation as a problem of edge addition, then subsequent graph contraction, which merges multiple 2D masks into unified object-level segmentations. We can then leverage 3D foundation models to produce segmented 3D representations. GraphSeg achieves robust segmentation with significantly fewer images and greater accuracy than prior methods. We demonstrate state-of-the-art performance on tabletop scenes and show that GraphSeg enables improved performance on downstream robotic manipulation tasks.

![Method Abs](/figures/gs_abs.png)