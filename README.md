## TODO
rgbd_demo need to be changed to be consistent against rgb demo.

## GraphSeg
GrasphSeg provides state-of-the-art 3D segmentation from multi-view images, without depth information. This repository is the official implementation of the paper [GraphSeg: Segmented 3D Representations via Graph Edge Addition and Contraction](https://arxiv.org/abs/2504.03129). It contains a combination of real world experiments we ran, instruction to run GraphSeg on some standard dataset, along with the implementation of method itself. 

Please cite our work via the bibtex:
```bash
@article{tang2025graphsegsegmented3drepresentations,
   title={GraphSeg: Segmented 3D Representations via Graph Edge Addition and Contraction},
   author={Haozhan Tang and Tianyi Zhang and Oliver Kroemer and Matthew Johnson-Roberson and Weiming Zhi},
   journal={arXiv preprint arXiv:2504.03129},
   year={2025}
}
```

![Method Overview](/figures/gseg_overview.jpg)


## Installation

It's recommended to use a package manager like [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) to create a package environment to install and manage required dependencies, and the following installation guide assumes a conda base environment is already initiated. Our code is tested on [Ubuntu 22.04.5 LTS (Jammy Jellyfish)](https://releases.ubuntu.com/jammy/).

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


## Attribution

This repository is implemented based on the following dependencies.

[Dust3r](https://github.com/naver/dust3r) under [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/) 

[RoMa](https://github.com/Parskatt/RoMa.git) under [MIT License](https://github.com/Parskatt/RoMa/blob/main/LICENSE) and its dependency [DINOv2](https://github.com/facebookresearch/dinov2.git) under [Apache 2 License](https://github.com/facebookresearch/dinov2/blob/main/LICENSE)

[Language Segment-Anything](https://github.com/luca-medeiros/lang-segment-anything.git) under [Apache 2 License](https://github.com/luca-medeiros/lang-segment-anything/blob/main/LICENSE)


## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en)