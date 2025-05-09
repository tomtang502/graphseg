# conda create -n graphseg python=3.10 cmake=3.14.0 -y
# conda activate graphseg
conda install cuda -c nvidia/label/cuda-12.4.1 -y # an alternative version would be 11.8, ignore this if you want to use global cudatoolkit
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 # change this if using cuda 11.8 referring to https://pytorch.org/get-started/previous-versions/
pip install numpy==1.26.4 # downgrade numpy, it is backward compatible

# install langsam and sam2
cp install_requirements/langsam_req.txt third_party/langsam/requirements.txt
cp install_requirements/langsam_modified.toml third_party/langsam/pyproject.toml
cd third_party/langsam && pip install -e . 

# install dust3r dependencies for end-to-end segmentation if camera does not provide depth
cd .. && cd dust3r && pip install -r requirements.txt

# install roma
cd ../roma && pip install -e . # there is a warning but should be fine to ignore it

cd ../.. && pip install pynvml==12.0.0
pip install plotly==6.0.0
pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu12==25.2.2" "cuml-cu12==25.2.1"

# download checkpoints
cd checkpoints
chmod +x download_ckpts.sh
./download_ckpts.sh