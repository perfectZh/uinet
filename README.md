

## Installation
git clone https://github.com/perfectZh/uinet.git
#### Clone the submodules.  
git submodule update --init    
#### Install dependencies
###Run the installation script to install all the dependencies.
```bash
bash install.sh conda_install_path uinet
```  


#### Test
Activate the conda environment and run the script pytracking/run_webcam.py to run using the webcam input.  
```bash
conda activate uinet
cd uinet
python run_tracker_demo.py 
```  
#### For VOT
##make sure run_vot_uinet.py is configured in your vot-toolkit path.


## Ref
### [ATOM](https://arxiv.org/pdf/1811.07628.pdf)

