# EnergyModels

## Description
EnergyModels is an extension to the [AIMM simulator](https://github.com/keithbriggs/AIMM-simulator), an open-source discrete event system level simulator for 5G new radio created by [Keith Briggs](https://keithbriggs.info/). The EnergyModels package additionally provides energy efficiency modelling and rapid prototyping of scenarios with hundreds of users and tens of base stations in a standalone macro network.

## Table of Contents
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)

## Dependencies
This module requires:

1. Python 3.8 or higher https://python.org.
2. NumPy https://numpy.org/.
3. Simpy https://pypi.org/project/simpy/.
4. Matplotlib https://matplotlib.org.
5. Pandas https://pandas.pydata.org.
6. AIMM simulator https://github.com/keithbriggs/AIMM-simulator.
7. Hexalattice package https://pypi.org/project/hexalattice/.


## Installation
1. In a virtual environment with Python 3.8 or greater: 
```
pip install numpy simpy matplotlib pandas AIMM-simulator hexalattice
```
2. Download the EnergyModels repo as a `.zip` and extract to a folder of your choice.
3. Open a terminal and navigate to the above folder.

## Usage
To run the EnergyModels repository, 

1. Change directory to the `/EnergyModels/KISS` folder.
2. Run the following: 
```
python run_kiss.py -c [path to configuration file]
``` 
3. Find the generated `.tsv` files in the `data/output/[name-of-experiment]/` folder for later analysis.

Examples of configuration files can be found in the `/data/input` sub-folders. For instance, running 
```
python run_kiss.py -c data/input/outer_cells_constant/outer_cells_constant.json
``` 
will execute a scenario with 12 users, while the inner cells reduce their transmit power in increments of 3dB. 

NOTE: Temporary json files will be created in the `/KISS` subfolder. These will be automatically removed when the simulation is complete OR if exited early by pressing <kbd>^ Ctrl</kbd>+<kbd>C</kbd>.

## Contact
For any inquiries or questions, please contact k.sthankiya@qmul.ac.uk.


