# Proactive Local-Minima-Free Robot Navigation: Blending Motion Prediction with Safe Control
This is the demo code of the MMP-MCBF control framework proposed in the paper "Proactive Local-Minima-Free Robot Navigation:
Blending Motion Prediction with Safe Control", accepted in IEEE RA-L on Feb 6, 2026. 

Authors: Yifan Xue*, Ze Zhang*, Knut Akesson, and Nadia Figueroa.

Link to the paper: https://arxiv.org/abs/2601.10233. 

## Quick Start
### MPC dependency
The MPC controller is implemented based on the OPEN engine (in Rust). To install the OPEN engine, please follow the instructions on the [website](https://alphaville.github.io/optimization-engine/).
### Install dependencies
It is recommended to create a virtual environment and install the dependencies using `pip`. Based on the code you want to run:
1. To be able to run the whole project (MCBF, MPC, and EBM), run:
    ```bash
    pip install -r requirements.txt
    ```
2. To only run the social navigation simulation (no EBM), run:
    ```bash
    pip install -r requirements_socialnav.txt
    ```
2. To only run the MCBF controller (no MPC), run:
    ```bash
    pip install -r requirements_mcbf.txt
    ```
### Run the code
To run the MPC part, 
```
python3 src/build_mpc_solver.py
```
A new folder `mpc_solver` will be created in the root directory. This folder contains the compiled solver and the corresponding shared library.

---

To run the social navigation simulation, run:
```bash
python3 src/main_social_nav.py
```
This will start the simulation with the MCBF controller with a predictive horizon of 100 (5s). To run MPC, set the `use_mpc` flag to `True`:
```bash
python3 src/main_social_nav.py --use_mpc True
```
To change the predictive horizon, set the `pred_len` flag (sampling time is 0.05s):
```bash
python3 src/main_social_nav.py --pred_len 50
```

---

To run the hospital navigation simulation, run:
```bash
python3 src/main.py
```
Note that this simulation runs the controller and multiple neural networks in parallel. It is recommended to generate the visualization offline, by changing `SAVE_VIDEO` in the script to `True`. This will save the video in the project root directory. 
