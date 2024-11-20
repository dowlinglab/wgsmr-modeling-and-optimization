# wgsmr-modeling-and-optimization
Water-gas shift membrane reactor (WGS-MR) modeling and optimization in Pyomo.
This repository contains scripts for reproducing results in a manuscript submitted for publication. 

## citation
If you find this useful, consider citing us as: Agi DT, Abu El Hawa HW, Dowling AW, Hawley K. Techno-economic Analysis and Optimization of Water-Gas Shift Membrane Reactors for Blue Hydrogen Production. ChemRxiv. 2024; doi:10.26434/chemrxiv-2024-9cjzc-v2  This content is a preprint and has not been peer-reviewed.
## dependencies
The scripts in this repository build the WGS-MR model via [Pyomo](https://www.pyomo.org/) and solve it using [Ipopt](https://coin-or.github.io/Ipopt/) with linear solver [ma27](https://www.hsl.rl.ac.uk/), distributed as part of the [Institute for the Design of Advanced Energy Systems Process Systems Engineering Framework (IDAES PSE)](https://idaes.org/). See this knowledge article on how to install idaes: [IDAES Installation](https://idaes-pse.readthedocs.io/en/1.5.1/install/index.html).

## repository content
The content of this repository is highlighted below:

### model library
membrane_reactor_v2: model library containing functions to create and initialize the WGS-MR model and some helper functions for sensitivity analysis and visualization.

### folders
input: contains input data used to make plots

* blending_data.csv: levelized cost of hydrogen (LCOH) as a function of feedstock blend.
* cost_contribution.csv: percentage contribution of various factors to LCOH.
* seperation_data.csv: LCOH as a function of feed utilization efficiency. 
* tea_summary_data.json: LCOH and feed utilization efficiency data.
* tornado_data.json: LCOH data for 5% perturbations in seven factors. 

output: containing folder to save figures generated from running the codes in this repository

### jupyter notebooks
effect-of-temperature-on-conversion: plot of the effect of temperature on the performance of a hypothetical WGS-MR with thermally isolated reaction chamber and membrane (Fig. 6 in manuscript).

GHSV-temperature-tensitivity-contours: make contour plots for WGS-MR performance as a function of GHSV and reactor temperature  (Figs. 7 and S5).

pressure_and_sweep_sensitivities: a) plot CO conversion and H2 recovery as functions of feed pressure at various reactor temperatures (Fig. 5) b) plot CO conversion and H2 recovery as functions of sweep-to-feed ratio at various reactor temperatures (Fig. S4).

simulations_all_data: simulate WGS-MR, inspect simulation results, visualize retentate (or permeate) side concentration profiles (Figs. 4 & S3).

staged-reactor-optimization: optimize a staged-temperature WGS-MR (Figs. S6 & S7).

tea_plots: make technoeconomic analysis result plots using imported data (Figs. 8, 9, 10, & S8)

### other files
README: this file.

WGSMR_data_PCI-H2A: model input based on bench-scale WGS-MR laboratory data and industrial-relevant operating conditions (Table 3 in associated paper).
