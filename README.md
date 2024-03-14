# Data release for Lattice investigations of the chimera baryon spectrum in the Sp(4) gauge theory

This repository contains the code used to prepare the plots and results included in [Lattice investigations of the chimera baryon spectrum in the Sp(4) gauge theory[2311.14663]](https://arxiv.org/abs/2311.14663).

## Requirements
- Python 3.11 (see `requirements.txt` for the required packages)

## Instructions: Running the analysis
- Install required dependencies (see below).
- Download the "raw data" from Zenodo and place them in `raw_data/`. `/generate/transform_h5.py` reads the log files and save them to one hdf5-file.
- Alternatively, download the hdf5-file, "data.h5", from Zenodo and place it in `tmp_data/`.
- Download the "metadata" from Zenodo and place in `metadata/`.
- Make
- Analysis results are saved to `CSVs/`.
- Figures and Tables can be found in
    - `fis/`
    - `tabs/`
- WARNING: `/analysis/analysis_AIC.py` requires some amount of computation resource (~15 hours with 40 CPUs). To skip this step, download the 'FIT_mass.csv' and place it in `CSVs/`.
