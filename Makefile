# Stop processes from thrashing
OMP_NUM_THREADS := 1
export OMP_NUM_THREADS

# Define variables
PYTHON := venv/bin/python3
VENV := venv/bin/activate

ENSEMBLES := 48x24x24x24b7.62 60x48x48x48b7.7 60x48x48x48b7.85 60x48x48x48b8.0 60x48x48x48b8.2
F_DATA := tmp_data/MASS_PS_F.npy CSVs/F_meson.csv
AS_DATA := tmp_data/MASS_PS_AS.npy CSVs/AS_meson.csv
CB_DATA := tmp_data/MASS_chimera.npy CSVs/CB_mass.csv
ALL_SPECTRUM := $(F_DATA) $(AS_DATA) $(CB_DATA)

FIG_NAMES := MCB_eolnc.pdf MCB_eo.pdf MCB_spin.pdf mh_h.pdf mh_l.pdf m_mps.pdf m_3D.pdf mh_mps2.pdf ss_mps2.pdf M_OC_AIC.pdf M_OV12_AIC.pdf M_OV32_AIC.pdf fix_AS_check.pdf fix_F_check.pdf plot_FIT_massless.pdf plot_quench_all_m0.pdf
FIGS := $(foreach fig, $(FIG_NAMES), figs/$(fig))
TABS := $(foreach tab, 1 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19, tabs/table_$(tab).tex)

# Phony targets
.PHONY: venv all clean analyze_data

# Default target
all: analyze_data

# Virtual environment setup

$(VENV): requirements.txt
	python3 -m venv venv
	$(PYTHON) -m pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt

venv: $(VENV)

# Target to analyze data
analyze_data: tmp_data/data.h5 $(ALL_SPECTRUM) CSVs/FIT_mass.csv $(FIGS) $(TABS)

# Transform output files to HDF5
tmp_data/data.h5:  $(foreach ensemble, $(ENSEMBLES), $(wildcard raw_data/$(ensemble)/chimera_out_*)) | $(VENV)
	$(PYTHON) generate/transform_h5.py;

# Analyze mesons in FUND. rep.
$(F_DATA): metadata/F_meson_meta.csv | $(VENV)
	$(PYTHON) analysis/analysis_F.py;

# Analyze mesons in ANTI. rep.
$(AS_DATA): metadata/AS_meson_meta.csv | $(VENV)
	$(PYTHON) analysis/analysis_AS.py;

# Analyze Chimera baryons
$(CB_DATA): metadata/CB_mass_meta.csv | $(VENV)
	$(PYTHON) analysis/analysis_CB.py;

# Analyze optimal fitting procedures
CSVs/FIT_mass.csv: $(ALL_SPECTRUM) | $(VENV)
	$(PYTHON) analysis/analysis_AIC.py;

# Analyze cross-check fitting procedures
CSVs/FIT_cross_fixAS.csv CSVs/FIT_cross_fixF.csv: $(ALL_SPECTRUM) | $(VENV)
	$(PYTHON) analysis/analysis_cross.py;

# Generate figures
$(FIGS): $(ALL_SPECTRUM) CSVs/FIT_cross_fixAS.csv CSVs/FIT_cross_fixF.csv | $(VENV)
	$(PYTHON) generate/plot_figs.py;


# Generate tables
$(TABS): CSVs/F_meson.csv CSVs/AS_meson.csv CSVs/CB_mass.csv CSVs/FIT_mass.csv CSVs/FIT_cross_fixAS.csv CSVs/FIT_cross_fixF.csv | $(VENV)
	$(PYTHON) generate/print_tabs.py;

# Clean target
clean:
	rm -rf __pycache__
	rm -rf Lib/__pycache__
	rm -rf venv
	rm -rf tabs
	rm -rf figs
	rm tmp_data/MASS_PS_F.npy tmp_data/MASS_PS_AS.npy tmp_data/MASS_chimera.npy
	rm CSVs/F_meson.csv CSVs/AS_meson.csv CSVs/CB_mass.csv
