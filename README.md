# InteracTor

## Description
InteracTor is a tool for molecular structure analysis and conversion, allowing the extraction of interactions and relevant features for biochemical studies.

## Installation and Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/Dias-Lab/InteracTor.git
   cd InteracTor
   python InteracTor.py examples examples.csv
   ```

2. Create the Conda environment for InteracTor:
   ```sh
   git clone https://github.com/Dias-Lab/InteracTor.git
   cd InteracTor
   conda create -n InteracTor
   conda activate InteracTor
   conda install conda-forge::openbabel
   ```

   Verify that the data directory contains both `.pdb` files and their respective `.mol2` files. If necessary, use OpenBabel to convert:
   ```sh
   obabel -ipdb data/protein.pdb -omol2 > data/protein.pdb.mol2
   ```

## Running InteracTor

Run InteracTor using an example dataset:
```sh
python InteracTor.py examples examples.csv
```
Alternatively, if you cloned the repository:
```sh
python InteracTor.py <path_to_examples> <path_to_examples_csv>
```

## File Structure
The following files are generated or used by InteracTor:

- `1_Reults.20887.interaction_terms_go_family_topfam.tsv` - Contains interaction terms and classifications.
- `data/` - Directory containing input data.
- `features.txt` - List of extracted features.
- `InteracTor.py` - Main script for running the program.
- `Scripts/` - Directory with auxiliary scripts.
- `1_Reults.interaction_terms.csv` - Identified interaction terms.
- `bonds.log` - Log of identified bonds.
- `examples/` - Directory with usage examples.
- `file_H-Bonds.pdb` - File containing hydrogen bond information.
- `l_result_result.txt` - Processing results.
- `tension.param` - Tension parameters used.
- `asa/` - Directory containing surface accessibility information.
- `pdbTomol2Example.txt` - Example conversion from PDB to MOL2.
- `examples.csv` - Example input file.
- `hydrophobicity.param` - Hydrophobicity parameters.
- `README.md` - Project documentation.
- `up_date.bash` - Script for system updates.

## Usage of Auxiliary Scripts
The following scripts provide additional functionalities:

- `Classification.py` - Performs classification based on interactions.
- `explainable_AInteracTor.ipynb` - Notebook for explainable analysis of InteracTor.
- `get_PDB_files_unique_chains.ipynb` - Extracts unique chains from PDB files.
- `Heatmap.py` - Generates heatmaps of interactions.
- `shap_ultils.py` - Tools for SHAP model interpretation.
- `UMAP_Analysis.py` - Performs dimensionality reduction analysis with UMAP.
- `drop.txt` - List of elements excluded from the analysis.
- `featureSelection.py` - Selects relevant features.
- `Graph.ipynb` - Notebook for graph visualization.
- `Heatmap.R` - Generates heatmaps using R.
- `UMAP3D_Analysis.py` - 3D UMAP analysis.
- `Wilcoxon.R` - Statistical tests using the Wilcoxon test.

This manual provides an overview of using InteracTor, its structure, and additional tools available for molecular interaction analysis.
