#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.feature_selection import mutual_info_classif

# Load the dataset containing gene ontology features
# The file "geneOntology.csv" is expected to be tab-separated
feature_df = pd.read_csv("1_Reults.20887.interaction_terms_go_family_topfam.tsv", sep="\t")

# Extract feature matrix (X) and target variable (y)
# X: All columns except the first and last two (assuming they are metadata)
# y: The "protein_family" column, which serves as the target labels
X = feature_df.iloc[:, 1:-2]
y = list(feature_df["protein_family"])

# Load a list of features to drop from an external file ("drop.txt")
drop = []
with open('drop.txt', 'r') as f:
    for d in f:
        drop.append(d.strip())  # Strip whitespace and newline characters

# Compute mutual information between features and target variable
res = mutual_info_classif(X, y)

# Store features with mutual information scores above 0.2
dat = []
for i in res:
    if float(i) > 0.2:
        dat.append(round(i, 3))

# Retrieve feature names
feat = list(X.columns)

# Print each feature along with its corresponding mutual information score
for i in range(len(res)):
    print(f"{feat[i]},{res[i]}")

