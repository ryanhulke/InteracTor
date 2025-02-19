#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.utils import all_estimators
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import make_scorer
from collections import defaultdict



# Load dataset
df = pd.read_csv('../extracted_features/1_Reults.20887.interaction_terms_go_family_topfam.tsv',sep='\t')
X = df.iloc[:,1:-2]
y = list(df['protein_family'])

d = defaultdict(list)

# interatomic interactions datasets
#d['Interatomic interactions'] = [X.iloc[:,0:12],y]

# Chemical properties of amino acid side chains (CPAASC) 
#d['Chemical properties of amino acid side chains (CPAASC)'] = [X.iloc[:,13:20],y]

# Monopeptide 
#d['Monopeptide'] = [X.iloc[:,21:46],y]

# Dipeptide 
#d['Dipeptide'] = [X.iloc[:,21:722],y]

# Tripeptide 
#d['Tripeptide'] = [X.iloc[:,723:],y]

# Data sets
l = list()
fin = open('list2.txt','r')
for f in fin:
    l.append(f.strip())
fin.close()
l = [col.strip() for col in l]
colunas_validas = [col for col in l if col in X.columns]

## Top 100 
d['Top 100'] = [X[colunas_validas[0:100]],y]

# Top 200 
d['Top 200'] = [X[colunas_validas[0:200]],y]

# Top 300 
d['Top 300'] = [X[colunas_validas[0:300]],y]

# Top 400 
d['Top 400'] = [X[colunas_validas[0:400]],y]

# Top 500 
d['Top 500'] = [X[colunas_validas[0:]],y]

# All Features
d['All Features'] = [X[colunas_validas[0:]],y]

out = open('Train_test_results_no_II.csv','w')
out.write("Algorithm, Dataset, CV F1 macro, Accuracy, Precision macro, Recall macro, F1 macro, MCC macro\n")
out.close()

for x in d:
    # d[x][1]

    splits = []
    for i in range(5):
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(d[x][0], d[x][1],test_size=0.20,shuffle=True)
        
        # Get all classifiers from sklearn
        classifiers = [est for est in all_estimators(type_filter='classifier')]
        
        # Store results
        results = []
        dataset = f"{x}_{i}"
        for name, Classifier in classifiers:
            out = open('Train_test_results_no_II.csv','a')
            try:
                print (f"{name} - {dataset}")
                model = Classifier()
                # Perform cross-validation (5-fold)
                cv_res = cross_val_score(model, d[x][0], d[x][1], cv=5, scoring='f1_macro')
                cv_f1= cv_res.mean()
                
                # Train on full training set
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
        
                # Compute metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
                rec = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')
                mcc = matthews_corrcoef(y_test, y_pred)
        
                out.write(f"{name},{dataset},{cv_f1}, {acc}, {prec}, {rec}, {f1}, {mcc}\n")
            
            except (ValueError, TypeError, NotFittedError, Exception) as e:
                out.write(f"{name},{dataset},0, 0, 0, 0, 0, 0\n")
            out.close()
out.close()


# In[ ]:





# In[ ]:




