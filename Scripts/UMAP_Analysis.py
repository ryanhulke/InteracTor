# Install packages if not already installed
# !pip install umap-learn pandas matplotlib seaborn

# Import necessary packages
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
data = pd.read_csv('1_Reults.20887.interaction_terms_go_family_topfam.tsv')

# 2. Prepare the data
# Remove the first column if it is an unnecessary identifier
data_values = data.iloc[:, 1:-1]  # Assuming the last column is the label

# Standardize the data
data_values_standardized = (data_values - data_values.mean()) / data_values.std()

# Handle NaN values by replacing them with the mean of the column
data_values_standardized = data_values_standardized.fillna(0)

# 3. Execute UMAP
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
umap_result = umap_model.fit_transform(data_values_standardized)

# 4. Create a DataFrame with UMAP results
umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
umap_df['Label'] = data.iloc[:, -1]  # Assuming the last column is the label

# 5. Define specific colors for each label
color_map = {
    "Short-chain dehydrogenases/reductases (SDR) family": "purple",
    "Cytochrome P450 family": "lightgreen",
    "Enoyl-CoA hydratase/isomerase family": "orange",
    "Bacterial solute-binding protein 2 family": "darkblue",
    "Class-I aminoacyl-tRNA synthetase family": "red",
    "Glycosyl hydrolase 5 (cellulase A) family": "pink",
    "Peptidase S1 family": "lightblue",
    "FPP/GGPP synthase family": "yellow"
}

#color_map = {
#    "DNA/nucleotide binding [GO:0003677, GO:0000166]": "lightblue",
#    "metal ion binding [GO:0046872]": "orange",
#    "carbohydrate binding [GO:0030246]": "lightgreen",
#}
# Map the colors to the DataFrame
umap_df['Color'] = umap_df['Label'].map(color_map)

# 6. Visualize with matplotlib and seaborn
plt.figure(figsize=(12, 8))
sns.scatterplot(x='UMAP1', y='UMAP2', hue='Label', data=umap_df, palette=color_map, s=100, alpha=0.8)
plt.title('UMAP Projection of Protein Families')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(loc='best', title='Family')
plt.show()
