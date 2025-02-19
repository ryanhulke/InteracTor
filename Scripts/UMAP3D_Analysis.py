# Import necessary packages
import pandas as pd
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# 1. Load the data
data = pd.read_csv('protein_family.csv')

# 2. Prepare the data
# Remove the first column if it is an unnecessary identifier
data_values = data.iloc[:, 1:-1]  # Assuming the last column is the label

# Standardize the data
data_values_standardized = (data_values - data_values.mean()) / data_values.std()

# Handle NaN values by replacing them with 0
data_values_standardized = data_values_standardized.fillna(0)

# 3. Execute UMAP with 3 components
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42, n_components=3)
umap_result = umap_model.fit_transform(data_values_standardized)

# 4. Create a DataFrame with UMAP results
umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2', 'UMAP3'])
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

# Map the colors to the DataFrame
umap_df['Color'] = umap_df['Label'].map(color_map)

# 6. Visualize with matplotlib in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the points in 3D
for label, color in color_map.items():
    subset = umap_df[umap_df['Label'] == label]
    ax.scatter(subset['UMAP1'], subset['UMAP2'], subset['UMAP3'], c=color, label=label, s=100, alpha=0.8)

ax.set_title('3D UMAP Projection of Protein Families')
ax.set_xlabel('UMAP Dimension 1')
ax.set_ylabel('UMAP Dimension 2')
ax.set_zlabel('UMAP Dimension 3')
ax.legend(loc='best', title='Family')
plt.show()
