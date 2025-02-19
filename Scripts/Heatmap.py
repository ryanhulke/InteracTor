# Instalar pacotes, caso ainda não estejam instalados
# !pip install pandas seaborn matplotlib scipy numpy

# Importar pacotes necessários
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import zscore
import numpy as np

# 1. Ler os dados
data = pd.read_csv('1_Reults.20887.interaction_terms_go_family_topfam.tsv')

# 2. Preparar os dados
# Definir a última coluna como rótulos de linha e remover a coluna usada como índice
data.set_index(data.iloc[:, -1], inplace=True)  # Usar a última coluna como rótulo das linhas
data = data.iloc[:, 1:-1]  # Remover a última coluna (que agora é o índice)
#data = data.iloc[,1:]  # Remover a última coluna (que agora é o índice)

# 3. Calcular o z-score para padronizar os dados
data_zscore = data.apply(zscore, axis=0)

# 4. Normalizar a escala do z-score para -5 a 5
data_zscore_normalized = np.clip(data_zscore, -5, 5)

# 5. Definir o mapa de cores padrão usando a paleta 'coolwarm' do seaborn
cmap = sns.color_palette("coolwarm", as_cmap=True)

# 6. Definir as cores para as famílias
row_colors = {
    'Short-chain dehydrogenases/reductases (SDR) family': 'purple',
    'Cytochrome P450 family': 'lightgreen',
    'Enoyl-CoA hydratase/isomerase family': 'orange',
    'Bacterial solute-binding protein 2 family': 'darkblue',
    'Class-I aminoacyl-tRNA synthetase family': 'red',
    'Glycosyl hydrolase 5 (cellulase A) family': 'pink',
    'Peptidase S1 family': 'lightblue',
    'FPP/GGPP synthase family': 'yellow'
}

# Mapear as cores de acordo com as labels
row_labels = data.index
row_colors_list = [row_colors.get(label, 'grey') for label in row_labels]
row_colors_df = pd.DataFrame({'color': row_colors_list}, index=row_labels)

# 7. Criar o Heatmap com agrupamento e cores de linha
plt.figure(figsize=(16, 14))

# Criar o clustermap
g = sns.clustermap(data_zscore_normalized, cmap=cmap, 
                   xticklabels=True, yticklabels=False,  # Remover labels das linhas
                   col_cluster=True, row_cluster=True,
                   row_colors=row_colors_df['color'],  # Adicionar cores para linhas
                   cbar_kws={'label': 'Z-score', 'shrink': 0.5},
                   figsize=(16, 14))  # Ajuste do tamanho do gráfico

# Adicionar legenda do z-score à direita
cbar = g.cax.colorbar
cbar.set_label('Z-score')
cbar.ax.tick_params(labelsize=10)  # Ajustar tamanho das labels do z-score

# Adicionar legenda das cores das famílias à esquerda
handles = [plt.Line2D([0], [0], color=color, lw=4) for color in row_colors.values()]
labels = row_colors.keys()
plt.legend(handles, labels, bbox_to_anchor=(-0.2, 0.5), loc='center left', title='Protein Families')

plt.title('Heatmap of Protein Families (Z-score Normalized)')

# Salvar a figura como SVG
plt.savefig('heatmap_protein_families.svg', format='svg')

plt.show()

