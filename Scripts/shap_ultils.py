import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold
import xgboost as xgb 

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier, cv
#from shaphypetune import BoostSearch, BoostRFE, BoostRFA, BoostBoruta
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import label_binarize
from sklearn.multioutput import MultiOutputClassifier
from itertools import cycle

from sklearn import svm
import shap
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score
from scipy.stats import pearsonr
import networkx as nx


import seaborn as sn
# Run classifier with cross-validation and plot ROC curves
import scipy
import shap
import os



def get_shap(model, X, original_df_labels, target_label, model_name, link=shap.links.logit, multiclass_index=None):
    
    npermutations =  max(500, (2*len(original_df_labels)+1))
    if npermutations > 500:
        explainer = shap.explainers.Permutation(model, X, feature_names=original_df_labels, link=link, max_evals = npermutations)
    else:
        explainer = shap.Explainer(model, X, feature_names=original_df_labels, link=link)
    #calculate feature importances
    #shap_values = explainer(X)
    shap_values = explainer(X)
    #samples, features, outcomes
    print("SHAP shape", shap_values.values.shape)
    if multiclass_index != None:
        shap_values.values =  np.squeeze(shap_values.values[:,:,multiclass_index])
        print("SHAP shape after extracting class", multiclass_index , ":", shap_values.values.shape)

    print(len(shap_values.values.shape))
    if len(shap_values.values.shape) > 2:
        shap_values_true = np.squeeze(shap_values.values[:,:,1::2])
    else:
        shap_values_true = shap_values.values
        
    print("shape1", shap_values.shape)
    #convert to absolute values
    shap_values_abs = np.abs(shap_values_true)
    #extract total importance of features
    shap_values_abs_sum = shap_values_abs.sum(0)
    print("shape2", shap_values_abs_sum.shape)
    #return indexes of each feature sorted by their impact
    shap_values_abs_sum_argsort = shap_values_abs_sum.argsort()
    
    shap_v = pd.DataFrame(shap_values_true)
    shap_v.columns = original_df_labels
    df_v = pd.DataFrame(X, columns=original_df_labels)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in original_df_labels:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(original_df_labels),pd.Series(corr_list)],axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
    # Plot it
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    
    k2_out_prefix = 'abs_SHAP_feature_importance_'+target_label+'_'+model_name.replace(' ','_')
    k2.to_csv(k2_out_prefix+'.csv', index=False)
    
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    k2 = k2.iloc[-50:]
    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist, figsize=(8,12),legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    ax.set_title(target_label)
        
    out_prefix = 'SHAP_feature_importance_'+target_label+'_'+model_name.replace(' ','_')
    plot_path=out_prefix+'_bar.pdf'
    plt.savefig(plot_path, dpi=400, bbox_inches = "tight")
    
    fig = plt.figure(figsize =(8, 12))
    sorted_indexes = shap_values_abs_sum_argsort[::-1]
    print(shap_values.values.shape, type(shap_values.values))
    mydf = pd.DataFrame(shap_values_true, columns=original_df_labels)
    mydf.to_csv(out_prefix+'.csv', index=False)
    
    if len(shap_values.values.shape) > 2:
        shap.plots.beeswarm(shap_values[:,:,1], max_display=50, plot_size=[8,12], order=sorted_indexes, show=False, color_bar=False)
    else:
        shap.plots.beeswarm(shap_values[:,:], max_display=50, plot_size=[8,12], order=sorted_indexes, show=False, color_bar=False)
    
    clb = plt.colorbar(shrink=0.33)
    #clb.ax.set_title(ylabel='Feature value',fontsize=12)
    clb.set_label("Feature value")
    
    plt.title(target_label)
    plot_path=out_prefix+'_beeswarm.pdf'
    plt.savefig(plot_path, dpi=400, bbox_inches = "tight")
    
    plt.show()

    return explainer

def get_shap_interactions(model, X, original_df_labels, target_label, model_name, X_train = [], pert = 'tree_path_dependent', scaling_factor=8000, smoothing_factor=2, cor_threshold=0.80):
    if len(X_train) == 0:
        explainer = shap.Explainer(model, feature_names=original_df_labels)
    else:
        explainer = shap.Explainer(model, masker=X_train, feature_names=original_df_labels)
    #explainer = shap.Explainer(model, X, feature_names=original_df_labels, feature_perturbation=pert)
 
    shap_interaction = explainer.shap_interaction_values(X)
    print('max', shap_interaction.max(), 'min', shap_interaction.min(), 'shape', shap_interaction.shape)

    #shape is (6, 58, 42, 42)
    #why 6? because:
    #class 1 = 0, class 1 = 1
    #class 2 = 0, class 2 = 1
    #class 3 = 0, class 3 = 1
    #tested with: np.asarray(classifiers['Random Forest'].predict_proba(X)).T[1]
    #ci=0
    #for c in range(1,np.asarray(shap_interaction).shape[0],2):
    #    ci=ci+1
    interaction_df = pd.DataFrame(shap_interaction.mean(0),
                                  index=original_df_labels,
                                  columns=original_df_labels)
    print(np.asarray(shap_interaction).shape)
    #print(shap_interaction[2][0][0])
    #print(shap_interaction[3][0][0])

    ######get_ipython().run_line_magic('matplotlib', 'inline')
    # Create a mask
    mask = np.triu(np.ones_like(interaction_df, dtype=bool))
    #custom palette
    #cmap = sns.color_palette("rocket_r", as_cmap=True)
    cmap = sn.color_palette("Reds", 5)
    #cmap = sns.color_palette("WtRd", 5)
    cmap = sn.color_palette("RdBu_r", 5)
    #cmap = sns.color_palette("vlag", as_cmap=True)
    #cmap= sns.cubehelix_palette(start=5, rot=0, dark=0.25, light=1.5)
    my_center = 0.5*(interaction_df.to_numpy().mean(0).max()-interaction_df.to_numpy().mean(0).min())+interaction_df.to_numpy().mean(0).min()
    my_center=0.00
    ax = plt.axes()
    
    
    sn.heatmap(interaction_df, center = my_center,
                square=True, mask=mask, cmap=cmap)
    ax.set_title(target_label)
    #interaction_df.style.background_gradient(cmap='Blues')
    out_prefix = 'SHAP_feature_interaction_importance_'+target_label+'_'+model_name.replace(' ','_')
    plt.savefig(out_prefix+".pdf", bbox_inches = 'tight')
    plt.show()
    plt.clf()

    df = interaction_df.where(np.triu(np.ones(interaction_df.shape)).astype(bool))
    df = df.stack().reset_index()
    df.columns = ['source','target','weight']
    df['color']=df['weight']
    df.color[df.weight < 0] = "blue"
    df.color[df.weight > 0] = "red"
    df.color[df.weight == 0] = "black"
    # Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
    Q1=np.quantile(interaction_df.to_numpy(), .20)
    Q3=np.quantile(interaction_df.to_numpy(), .80)
    print(Q1,Q3)
    out_prefix='SHAP_edges_filtered_extreme_quartiles_'+target_label+'_'+model_name.replace(' ','_')
    edges_filtered=df.loc[ ((df['weight'] >= Q3) | (df['weight'] <= Q1)) & (df['source'] != df['target']) ]
    plt.figure(3,figsize=(12,12)) 
    edges_filtered.to_csv(out_prefix+'.csv', index=False)
    
    out_prefix='SHAP_edges_unfiltered_'+target_label+'_'+model_name.replace(' ','_')
    
    df.to_csv(out_prefix+'.csv', index=False)

    print("head",edges_filtered.head())
    #scaler = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(np.array(edges_filtered['weight']).reshape(-1, 1))
    #edges_filtered['weight'] = scaler.transform(np.array(edges_filtered['weight']).reshape(-1, 1))


    # Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
    threshold = cor_threshold
    #links_filtered=links.loc[ ((links['value'] > threshold) | (links['value'] < -threshold)) & (links['var1'] != links['var2']) ]

    # create a new graph from edge list
    Gx = nx.from_pandas_edgelist(edges_filtered, "source", "target", edge_attr=["weight"])
    

    # list to store edges to remove
    remove = []
    keep = []
    # loop through edges in Gx and find correlations which are below the threshold
    for var1, var2 in Gx.edges():
        corr = Gx[var1][var2]["weight"]
        # add to remove node list if abs(corr) < threshold
        if abs(corr) < threshold:
            remove.append((var1, var2))
        elif var1 == var2:
            remove.append((var1, var2))
        else:
            keep.append((var1, var2))

    # remove edges contained in the remove list
    Gx.remove_edges_from(remove)
    Gx.remove_nodes_from(list(nx.isolates(Gx)))

    print(str(len(remove)) + " edges removed, remaining: " + str(len(keep)))

    
    
   
    Gx.remove_nodes_from(list(nx.isolates(Gx)))
    print(edges_filtered.describe())
    # Build your graph
    #G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
    #G.edges(data=True)

    edge_colours = []
    edge_width = []
    for key, value in nx.get_edge_attributes(Gx, "weight").items():
        edge_colours.append(assign_colour(value))
        edge_width.append(assign_thickness(value, benchmark_thickness=50, scaling_factor=1))

    # assign node size depending on number of connections (degree)
    node_size = []
    for key, value in dict(Gx.degree).items():
        node_size.append(assign_node_size(value, scaling_factor=300))

    df = pd.DataFrame(index=Gx.nodes(), columns=Gx.nodes())
    for row, data in nx.shortest_path_length(Gx):
        for col, dist in data.items():
            df.loc[row,col] = dist

    df = df.fillna(df.max().max())
    df = df+(df.max().max())
    layout = nx.kamada_kawai_layout(Gx, dist=df.to_dict())


    fig, ax = plt.subplots()

    #plt.figure(figsize=(7,6))


    #sn.set(rc={"figure.figsize": (10, 10)})
    plt.figure(figsize=(7,6))

    nx.draw(
        Gx,
        #pos=nx.fruchterman_reingold_layout(Gx),
        #pos=nx.spectral_layout(Gx),
        #pos=nx.circular_layout(Gx),
        #pos=nx.nx_agraph.graphviz_layout(Gx, prog="neato"),
        #pos=nx.spring_layout(Gx),
        #pos=nx.nx_pydot.graphviz_layout(Gx),
        pos=layout,
        with_labels=True,
        node_size=node_size,
        node_color=assign_node_colors(Gx),#"#e1575c",
        edge_color=edge_colours,
        width=edge_width,
        #width=1.2,
        font_size=8,
        ax=ax
    )
    
    out_prefix='SHAP_interaction_network_'+target_label+'_'+model_name.replace(' ','_')
    
    plt.savefig(out_prefix+'.png',dpi=400, bbox_inches='tight')
 
    #ax.legend(*get_legend(np.array(node_size), num=4), title="Degree")
    
    #plt.colorbar(ec)
    #plt.axis('off')

    plt.show()
    plt.clf()

    return interaction_df


def assign_colour(correlation):
    if correlation > 0:
        return "#ffa09b"  # red
    else:
        #return "#9eccb7"  # green
        return "#0047ab"  # blue


def assign_thickness(correlation, benchmark_thickness=4, scaling_factor=1000):
    return benchmark_thickness * abs(correlation) ** scaling_factor


def assign_node_size(degree, scaling_factor=35):
    return degree * scaling_factor

# helper method for getting legend
def get_legend(sizes, num=10):
    fig, ax = plt.subplots()
    sc = ax.scatter([0]*len(sizes), [0]*len(sizes), s=sizes, color='blue')
    store = sc.legend_elements("sizes", num=num)
    fig.clf()
    return store

def assign_node_clustering(Gx):
    cluster_ids = np.unique(list(nx.clustering(Gx).values()))
    print(cluster_ids)
    n=len(cluster_ids) # number of colors you want to get
    cmap = plt.cm.get_cmap('rainbow', n) 
    my_map=dict(zip(cluster_ids, [cmap(i) for i in range(n)]))
    colors=[]
    for node, cluster in nx.clustering(Gx).items():
        colors.append(my_map[cluster])
    return colors

def assign_node_colors(Gx):
    node_cluster={}
    cluster_ids=[]
    for cluster_id, nodes in enumerate(nx.connected_components(Gx)):
        cluster_ids.append(cluster_id)
        for node in nodes:
            node_cluster[node]=cluster_id
    cmap = plt.cm.get_cmap('tab20', len(cluster_ids)) 
    my_map=dict(zip(cluster_ids, [cmap(i) for i in range(len(cluster_ids))]))    
    colors=[]
    for node in Gx.nodes():
        colors.append(my_map[node_cluster[node]])
        
    return colors       

if __name__ == "__main__":
    print("Usage demo.")
    print("from shap_ultils import get_shap, get_shap_interactions")
    