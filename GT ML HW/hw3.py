#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn import datasets
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from scipy.stats.stats import pearsonr
from sklearn import random_projection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder


# In[3]:


#KMEANS NBA SHOTS DATA


# In[4]:


def read_nba():
    df = pd.read_csv("final.csv")
    df.drop(['firstname','lastname'],axis=1,inplace=True)
    df['pot_asst'] = np.where(~np.isnan(df['passer']),1,0)
    df.dropna(subset=['shot_shot_clock','ndd'],inplace=True)
    
    return df


# In[29]:


def load_wine():
    #KMEANS WINE DATA
    wine = datasets.load_wine()
    
    wine_df = pd.DataFrame(wine.data)
    wine_df.columns = wine.feature_names
    
    wine_df['target'] = wine.target
    
    wine_df_feats = wine_df.drop('target',axis=1)
    scaler = MinMaxScaler()
    wine_sc = scaler.fit_transform(wine_df_feats)
    
    return wine_df, wine_sc


# In[13]:


def prep_nba(df):
    #X and y variable spaces for first classification task "Was the shot made?"
    X = df[['dribbles','distance_travelled','ndd','shot_dist','height','shot_shot_clock']]
    y = df['made']
    
    #scale to put features on the same scale
    scaler = MinMaxScaler()
    X_sc = scaler.fit_transform(X)
    
    return X, y, X_sc


# In[7]:


def k_means_loop(input_data, test_clusters):
    # k means determine k
    distortions = []
    K = range(1,test_clusters)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(input_data)
        kmeanModel.fit(input_data)
        distortions.append(sum(np.min(cdist(input_data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / input_data.shape[0])
    
    plt.plot(range(1,test_clusters),distortions)


# In[8]:


def final_kmeans(k,input_data,append_df):
    km = KMeans(n_clusters = k, random_state = 10)
    labels = km.fit_predict(input_data)
    append_df['cluster'] = labels
    
    return append_df


# In[24]:


def avg_feature(X):
    return X.groupby('cluster').mean().reset_index()


# In[9]:


def get_meaningful_clusters(input_data):

    #get most meaningful value per cluster
    cluster_avgs = input_data.groupby('cluster').agg(np.mean).reset_index()
    cluster_avgs = cluster_avgs.iloc[:,1:]
    lg_avgs = pd.DataFrame(input_data.mean()).transpose().iloc[:,:-1]

    mults = pd.DataFrame()
    for i in range(0,len(cluster_avgs)):
        mults = mults.append(abs((cluster_avgs.iloc[i] - lg_avgs) / cluster_avgs.iloc[i]))

    return mults.idxmax(axis=1)


# In[10]:


def get_cluster_accuracies(input_data,target_column):
    cluster_totals = input_data.groupby('cluster').agg({target_column: ['sum','count']}).reset_index()
    cluster_totals.columns = cluster_totals.columns.droplevel()
    
    cluster_totals['outcome'] = cluster_totals['sum'] / cluster_totals['count']
    
    return cluster_totals


# In[30]:


#### LOAD DATA #### 
#nba data
nba_df = read_nba()
nba_X, nba_y, nba_x_sc = prep_nba(nba_df)

#wine data
wine_df, wine_sc = load_wine()


# In[16]:


###### RUN NBA CLUSTER ANALYSIS #####
k_means_loop(nba_x_sc,20)


# In[19]:


nba_X = final_kmeans(6,nba_x_sc,nba_X)


# In[25]:


pd.DataFrame(avg_feature(nba_X))


# In[31]:


###### RUN WINE CLUSTER ANALYSIS #####
k_means_loop(wine_sc,20)


# In[32]:


wine_clust = final_kmeans(3,wine_sc,wine_df)


# In[33]:


pd.crosstab(wine_clust['target'],wine_clust['cluster'])


# In[38]:


def run_pca_loop(scaled_data,columns):

    pca = PCA()
    pca_X = pca.fit_transform(scaled_data)
    #create component indices
    component_names = ["comp_"+str(comp) for comp in range(1, len(pca.explained_variance_)+1)]

    #generate new component dataframe
    pca_X_df = pd.DataFrame(pca_X,columns=component_names)

    #generate component loadings on original features
    component_matrix = pd.DataFrame(pca.components_, index=component_names, columns = columns)

    #add additional columns to describe what
    component_matrix["explained_variance_ratio"] = pca.explained_variance_ratio_
    component_matrix["eigenvalue"] = pca.explained_variance_

    print(plt.plot(component_names,component_matrix.explained_variance_ratio.cumsum()))


# In[84]:


def set_pca(input_data, n_feats):
    pca = PCA()
    pca_X = pd.DataFrame(pca.fit_transform(input_data))
    return pca_X.iloc[:,:n_feats]


# In[87]:


nba_pca_df = set_pca(nba_x_sc,4)


# In[89]:


wine_pca_df = set_pca(wine_sc,7)


# In[39]:


######################################
### RUN THE PCA on the NBA dataset ###
######################################
run_pca_loop(nba_x_sc,nba_X.columns)


# In[43]:


#######################################
### RUN THE PCA on the wine dataset ###
#######################################
run_pca_loop(wine_sc,wine_df.columns[:-2])


# In[ ]:


###### SET PCA DFs moving forward


# In[64]:


nba_pca_df = nba_pca_X_df.iloc[:,:-2]


# In[52]:


wine_pca_df = wine_pca_df_X.iloc[:,:-1]


# In[45]:


##### ICA DATA #####

def print_kurtosis(scaled_data):
#print the kurtosis of the scaled data
    print "Kurotsis of original DF:", kurtosis(scaled_data)

    #print the kurtosis of the ICA transformed columns 
    for i in range(1,len(scaled_data[0])+1):
        ica = FastICA(n_components=i)
        ica_fit = ica.fit_transform(scaled_data)

        print "Kurtosis of ICA Transformed data when i=" + str(i) + ":", kurtosis(ica_fit)


# In[47]:


def print_ica_plot(comp,scaled_data):
    ica = FastICA(n_components=comp)
    ica_fit = ica.fit_transform(scaled_data)
    ica_df = pd.DataFrame(ica_fit)

    sns.pairplot(ica_df)


# In[ ]:


##### RUN NBA ICA #####


# In[46]:


print_kurtosis(nba_x_sc)


# In[48]:


print_ica_plot(5,nba_x_sc)


# In[49]:


#### RUN WINE ICA ####


# In[50]:


print_kurtosis(wine_sc)


# In[52]:


print_ica_plot(7,wine_sc)


# In[53]:


##################################################
### RUN RANDOMIZED PROJECTION ON BOTH DATASETS ###
##################################################

#check the Johnson Lindenstrauss minimum dimensions

def min_features(scaled_data):
    print johnson_lindenstrauss_min_dim(len(scaled_data),eps=0.1)
    
    
def run_randomized_components_analysis(input_data, target_data):
    #split our data first
    X_sc_train, X_sc_test, y_train, y_test = train_test_split(input_data,target_data,test_size=0.33,random_state =42)
    
    #set baseline
    lr = LogisticRegression()
    lr.fit(X_sc_train,y_train)
    baseline_preds = lr.predict(X_sc_test)
    baseline = accuracy_score(y_test, baseline_preds)

    #loop over n_components to test randomized projections to see which is best
    accuracies = []
    for i in range(1,len(X_sc_train[0])+1):
        transformer = random_projection.GaussianRandomProjection(n_components=i, random_state=5000000)
        X_new = transformer.fit_transform(X_sc_train)
        lr_rand = LogisticRegression()
        lr_rand.fit(X_new,y_train)
        test_data = transformer.transform(X_sc_test)
        new_preds = lr_rand.predict(test_data)
        accuracies.append(accuracy_score(y_test,new_preds))
    return baseline, accuracies


# In[56]:


print "JL Min Features NBA:", min_features(nba_x_sc)
print "JL Min Features Wine:", min_features(wine_sc)


# In[58]:


#run this for the NBA data
results = run_randomized_components_analysis(nba_x_sc,nba_y)
print "baseline score =", results[0]
print "best RCA component =", results[1].index(max(results[1]))+1
print "best RCA component score:", max(results[1])
print "Did RCA improve score of model?:", max(results[1]) > results[0]


# In[59]:


#set NBA data to best RCA
nba_transformer = random_projection.GaussianRandomProjection(n_components=5)
rca_nba = pd.DataFrame(nba_transformer.fit_transform(nba_x_sc))


# In[60]:


#run this for the wine data
results = run_randomized_components_analysis(wine_sc,wine_df['target'])
print "baseline score =", results[0]
print "best RCA component =", results[1].index(max(results[1]))+1
print "best RCA component score:", max(results[1])
print "Did RCA improve score of model?:", max(results[1]) > results[0]


# In[61]:


#set NBA data to best RCA
winetransformer = random_projection.GaussianRandomProjection(n_components=9)
rca_wine = pd.DataFrame(winetransformer.fit_transform(wine_sc))


# In[62]:


##########################################
### RUN CLUSTERING ON TRANSFORMED DATA ###
##########################################
def run_clusters(transformed_data):
    # k means determine k
    distortions = []
    K = range(1,20)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(transformed_data)
        kmeanModel.fit(transformed_data)
        distortions.append(sum(np.min(cdist(transformed_data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / transformed_data.shape[0])
        
    plt.plot(range(1,20),distortions)


# In[91]:


run_clusters(nba_pca_df)


# In[92]:


nba_pca_clust = KMeans(n_clusters=6).fit(nba_pca_df).labels_
nba_pca_df['cluster'] = nba_pca_clust


# In[93]:


run_clusters(wine_pca_df)


# In[94]:


run_clusters(nba_ica_df)


# In[ ]:


nba_ica_clust = KMeans(n_clusters=6).fit(nba_ica_df).labels_
nba_ica_df['cluster'] = nba_ica_clust


# In[ ]:


run_clusters(wine_ica_df)


# In[ ]:


run_clusters(rca_nba)


# In[ ]:


nba_rca_clust = KMeans(n_clusters=6).fit(rca_nba).labels_
rca_nba['cluster'] = nba_rca_clust


# In[ ]:


run_clusters(rca_wine)


# In[72]:


###############################################
### NEURAL NET  CLASSIFIER FROM HOMEWORK #1 ###
###############################################

def run_nn(X,y):
    # Create CV training and test scores for various training set sizes 
    #this is for Neural Network Classification problem of shots made
    train_sizes1, train_scores1, test_scores1 = learning_curve(MLPClassifier(),
                                                            X, 
                                                            y,
                                                            # Number of folds in cross-validation
                                                            cv=10,
                                                            # Evaluation metric
                                                            scoring='accuracy',
                                                            # Use all computer cores
                                                            n_jobs=-1, 
                                                            # 50 different sizes of the training set
                                                            train_sizes=np.linspace(0.01, 1.0, 50))

    # Create means and standard deviations of training set scores
    train_mean1 = np.mean(train_scores1, axis=1)
    train_std1 = np.std(train_scores1, axis=1)

    print "Avg. Accuracy Score of Training Set: ", np.mean(train_mean1)
    # Create means and standard deviations of test set scores
    test_mean1 = np.mean(test_scores1, axis=1)
    test_std1 = np.std(test_scores1, axis=1)
    print "Avg. Accuracy Score of Test Set: ", np.mean(test_mean1)

    # Draw lines
    plt.plot(train_sizes1, train_mean1, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes1, test_mean1, color="#111111", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes1, train_mean1 - train_std1, train_mean1 + train_std1, color="#DDDDDD")
    plt.fill_between(train_sizes1, test_mean1 - test_std1, test_mean1 + test_std1, color="#DDDDDD")

    # Create plot
    plt.title("Learning Curve for Shot Made Classification Problem Neural Network")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


# In[351]:


#run neural network on base data:
run_nn(X,y)


# In[352]:


#run neural network on PCA
run_nn(nba_pca_df,y)


# In[353]:


#run neural network on ICA
run_nn(nba_ica_df,y)


# In[354]:


nba_ica_df#run neural network on ICA
run_nn(rca_nba,y)


# In[95]:


###  ###########################################
# ################# PART 5 ##################
# ###########################################

# Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction 
# algorithms (you've probably already done this), treating the clusters as if they were new features. 
# In other words, treat the clustering algorithms as if they were dimensionality reduction algorithms. 
# Again, rerun your neural network learner on the newly projected data. THUR/FRI 


# In[98]:


def cluster_dummies(df):
    dummies = pd.get_dummies(df['cluster'])
    add_dummies = pd.merge(df,dummies, how='inner',left_index=True,right_index=True)
    add_dummies.drop('cluster',axis=1,inplace=True)
    
    return add_dummies


# In[100]:


run_nn(cluster_dummies(nba_pca_df),y)


# In[101]:


run_nn(cluster_dummies(nba_ica_df),y)


# In[102]:


run_nn(cluster_dummies(rca_nba),y)

