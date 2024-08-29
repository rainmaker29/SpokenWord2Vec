import os
default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"
import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
import torch 
import argparse
import gc
 
parser = argparse.ArgumentParser(description="Feature processing script.")
parser.add_argument('--feature_type', type=str, default='hubert', choices=['mfcc', 'wav2vec2', 'hubert'], 
        help='Specify the type of features to process (mfcc, wav2vec2, hubert).')
args = parser.parse_args()



# Directory where the feature files are stored
output_dir = os.path.join("./features", f"{args.feature_type}")
feats_dir = os.path.join(output_dir, "speakers")
output_file = os.path.join(output_dir, "kmeans_")

K = 100
normalize = True

with open(os.path.join(output_dir, "feats.pkl"), "rb") as f:
    wv = pickle.load(f)
 

# Assuming `data` is a list of lists of feature sequences.
# Each element in `data` is a list of words, each word is a sequence of 39-dimension MFCC features (or other features, like wav2vec2)

# Flatten the data and sample 10% of it, to enable faster Means
print("flattening... ")
all_features = [feature for word in wv for feature in word.squeeze(0).numpy()]

#normalize the features:
print("convert to numpy array ... ")
all_features = np.array(all_features)

if normalize:
    print("Normalizing ...")
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)


sampled_features, _ = train_test_split(all_features, train_size=0.1)  

data = np.vstack(sampled_features)

# Clustering
print('Clustering ...')
n_clusters = K  
kmeans = KMeans(n_clusters=n_clusters).fit(sampled_features)

centroids = kmeans.cluster_centers_

# Transform Data
print('Transforming Data ...')
#following will store the data as sequence of cluster ids .. 
data_transformed_centroids = []
#following is the same as above, but any contiguous identical cluster ids will be merged into one.
data_transformed_denoised = []

del data
del sampled_features
i=0
for word in tqdm(wv, total=len(wv)):
    word = word.squeeze(0)
    j=word.shape[0]
    c_ids=kmeans.predict(all_features[i:i+j])
    i += j
    unique_elements = [c_ids[0]]
    for element in c_ids[1:]:
        if element != unique_elements[-1]:
            unique_elements.append(element)
    unique_c_ids= np.array(unique_elements, dtype=c_ids.dtype)
    data_transformed_denoised.append(unique_c_ids.tolist())
    data_transformed_centroids.append([centroids[unique_c_ids]])

print(len(data_transformed_centroids), len(data_transformed_centroids[0]), data_transformed_centroids[0])
print(len(data_transformed_denoised), len(data_transformed_denoised[0]), data_transformed_denoised[0])
with open(output_file+str(K)+"_centroids.pkl","wb") as f:
    pickle.dump(data_transformed_centroids,f)
with open(output_file+str(K)+"_cids.pkl","wb") as f:
    pickle.dump(data_transformed_denoised,f)

