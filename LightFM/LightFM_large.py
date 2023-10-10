# import libraries
import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k

# ----------------- 1. Read And Process Data ----------------- #
train = pd.read_csv("/Users/colleenjin/Desktop/Data/train.csv")
train.rename(columns = {'user':'user_id', 'item':'recording_id', 'rating':'count'}, inplace = True)

val = pd.read_csv("/Users/colleenjin/Desktop/Data/val.csv")
val.rename(columns = {'user':'user_id', 'item':'recording_id', 'rating':'count'}, inplace = True)

test = pd.read_csv("/Users/colleenjin/Desktop/Data/test.csv")
test.rename(columns = {'user':'user_id', 'item':'recording_id', 'rating':'count'}, inplace = True)

# Extract 1/100 of the validation and test data
sampled_val= val.sample(frac=0.0005)
sampled_test = test.sample(frac=0.0005)

# ----------------- 2. Fit Dataset ----------------- #
total = pd.concat([train, sampled_val, sampled_test]).drop_duplicates()
unique_users = total['user_id'].unique()
unique_items = total['recording_id'].unique()

data = Dataset()
data.fit(users = unique_users, items = unique_items)

# ----------------- 3. Build Interaction Matrix ----------------- #
interactions_train, weights_train = data.build_interactions([(train['user_id'][i], 
                                                              train['recording_id'][i], 
                                                              train['count'][i]) for i in range(train.shape[0])])
interactions_val, weights_val = data.build_interactions([(sampled_val.iloc[i]['user_id'],
                                                          sampled_val.iloc[i]['recording_id'],
                                                          sampled_val.iloc[i]['count']) for i in range(sampled_val.shape[0])])
interactions_test, weights_test = data.build_interactions([(sampled_test.iloc[i]['user_id'],
                                                          sampled_test.iloc[i]['recording_id'],
                                                          sampled_test.iloc[i]['count']) for i in range(sampled_test.shape[0])])

# ----------------- 4. Training the model and parameter tuning ----------------- #
param_grid = {
        'max_iter': [1, 5],
        # 'learning_schedule': ['adagrad', 'adadelta'],
        'no_components': [10, 20, 30],
        'loss': ['warp', 'bpr'],
        'user_alpha': [1, 0.1, 0.01],
        'item_alpha': [1, 0.1, 0.01],
    }

# Define a function to convert np array of precisions to average precision
def calculate_map_at_k(model, interactions_test, k=100):
    precision = precision_at_k(model, interactions_test, k=k)
    ap = []
    for user_precision in precision:
        num_correct = sum(user_precision)
        if num_correct > 0:
            user_ap = sum([(correct / (i + 1)) for i, correct in enumerate(user_precision) if correct > 0]) / num_correct
        else:
            user_ap = 0.0
        ap.append(user_ap)
    
    map_at_k = sum(ap) / len(ap)  # Calculate Mean Average Precision (MAP) across all users
    
    return map_at_k


for max_iter in param_grid['max_iter']:  # number of epochs
    for no_components in param_grid['no_components']:  # latent dimensionality
        for loss in param_grid['loss']:  # loss function
            for user_alpha in param_grid['user_alpha']:  # regularization parameter
                for item_alpha in param_grid['item_alpha']:  # regularization parameter
                    print(f"start training {loss} model")
                    start = time.time()
                    print("initiating LightFM...")
                    model = LightFM(no_components=no_components,
                                    loss=loss,
                                    user_alpha=user_alpha)
                    print("fitting LightFM...")
                    model = model.fit(interactions=interactions_train, sample_weight=weights_train, epochs=1, verbose=False)
                    end = time.time()
                    print(f"Time cost (fit): {end-start}")
                    
                    print("computing precision_at_k...")
                    start = time.time()
                    map_at_100 = calculate_map_at_k(model, interactions_val, k=100)
                    end = time.time()
                    

                    print(f"LightFM model with rank=100, no_components={no_components}, loss={loss}, user_alpha={user_alpha}:")
                    print(f"Average precision at k=100: {map_at_100}")
                    print(f"Time cost (evaluate): {end-start}")
                    
# ----------------- 5. Evaluate the model ----------------- #

# Final set of parameters
k = 100
no_components = 10
loss = 'warp'
user_alpha = 0.1
item_alpha = 0
epoch = 1

print(f"start training {loss} model")
start = time.time()
print("initiating LightFM...")
model = LightFM(no_components=no_components,
                loss=loss,
                user_alpha=user_alpha,
                item_alpha=item_alpha)
print("fitting LightFM...")
model = model.fit(interactions=interactions_train, sample_weight=weights_train, epochs=1, verbose=False)
end = time.time()
print(f"Time cost (fit): {end-start}")

print("computing precision_at_k...")
start = time.time()
val_precision = precision_at_k(model, interactions_test, k=k).mean()
end = time.time()

print(f"LightFM model with rank={k}, no_components={no_components}, loss={loss}, user_alpha={user_alpha}, item_alpha={item_alpha}:")
print(f"Average precision at k={k}: {val_precision}")
print(f"Time cost (evaluate): {end-start}")