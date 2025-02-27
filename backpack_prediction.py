import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from cuml.preprocessing import TargetEncoder
import optuna

sns.set(style="whitegrid", font_scale=1.2)

categorical_columns = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
numerical_columns = ['Weight Capacity (kg)']
nominal_columns = ['Brand', 'Material', 'Style', 'Color']
target_column = ['Price']

# Load the data
train_df = pd.read_csv('dataset/train.csv').drop('id', axis=1)
test_df = pd.read_csv('dataset/test.csv')

test_id = test_df['id']
test_df = test_df.drop('id', axis=1)


# Evaluate the model
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Handling Missing Values
def handle_missing_values(df):
    # Fill missing values for categorical columns with 'None'
    for col in categorical_columns:
        df[col].fillna('None' , inplace=True)

    # # Fill missing values for categorical columns with the mode
    # for col in categorical_columns:
    #     df[col].fillna(df[col].mode()[0], inplace=True)
    # Fill missing values for numerical columns with the mean
    for col in numerical_columns:
        df[col].fillna(df[col].mean(), inplace=True)

    return df


# Convert categorical columns to category type
def convert_categorical_columns(df):
    for col in categorical_columns:
        try:
            df[col] = df[col].astype('category')
        except KeyError as e:
            print(f"Column {col} not found in one of the datasets: {e}")
        except Exception as e:
            print(f"Error converting {col} to category: {e}")
    return df


# Feature Engineering

# Mapping columns
def map_columns(df):
    df['Size'] = df['Size'].map({'Small': 1, 'Medium': 2, 'Large': 3}).astype('Int64')
    df['Compartments'] = df['Compartments'].astype('Int64')
    df['Compartments'] = df['Compartments']/df['Compartments'].max()
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'No': 0, 'Yes': 1}).astype('Int64')
    df['Waterproof'] = df['Waterproof'].map({'No': 0, 'Yes': 1}).astype('Int64')

    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)']/df['Weight Capacity (kg)'].max()

    return df


# Split the data into train and validation sets
def split_data(df):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val

# One Hot Encoder Generator
def one_hot_encoder(X):

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    oh_encoded = ohe.fit_transform(X[nominal_columns])
    ohdf = pd.DataFrame(oh_encoded, columns=ohe.get_feature_names_out(nominal_columns))
    X_encoded = pd.concat([X.drop(nominal_columns, axis=1), ohdf], axis=1)
    return X_encoded, ohe, oh_encoded


# One Hot Encoding for test data
def one_hot(test_df, encoder):
    oh_encoded_test = encoder.transform(test_df[nominal_columns])

    ohdf = pd.DataFrame(oh_encoded_test, columns=encoder.get_feature_names_out(nominal_columns))
    test_df_encoded = pd.concat([test_df.drop(nominal_columns, axis=1), ohdf], axis=1)
    return test_df_encoded


# Target Encoding
def target_encode(X, y):
    for col in nominal_columns:
        mean_encoded = X.groupby(col)[y].mean()
        X[col] = X[col].map(mean_encoded)
    return X


# PCA   
def X_pca(X, n_components=0.95):
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(X)

    pca_df = pd.DataFrame(
    pca_features, 
    columns=[f'PCA_{i+1}' for i in range(pca_features.shape[1])]
)
    X_processed = pd.concat([
    X,
    pca_df
], axis=1)

    return pca_features, pca, X_processed


def test_pca(test_df, pca):
    pca_features_test = pca.transform(test_df)

    pca_df_test = pd.DataFrame(
        pca_features_test, 
        columns=[f'PCA_{i+1}' for i in range(pca_features_test.shape[1])]
    )

    test_df_final = pd.concat([
    test_df,
    pca_df_test
], axis=1)

    return test_df_final


# Random Search for Hyperparameter Tuning
def random_search(X_train, y_train):
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    params = {
        'n_estimators': [100, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.25, 0.1],
        'max_depth': [1, 3, 5, 7, 9],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.4, 0.6, 0.8],
        'colsample_bytree': [0.5, 0.8, 0.9],
        'reg_lambda': [0, 1, 5, 10],
        'min_split_loss': [0, 0.1, 0.5, 1, 3, 5, 10],
        'reg_alpha': [0, 0.1, 1]
    }
    
    random_search = RandomizedSearchCV(xgb, params, 
                                     cv=5, 
                                     scoring='neg_mean_squared_error',
                                     random_state=42, 
                                     verbose=2, 
                                     n_jobs=-1)
    
    random_search.fit(X_train, y_train)
    print("Best parameters:", random_search.best_params_)
    print("Best Score:", random_search.best_score_)

    return random_search.best_params_, random_search.best_score_


# Random Forest Model
def random_forest(X_train, y_train, X_test, params=None):
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
    
    # Initialize and train model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_test_pred = model.predict(X_test)
    
    return y_test_pred
    

# XGBoost Model
def xgboost_model(X_train, y_train, X_test, params=None):
    if params is None:
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    
    model = XGBRegressor(objective='reg:squarederror', early_stopping_rounds=50, eval_metric='rmse',random_state=42, **params)
    
    # Initialize and train model
    model.fit(X_train, y_train,eval_set=[(X_test, y_test)])
    
    # Make predictions
    y_test_pred = model.predict(X_test, iteration_range=(0, model.best_iteration))
    
    return y_test_pred


# Ensemble Model
def ensemble_model(X_train, y_train, X_test, params=None):
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
    
    params2 = params.copy()
    params2['max_depth'] = params['max_depth'] + 1

    params3 = params.copy()
    params3['learning_rate'] = params['learning_rate'] * 0.8

    model1 = XGBRegressor(**params)
    model2 = XGBRegressor(**params2)
    model3 = XGBRegressor(**params3)

    ensemble = VotingRegressor([
        ('m1', model1),
        ('m2', model2),
        ('m3', model3)
    ])  

    ensemble.fit(X_train, y_train)

    # Make predictions
    y_test_pred = ensemble.predict(X_test)
    
    return y_test_pred


# Find best hyperparameters for CatBoost
def tune_hyperparameters(X_train, y_train,n_trials=20):
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 1500),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0, 1),
            "random_strength": trial.suggest_float("random_strength", 0, 1),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "verbose": False
            # "task_type": "GPU",  # Use GPU for CatBoost
            # "devices": "0"  # Specify GPU device
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_idx, valid_idx in kf.split(X_train):
            X_train_kf = X_train.iloc[train_idx]
            y_train_kf = y_train.iloc[train_idx]
            X_val_kf = X_train.iloc[valid_idx]
            y_val_kf = y_train.iloc[valid_idx]
            
            # Catboost Model
            model = CatBoostRegressor(**params)
            model.fit(X_train_kf, y_train_kf, eval_set=(X_val_kf, y_val_kf), early_stopping_rounds=50, verbose=0)
            
            # Assessment and Scores
            preds = model.predict(X_val_kf)
            rmse = np.sqrt(mean_squared_error(y_val_kf, preds))
            scores.append(rmse)
        

        return np.mean(scores)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_trial.params


# Catboost Model
def catboost_model(X_train, y_train, X_test, y_test, params):

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=0)

    # Make predictions
    y_test_pred = model.predict(X_test)

    return y_test_pred, model

#############################Data Preprocessing######################################
train_df_handled = handle_missing_values(train_df.copy())
test_df_handled = handle_missing_values(test_df.copy())

train_df_handled = convert_categorical_columns(train_df_handled)
test_df_handled = convert_categorical_columns(test_df_handled)

#############################Feature Engineering######################################
train_df_handled = map_columns(train_df_handled)
test_df_handled = map_columns(test_df_handled)

print(train_df_handled)

X = train_df_handled.drop(target_column, axis=1)
y = train_df_handled[target_column]

# X_encoded, ohe, oh_encoded = one_hot_encoder(X)
# test_df_encoded = one_hot(test_df_handled, ohe)

X_encoded = target_encode(X, y)

# print(X_encoded)

# pca_features, pca, X_processed = X_pca(X_encoded)
# test_processed = test_pca(test_df_encoded, pca)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


#############################Model######################################
# print("原始One-Hot编码特征数:", oh_encoded.shape[1])
# print("PCA降维后特征数:", pca_features.shape[1])
# print("解释方差比:", sum(pca.explained_variance_ratio_))


# best_params, _ = random_search(X_train, y_train)
# best_params = {
#     'subsample': 0.4,
#   'reg_lambda': 0,
#   'reg_alpha': 0,
#   'n_estimators': 1000,
#   'min_split_loss': 0,
#   'min_child_weight': 5,
#   'max_depth': 1,
#   'learning_rate': 0.1,
#   'colsample_bytree': 0.9
# }

# catboost_best_params = tune_hyperparameters(X_train, y_train)
# print("Best CatBoost Hyperparameters:", catboost_best_params)

catboost_best_params = {'iterations': 1168, 
     'depth': 3, 
     'learning_rate': 0.09617371243696217, 
     'l2_leaf_reg': 0.3327534768702417, 
     'random_strength': 0.9282015330678425, 
     'bagging_temperature': 0.7102677755287592, 
     'border_count': 142}

y_test_pred, cat_model= catboost_model(X_train, y_train, X_test, y_test, catboost_best_params)

# y_test_pred = ensemble_model(X_train, y_train, X_test, best_params)

# y_test_pred = xgboost_model(X_train, y_train, X_test, best_params)

# y_test_pred = random_forest(X_train, y_train, X_test)



print(f"RMSE: {rmse(y_test, y_test_pred)}")
############################################################
# Prediction on test data
final_predictions = cat_model.predict(test_df_encoded)

# Save predictions to CSV
submission_df = pd.DataFrame({
   'id': test_id,
   'Price': final_predictions
})
submission_df.to_csv('submission.csv', index=False)

print(submission_df.head(10))