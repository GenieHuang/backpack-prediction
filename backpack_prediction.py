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
import category_encoders as ce
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
    encoder = ce.TargetEncoder(cols=categorical_columns)
    X_encoded = encoder.fit_transform(X, y)

    return X_encoded, encoder


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
    
    return y_test_pred, model


# Ensemble Model
def ensemble_model(X_train, y_train, X_test, params):

    final_models = []
    for i in range(3):
        model = XGBRegressor(**params[f"model_{i}"])
        model.fit(X_train, y_train)
        final_models.append(model)

    weights = best_params["weights"]
    y_test_pred = np.zeros(len(X_test))
    for i, model in enumerate(final_models):
        y_test_pred += weights[i] * model.predict(X_test)
    
    return y_test_pred, final_models


def predict_with_ensemble(final_models, weights, X_test):
    
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    preds = np.zeros(len(X_test))
    
    for i, model in enumerate(final_models):
        model_preds = model.predict(X_test)
        preds += weights[i] * model_preds
    
    return preds


def process_params(best_params):
    processed_params = {}
    
    if any(k.startswith("model0_") for k in best_params.keys()):
        for i in range(3):
            model_prefix = f"model{i}_"
            processed_params[f"model_{i}"] = {
                k.replace(model_prefix, ""): v 
                for k, v in best_params.items() 
                if k.startswith(model_prefix)
            }
        
        weights = [
            best_params.get(f"weight{i}", 1.0) for i in range(3)
        ]
        
        if "weights" in best_params:
            weights = best_params["weights"]
        else:
            weights = [1.0, 1.0, 1.0]
    
    total_weight = sum(weights)
    processed_params["weights"] = [w / total_weight for w in weights]
    
    return processed_params


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


def ensemble_tune_hyperparameters(X_train, y_train, n_trials=20):
    def objective(trial):

        models_params = []
        for i in range(3):
            params = {
                "n_estimators": trial.suggest_int(f"model{i}_n_estimators", 100, 1000),
                "max_depth": trial.suggest_int(f"model{i}_max_depth", 3, 10),
                "learning_rate": trial.suggest_float(f"model{i}_learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float(f"model{i}_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float(f"model{i}_colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int(f"model{i}_min_child_weight", 1, 10),
                "gamma": trial.suggest_float(f"model{i}_gamma", 0, 1),
                "reg_alpha": trial.suggest_float(f"model{i}_reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float(f"model{i}_reg_lambda", 0, 1),
                "random_state": 42
            }
            models_params.append(params)
        

        weights = [
            trial.suggest_float("weight0", 0.1, 1.0),
            trial.suggest_float("weight1", 0.1, 1.0),
            trial.suggest_float("weight2", 0.1, 1.0)
        ]

        sum_weights = sum(weights)
        weights = [w/sum_weights for w in weights]
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, valid_idx in kf.split(X_train):
            X_train_kf = X_train.iloc[train_idx]
            y_train_kf = y_train.iloc[train_idx]
            X_val_kf = X_train.iloc[valid_idx]
            y_val_kf = y_train.iloc[valid_idx]
            

            models = []
            for i in range(3):
                model = XGBRegressor(**models_params[i])
                model.fit(
                    X_train_kf, y_train_kf,
                    eval_set=[(X_val_kf, y_val_kf)],
                    verbose=0
)
                models.append(model)
            

            ensemble_preds = np.zeros(len(y_val_kf))
            for i, model in enumerate(models):
                preds = model.predict(X_val_kf)
                ensemble_preds += weights[i] * preds
            

            rmse = np.sqrt(mean_squared_error(y_val_kf, ensemble_preds))
            scores.append(rmse)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    

    best_params = {
        "model_0": {k.replace("model0_", ""): v for k, v in study.best_params.items() if k.startswith("model0_")},
        "model_1": {k.replace("model1_", ""): v for k, v in study.best_params.items() if k.startswith("model1_")},
        "model_2": {k.replace("model2_", ""): v for k, v in study.best_params.items() if k.startswith("model2_")},
        "weights": [
            study.best_params["weight0"],
            study.best_params["weight1"],
            study.best_params["weight2"]
        ]
    }
    

    sum_weights = sum(best_params["weights"])
    best_params["weights"] = [w/sum_weights for w in best_params["weights"]]
    
    return best_params


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

X_encoded, target_encoder = target_encode(X, y)
test_df_encoded = target_encoder.transform(test_df_handled)

# print(X_encoded)

# pca_features, pca, X_processed = X_pca(X_encoded)
# test_processed = test_pca(test_df_encoded, pca)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


#############################Model######################################


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

# catboost_best_params= {'iterations': 1352, 
#                        'depth': 5, 
#                        'learning_rate': 0.022450500385312554, 
#                        'l2_leaf_reg': 0.4284227228311264, 
#                        'random_strength': 0.12546850412679345, 
#                        'bagging_temperature': 0.8198063284866197, 
#                        'border_count': 183}

# catboost_best_params = {'iterations': 1168, 
#      'depth': 3, 
#      'learning_rate': 0.09617371243696217, 
#      'l2_leaf_reg': 0.3327534768702417, 
#      'random_strength': 0.9282015330678425, 
#      'bagging_temperature': 0.7102677755287592, 
#      'border_count': 142}

# y_test_pred, cat_model= catboost_model(X_train, y_train, X_test, y_test, catboost_best_params)
# best_params = ensemble_tune_hyperparameters(X_train, y_train)

# best_params = {'model0_n_estimators': 252, 'model0_max_depth': 4, 'model0_learning_rate': 0.015143071414466724, 'model0_subsample': 0.8315345012418895, 'model0_colsample_bytree': 0.9644124617433968, 'model0_min_child_weight': 10, 'model0_gamma': 0.35216662983084956, 'model0_reg_alpha': 0.3105605677930154, 'model0_reg_lambda': 0.3547394710233012, 'model1_n_estimators': 557, 'model1_max_depth': 5, 'model1_learning_rate': 0.018331021449155598, 'model1_subsample': 0.7147085715855364, 'model1_colsample_bytree': 0.598194303762557, 'model1_min_child_weight': 8, 'model1_gamma': 0.7198936027399416, 'model1_reg_alpha': 0.6571128147750868, 'model1_reg_lambda': 0.6575855790717202, 'model2_n_estimators': 108, 'model2_max_depth': 9, 'model2_learning_rate': 0.021302419448423867, 'model2_subsample': 0.8992953092810227, 'model2_colsample_bytree': 0.6908558820353785, 'model2_min_child_weight': 9, 'model2_gamma': 0.8630403601869322, 'model2_reg_alpha': 0.49782641380343207, 'model2_reg_lambda': 0.031067778380564744, 'weight0': 0.8866750402055176, 'weight1': 0.5023241180485698, 'weight2': 0.666289954575352}
best_params = {'model0_n_estimators': 106, 'model0_max_depth': 3, 'model0_learning_rate': 0.03187905158741838, 'model0_subsample': 0.6763571582186845, 'model0_colsample_bytree': 0.9701448228171785, 'model0_min_child_weight': 7, 'model0_gamma': 0.3770878959183077, 'model0_reg_alpha': 0.7031272356524338, 'model0_reg_lambda': 0.21570348043226606, 'model1_n_estimators': 708, 'model1_max_depth': 3, 'model1_learning_rate': 0.07213387746367997, 'model1_subsample': 0.7066344693160838, 'model1_colsample_bytree': 0.6646023302345823, 'model1_min_child_weight': 3, 'model1_gamma': 0.16519992714604143, 'model1_reg_alpha': 0.9960020826994176, 'model1_reg_lambda': 0.2778732606529881, 'model2_n_estimators': 105, 'model2_max_depth': 7, 'model2_learning_rate': 0.018262010536530184, 'model2_subsample': 0.5061719940590499, 'model2_colsample_bytree': 0.6674518957827519, 'model2_min_child_weight': 6, 'model2_gamma': 0.7209954939624005, 'model2_reg_alpha': 0.503750496545143, 'model2_reg_lambda': 0.13426464968535673, 'weight0': 0.9992266581985025, 'weight1': 0.9822164895272267, 'weight2': 0.18676014421800924}
best_params = process_params(best_params)
y_test_pred, ensemble = ensemble_model(X_train, y_train, X_test, best_params)

# y_test_pred, xg_model= xgboost_model(X_train, y_train, X_test, best_params)

# y_test_pred = random_forest(X_train, y_train, X_test)



print(f"RMSE: {rmse(y_test, y_test_pred)}")
############################################################
# Prediction on test data
# final_predictions = cat_model.predict(test_df_encoded)
# final_predictions = ensemble.predict(test_df_encoded)
final_predictions = predict_with_ensemble(ensemble, best_params["weights"], test_df_encoded)

# Save predictions to CSV
submission_df = pd.DataFrame({
   'id': test_id,
   'Price': final_predictions
})
submission_df.to_csv('submission.csv', index=False)

print(submission_df.head(10))