import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.utils import resample
from scipy import stats
from scipy.stats import spearmanr


# Data processing (Task 1)
def stratified_split(input_df, split_ratio=0.2, seed=42):
    ''' 
    - loads df
    - bins by age
    - splits to training / validation sets 
    '''
    df = input_df.copy()
    
    # 10 bins using quantiles: ~45 samples/bin
    df['age_bins'] = pd.qcut(df['age'], q=10, labels=False, duplicates='drop')

    train_df, val_df = train_test_split(df,
                                        test_size=split_ratio,
                                        random_state=seed,
                                        stratify=df['age_bins'])

    return train_df.drop(columns=['age_bins']), val_df.drop(columns=['age_bins'])


def xyz(df, pipe, is_train=False):
    '''
    Features (X), Targets (y) & Metadata (z) generation
    - Features (X): imputed & scaled 
    - Metadata: sex & ethnicity binary encoded 
    '''
    y = df['age']

    z = pd.get_dummies(df[['sex', 'ethnicity']], drop_first=True)
    z.columns = ['is_Male', 'is_Hispanic'] # Sex: M/F, Ethnicity: Hispanic/Caucasian 
    
    X = df.drop(columns=['sample_id', 'age', 'ethnicity', 'sex'])

    X = pipe.fit_transform(X) if is_train else pipe.transform(X)
        
    return X, y, z


# Model Analysis (Task 2-4)
def bootstrap_metrics(y_true, y_pred, seed=42):
    '''
    Calculates RMSE, MAE, R^2, and Pearson r with 95% Confidence Intervals.
    '''
    np.random.seed(seed)
    metrics = []

    ids = np.arange(len(y_true))
    
    for _ in range(1000):
        bs_ids = resample(ids, replace=True)
        
        y_t = y_true.iloc[bs_ids]
        y_p = y_pred[bs_ids]
        
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae = mean_absolute_error(y_t, y_p)
        r2 = r2_score(y_t, y_p)
        pearson_r, _ = stats.pearsonr(y_t, y_p)
        
        metrics.append([rmse, mae, r2, pearson_r])
    
    metrics = np.array(metrics)
    
    means = np.mean(metrics, axis=0)
    stds = np.std(metrics, axis=0)
    lower_pt = np.percentile(metrics, 2.5, axis=0)
    upper_pt = np.percentile(metrics, 97.5, axis=0)
    
    names = ['RMSE', 'MAE', 'R2', 'Pearson_r']
    results = {}
    for i, name in enumerate(names):
        results[name] = f'{means[i]:.3f} ± {stds[i]:.3f}, [{lower_pt[i]:.3f} - {upper_pt[i]:.3f}]'
    
    return results, pd.DataFrame(metrics, columns=names)


def stability_selection(X_train, y_train, seed=42):
    '''
    Stability Selection using Spearman Correlation.
    '''
    np.random.seed(seed)
    
    N_samples, N_features = X_train.shape
    feature_names = X_train.columns
    
    selection_counts = pd.Series(0, index=feature_names)
    
    for i in range(50):
        sub_ids = np.random.choice(N_samples, size=int(N_samples * 0.8), replace=False)
        X_sub = X_train.iloc[sub_ids]
        y_sub = y_train.iloc[sub_ids]
        
        corrs = []
        for col in X_sub.columns:
            corr, _ = spearmanr(X_sub[col], y_sub)
            corrs.append(abs(corr))
        
        corr_series = pd.Series(corrs, index=feature_names)
        
        top_200 = corr_series.nlargest(200).index
        selection_counts[top_200] += 1
        
    stable_features = selection_counts[selection_counts > (25)].index.tolist()
    
    return stable_features, selection_counts


# B Bonus

def xyz_sex(df, pipe, is_train=False):
    '''
    Features (X) & Targets (y) generation
    - Features (X): imputed & scaled 
    -  Targets (y): sex binary encoded 
    '''
    y = pd.get_dummies(df[['sex']], drop_first=True)
    y = y.rename(columns={'sex_M': 'sex'})
    
    X = df.drop(columns=['sample_id', 'age', 'ethnicity', 'sex'])

    X = pipe.fit_transform(X) if is_train else pipe.transform(X)
        
    return X, y

def bootstrap_classification(model, X_test, y_test):
    metrics = {'Accuracy': [], 'F1': [], 'MCC': [], 'ROC-AUC': [], 'PR-AUC': []}
    n_samples = len(y_test)
    
    # Predict probabilities and classes once for the base model
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] 
    
    for _ in range(1000):
        ids = np.random.randint(0, n_samples, n_samples)
        y_boot = y_test.iloc[ids] if isinstance(y_test, pd.Series) else y_test[ids]
        preds_boot = preds[ids]
        probs_boot = probs[ids]
        
        try:
            metrics['Accuracy'].append(accuracy_score(y_boot, preds_boot))
            metrics['F1'].append(f1_score(y_boot, preds_boot))
            metrics['MCC'].append(matthews_corrcoef(y_boot, preds_boot))
            metrics['ROC-AUC'].append(roc_auc_score(y_boot, probs_boot))
            metrics['PR-AUC'].append(average_precision_score(y_boot, probs_boot))
        except ValueError:
            continue

    results = {}
    for k, v in metrics.items():
        results[k] = f"{np.mean(v):.3f} ± {np.std(v):.3f}, [{np.percentile(v, 2.5):.3f}-{np.percentile(v, 97.5):.3f}]"
    return results, metrics




# from sklearn.linear_model import BayesianRidge, LogisticRegression
# from sklearn.metrics import mean_squared_error, accuracy_score
# import numpy as np

# def auto_mrmr_optimize(X_train, y_train, X_val, y_val, k_range=range(10, 201, 5)):
#     """
#     Automatically detects if the task is Regression or Classification 
#     based on the target variable and runs the mRMR optimization.
#     """
    
#     # 1. Logic to determine Task Type
#     # We check if y is float-like (Regression) or object/int with few unique values (Classification)
#     is_regression = np.issubdtype(y_train.dtype, np.floating) or y_train.nunique() > 10
    
#     if is_regression:
#         print("--- Mode Detected: Regression (Bayesian Ridge) ---")
#         model = BayesianRidge()
#         metric_func = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
#         minimize = True
#     else:
#         print("--- Mode Detected: Classification (Logistic Regression) ---")
#         model = LogisticRegression(max_iter=1000)
#         metric_func = accuracy_score
#         minimize = False

#     # 2. Get mRMR Ranking once
#     max_k = max(k_range)
#     mrmr_rank = mrmr_regression(X=X_train, y=y_train, K=max_k, 
#                                  return_scores=False, show_progress=False)
    
#     # 3. Loop through Ks
#     ks = list(k_range)
#     scores = []

#     for k in ks:
#         features = mrmr_rank[:k]
#         model.fit(X_train[features], y_train)
#         y_hat = model.predict(X_val[features])
#         scores.append(metric_func(y_val, y_hat))

#     # 4. Find optimal result
#     best_idx = np.argmin(scores) if minimize else np.argmax(scores)
#     optimal_k = ks[best_idx]
    
#     print(f"Optimal K found: {optimal_k}")
#     return optimal_k, mrmr_rank[:optimal_k]












