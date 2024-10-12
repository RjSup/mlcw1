import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_penguins():
    url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv'
    return pd.read_csv(url)

def preprocess(x, y, apply_pca=False, n_components=2, scale=True, handle_missing='impute'):
    if handle_missing == 'delete':
        mask = np.isnan(x).any(axis=1)
        x = x[~mask]
        y = y[~mask]
    elif handle_missing == 'impute':
        imputer = SimpleImputer(strategy='mean')
        x = imputer.fit_transform(x)
    
    x = np.nan_to_num(x)
    x_scaled = StandardScaler().fit_transform(x) if scale else x
    x_processed = PCA(n_components=n_components).fit_transform(x_scaled) if apply_pca else x_scaled
    return x_processed, y

def calculate_metrics(y_true, y_pred):
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1_score = np.zeros(n_classes)
    
    for i, cls in enumerate(classes):
        true_positive = np.sum((y_true == cls) & (y_pred == cls))
        false_positive = np.sum((y_true != cls) & (y_pred == cls))
        false_negative = np.sum((y_true == cls) & (y_pred != cls))
        
        precision[i] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall[i] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'macro_f1_score': np.mean(f1_score)
    }

def model_performance(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    train_pred, test_pred = model.predict(x_train), model.predict(x_test)
    
    train_metrics = calculate_metrics(y_train, train_pred)
    test_metrics = calculate_metrics(y_test, test_pred)
    
    return {
        'model': model,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'y_test': y_test,
        'y_pred': test_pred,
        'x_test': x_test
    }

def naive_bayes(x_train, x_test, y_train, y_test, tune=False):
    result = model_performance(GaussianNB(), x_train, x_test, y_train, y_test)
    result['best_type'] = 'GaussianNB'
    return result

def knn(x_train, x_test, y_train, y_test, k_values, tune=False):
    if tune:
        grid_search = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': k_values}, cv=5, scoring='accuracy')
        grid_search.fit(x_train, y_train)
        best_k = grid_search.best_params_['n_neighbors']
        knn = grid_search.best_estimator_
    else:
        results = [(k, model_performance(KNeighborsClassifier(n_neighbors=k), x_train, x_test, y_train, y_test)['test_metrics']['accuracy']) for k in k_values]
        best_k, _ = max(results, key=lambda x: x[1])
        knn = KNeighborsClassifier(n_neighbors=best_k)
    
    result = model_performance(knn, x_train, x_test, y_train, y_test)
    result['k'] = best_k
    return result

def experiments(x, y, apply_pca=False, use_cv=False, tune=False, handle_missing='impute'):
    k_values = range(1, 22, 2)
    results = {'naive_bayes': [], 'knn': []}
    
    x_processed, y_processed = preprocess(x, y, apply_pca=apply_pca, handle_missing=handle_missing)
    
    if use_cv:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for fold, (train_index, test_index) in enumerate(skf.split(x_processed, y_processed), 1):
            x_train, x_test = x_processed[train_index], x_processed[test_index]
            y_train, y_test = y_processed[train_index], y_processed[test_index]
            
            nb_result = naive_bayes(x_train, x_test, y_train, y_test, tune=tune)
            nb_result['fold'] = fold
            results['naive_bayes'].append(nb_result)
            
            knn_result = knn(x_train, x_test, y_train, y_test, k_values, tune=tune)
            knn_result['fold'] = fold
            results['knn'].append(knn_result)
    else:
        split_ratios = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        for test_size in split_ratios:
            x_train, x_test, y_train, y_test = train_test_split(x_processed, y_processed, test_size=test_size, stratify=y_processed, random_state=42)
            
            nb_result = naive_bayes(x_train, x_test, y_train, y_test, tune=tune)
            nb_result['split'] = f"{1-test_size:.1f}-{test_size:.1f}"
            results['naive_bayes'].append(nb_result)
            
            knn_result = knn(x_train, x_test, y_train, y_test, k_values, tune=tune)
            knn_result['split'] = f"{1-test_size:.1f}-{test_size:.1f}"
            results['knn'].append(knn_result)
    
    return results

def plot_accuracy_comparison(results, use_cv):
    plt.figure(figsize=(12, 6))
    x = range(len(results['naive_bayes']))
    plt.plot(x, [r['test_metrics']['accuracy'] for r in results['naive_bayes']], 'bo-', label='Naive Bayes')
    plt.plot(x, [r['test_metrics']['accuracy'] for r in results['knn']], 'ro-', label='KNN')
    plt.xlabel('Fold' if use_cv else 'Split')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    plt.xticks(x, [r['fold' if use_cv else 'split'] for r in results['naive_bayes']])
    plt.show()

def plot_best_k(results, use_cv):
    plt.figure(figsize=(12, 6))
    x = range(len(results['knn']))
    plt.plot(x, [r['k'] for r in results['knn']], 'go-')
    plt.xlabel('Fold' if use_cv else 'Split')
    plt.ylabel('Best k')
    plt.title('Best k for KNN')
    plt.xticks(x, [r['fold' if use_cv else 'split'] for r in results['knn']])
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_roc_curve(model, X_test, y_test, title):
    y_score = model.predict_proba(X_test)
    classes = np.unique(y_test)
    n_classes = len(classes)
    
    plt.figure(figsize=(8, 6))
    for i, class_label in enumerate(classes):
        y_true = (y_test == class_label).astype(int)
        if np.sum(y_true) > 0:  # Only plot if there are positive samples
            fpr, tpr, _ = roc_curve(y_true, y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve (class {class_label}) (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def print_metrics(results, model_type):
    print(f"\n{model_type.capitalize()} Results:")
    for result in results:
        print(f"Split/Fold: {result.get('split') or result.get('fold')}")
        print("Train Metrics:")
        print(f"  Accuracy: {result['train_metrics']['accuracy']:.4f}")
        print(f"  Macro F1-score: {result['train_metrics']['macro_f1_score']:.4f}")
        print("Test Metrics:")
        print(f"  Accuracy: {result['test_metrics']['accuracy']:.4f}")
        print(f"  Macro F1-score: {result['test_metrics']['macro_f1_score']:.4f}")
        if model_type == 'knn':
            print(f"  Best k: {result['k']}")
        print()

def visualise_results(results, use_cv, apply_pca, tune, handle_missing):
    plot_accuracy_comparison(results, use_cv)
    plot_best_k(results, use_cv)
    
    print_metrics(results['naive_bayes'], 'naive_bayes')
    print_metrics(results['knn'], 'knn')
    
    for model_type in ['naive_bayes', 'knn']:
        last_result = results[model_type][-1]
        plot_confusion_matrix(last_result['y_test'], last_result['y_pred'], f'Confusion Matrix - {model_type.capitalize()}')
        plot_roc_curve(last_result['model'], last_result['x_test'], last_result['y_test'], f'ROC Curve - {model_type.capitalize()}')

    print(f"\nExperiment Configuration:")
    print(f"Cross-validation: {'Yes' if use_cv else 'No'}")
    print(f"PCA: {'Yes' if apply_pca else 'No'}")
    print(f"Hyperparameter Tuning: {'Yes' if tune else 'No'}")
    print(f"Missing Data Handling: {handle_missing}")

def main():
    penguins = load_penguins()
    x = penguins.drop(['species', 'island', 'sex'], axis=1).values
    y = penguins['species'].values
    
    for handle_missing in ['delete', 'impute']:
        for tune in [False, True]:
            for cv in [False, True]:
                for pca in [False, True]:
                    logging.info(f"Experiments with missing data {handle_missing}, {'with' if cv else 'without'} CV, {'with' if pca else 'without'} PCA, {'with' if tune else 'without'} tuning:")
                    results = experiments(x, y, apply_pca=pca, use_cv=cv, tune=tune, handle_missing=handle_missing)
                    visualise_results(results, cv, pca, tune, handle_missing)

if __name__ == "__main__":
    main()