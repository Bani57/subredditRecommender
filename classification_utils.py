import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

classifiers = {
    # 'KNN': KNeighborsClassifier(
    #     n_neighbors=30,
    #     weights="distance",
    #     algorithm="auto",
    #     metric=cosine,
    #     n_jobs=-1),
    'Logistic Regression': LogisticRegression(
        C=1e-2,
        solver='lbfgs',
        tol=1e-8,
        max_iter=1000,
        class_weight="balanced",
        random_state=57,
        n_jobs=1),
    # 'SVM': SVC(
    #     C=1e-2,
    #     kernel='rbf',
    #     gamma="scale",
    #     decision_function_shape="ovr",
    #     probability=True,
    #     class_weight="balanced",
    #     verbose=True,
    #     random_state=57),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(200, 100),
        activation="tanh",
        solver="sgd",
        learning_rate="invscaling",
        learning_rate_init=0.01,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=57),
    'Random Forest': RandomForestClassifier(
        n_estimators=1000,
        criterion="entropy",
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=57,
        n_jobs=-1)}

param_grids = {
    'Logistic Regression': {'C': [10 ** value for value in range(0, -11, -1)],
                            'solver': ['liblinear', 'newton-cg', 'lbfgs'], 'max_iter': [1e4, ],
                            'tol': [1e-8, ], 'class_weight': ['balanced', ], 'n_jobs': [1, ]},
    'Neural Network': [{'hidden_layer_sizes': [(value,) for value in (10, 25, 50, 100, 250)],
                        'activation': ['tanh', ], 'solver': ['adam'],
                        'alpha': [10 ** value for value in range(0, -12, -2)],
                        'learning_rate': ['constant', ],
                        'learning_rate_init': [1e-2, ],
                        'max_iter': [1000, ],
                        'tol': [1e-8, ]},
                       {'hidden_layer_sizes': [(value,) for value in (10, 25, 50, 100, 250)],
                        'activation': ['tanh', ], 'solver': ['sgd'],
                        'alpha': [10 ** value for value in range(0, -12, -2)],
                        'learning_rate': ['constant', 'invscaling', 'adaptive'],
                        'learning_rate_init': [1e-2, ],
                        'max_iter': [1000, ],
                        'tol': [1e-8, ]}
                       ],
    'Random Forest': {'n_estimators': (10, 25, 50, 100, 250, 500, 1000), 'criterion': ['gini', 'entropy'],
                      'max_depth': [None, 10, 25, 50, 100], 'min_samples_split': (2, 3, 5, 10),
                      'min_samples_leaf': (2, 3, 5, 10), 'n_jobs': [-1, ]}
}


def get_dataset_tuples_for_validation(dataset, labels, n=10):
    dataset_tuples = []
    for i in range(n):
        dataset_tuples.append(tuple(train_test_split(dataset, labels, shuffle=True, stratify=labels, test_size=0.3)))
    return dataset_tuples


def score_classifier(predicted_labels, predicted_label_probabilities, test_labels):
    scores = {"Accuracy": accuracy_score(test_labels, predicted_labels),
              "Precision": precision_score(test_labels, predicted_labels, average='weighted'),
              "Recall": recall_score(test_labels, predicted_labels, average='weighted'),
              "F1": f1_score(test_labels, predicted_labels, average='weighted'),
              "ROC AUC": roc_auc_score(test_labels, predicted_label_probabilities[:, 1]),
              "Cross-entropy loss": log_loss(test_labels, predicted_label_probabilities)}
    return scores


def train_model(dataset_tuple, optimize_parameters=False):
    train_set, train_labels, test_set, test_labels = dataset_tuple

    scores = {}
    scoring = 'neg_log_loss'
    trained_classifiers = {}

    if optimize_parameters:
        for classifier_name, classifier_model in classifiers.items():
            print('Optimizing and training ' + classifier_name + '...')
            classifier_grid_search = GridSearchCV(classifier_model, param_grids[classifier_name],
                                                  scoring=scoring, cv=3, refit=True, verbose=1)
            classifier_grid_search.fit(train_set, train_labels)
            print('Best found parameter combination: ' + str(classifier_grid_search.best_params_))
            best_classifier = classifier_grid_search.best_estimator_
            trained_classifiers[classifier_name] = best_classifier
            print('Scoring ' + classifier_name + '...')
            predicted_labels = best_classifier.predict(test_set)
            predicted_probs = best_classifier.predict_proba(test_set)
            scores[classifier_name] = score_classifier(predicted_labels, predicted_probs, test_labels)

    else:
        for classifier_name, classifier_model in classifiers.items():
            print('Training and scoring ' + classifier_name + '...')
            classifier_model.fit(train_set, train_labels)
            trained_classifiers[classifier_name] = classifier_model
            predicted_labels = classifier_model.predict(test_set)
            predicted_probs = classifier_model.predict_proba(test_set)
            scores[classifier_name] = score_classifier(predicted_labels, predicted_probs, test_labels)

    best_model_name, _ = min(scores.items(), key=lambda x: x[1]["Cross-entropy loss"])
    best_model = trained_classifiers[best_model_name]

    return best_model, scores


def scores_to_table(scores):
    scores_items = list(scores.items())
    scores_metrics = list(scores_items[0][1].keys())
    scores_classifiers = list(scores.keys())
    scores_values = [list(metrics.values()) for classifier, metrics in scores_items]
    scores_table = pd.DataFrame(np.array(scores_values, dtype='float'), index=scores_classifiers,
                                columns=scores_metrics)
    return scores_table
