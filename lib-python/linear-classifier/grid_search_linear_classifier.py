from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
import numpy as np

def build_classifier(name):
    #Return a dictionary with clf name as key, and model plus grids (params) as values

    if name == 'logit':
        model = LogisticRegression(
            penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True,
            intercept_scaling=1.0, class_weight=None, random_state=None)
        model.grid_search = {'logit__C' : (0.1, 1, 5, 10)}
        model.grid_tuned = {'logit__C' : [(1)]}
        #
    elif name == 'sgd':
        model = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.0)
        model.grid_search = {'sgd__l1_ratio': (0, 0.25, 0.5, 1), 'sgd__alpha': (0.0001, 0.001, 0.01)}
        model.grid_tuned = {'sgd__l1_ratio': (1), 'sgd__alpha': (0.001)}
        #
    elif name == 'svc':
        model = SVC(kernel='linear', probability=True)
        model.grid_search = {'svc__C': [0.1, 1, 10, 100]}
        model.grid_tuned = {'svc__C': [1]}
        #
    elif name == 'rf':
        model = RandomForestClassifier(n_estimators=500, max_features=15, max_depth=10, min_samples_split=3, n_jobs=10, verbose=0)
        model.grid_search = {'rf__max_depth': [5, 10, 15], 'rf__max_features': [0.1, 0.5, 1.0], 'rf__min_samples_split': [3,6]}
        model.grid_tuned = {'rf__max_depth': [10], 'rf__max_features': [0.1], 'rf__min_samples_split': [6]}
        #
    elif name == 'gbc':
        model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.05, max_depth=15, min_samples_split=9, verbose=0, max_features=8)
        model.grid_search = {'gbc__max_depth': [5, 10], 'gbc__max_features': [0.1, 0.5], 'gbc__learning_rate': [0.01, 0.05], 'gbc__subsample': [0.5, 1]}
        model.grid_tuned = {'gbc__max_depth': [5], 'gbc__max_features': [0.1], 'gbc__learning_rate': [0.01], 'gbc__subsample': [0.5]}
        #
    elif name == 'adab':
        model = AdaBoostClassifier()
        n_estimators = [int(x) for x in np.linspace(start = 1, stop = 1000, num = 300)]
        learning_rate = [0.5, 0.75, 1.0, 2.0]
        model.grid_search = {'adab__n_estimators': n_estimators,
                        'adab__learning_rate': learning_rate}
        model.grid_tuned = {'adab__n_estimators': [500],
                            'adab__learning_rate': [0.75]}
    else : #default is nb
        model = MultinomialNB()
        model.grid_search = {'nb__alpha': [0.01, 0.05, 0.1, 0.4, 0.6, 0.8, 1.0, 1.5],
                        'nb__fit_prior': [False, True]}
        model.grid_tuned = {'nb__alpha': [0.1],
                            'nb__fit_prior': [False]}

    model.name = name
    return model