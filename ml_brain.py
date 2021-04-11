from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import sys
sys.path.append('.')

import pandas as pd
from sklearn.model_selection import KFold

from tqdm import tqdm

#keras.backend.clear_session()
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.utils import resample
import copy
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

import numpy as np
import pandas as pd
import pickle
import traceback
import nltk

from scipy.sparse import issparse
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ARDRegression, LinearRegression

# Scoring
from sklearn.metrics import make_scorer, f1_score, homogeneity_score, mean_squared_error

def supervised_classifier_defs(setting):
    '''Esablish supervised classifier definitions which are non-specific to embeddor,
       and therefore, non-specific to the NLP application
    '''
    classes = {
            'LinearSVC': LinearSVC(),
            'SVC': SVC(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'MLPClassifier': MLPClassifier(),
            'LogisticRegression': LogisticRegression(),
            'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),
            'RidgeClassifier': RidgeClassifier(),
            'SGDClassifier': SGDClassifier(),
            'ExtraTreesClassifier': ExtraTreesClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'BaggingClassifier': BaggingClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),
    }

    if setting == 'normal':
      search_space = {
                        'LinearSVC': {
                        'clf__C': [1e-2,1e-1],
                        'clf__max_iter':[800,1000],
                        },
                        'SVC': {
                        'clf__C':[1.],
                        'clf__gamma': [0.5, 0.1, 0.01],
                        'clf__kernel': ['rbf']
                        },
                        'DecisionTreeClassifier': {
                        'clf__criterion':['gini'],
                        'clf__splitter': ['best'],
                        'clf__min_samples_split': [2],
                        'clf__min_samples_leaf': [1]
                        },
                        'MLPClassifier': {
                        'clf__hidden_layer_sizes': [(100,)],
                        'clf__solver': ['adam'],
                        'clf__alpha': [1e-2],
                        'clf__batch_size': ['auto'],
                        'clf__learning_rate': ['adaptive'],
                        'clf__max_iter': [1000]
                        },
                        'LogisticRegression': {
                        'clf__solver': ['newton-cg', 'lbfgs', 'sag'],
                        'clf__C': np.logspace(0,4,10)
                        },
                        'PassiveAggressiveClassifier': {
                        'clf__C': [0., 0.01, 0.1, 1.],
                        'clf__loss': ['hinge', 'squared_hinge'],
                        },
                        'RidgeClassifier': {
                        'clf__alpha': [0.,1e-3,1.],
                        'clf__normalize': [False,True]
                        },
                        'SGDClassifier': {
                        'clf__loss': ['squared_hinge'],
                        'clf__alpha': [1e-3,1e-2],
                        },
                        'ExtraTreesClassifier': {
                        'clf__n_estimators': [200,500],
                        'clf__criterion':['gini'],
                        'clf__min_samples_split': [2],
                        'clf__min_samples_leaf': [1]
                        },
                        'RandomForestClassifier': {
                        'clf__n_estimators': [200,500],
                        'clf__criterion':['gini'],
                        'clf__min_samples_split': [2],
                        'clf__min_samples_leaf': [1],
                        },
                        'BaggingClassifier': {
                        'clf__n_estimators': [30,50,100],
                        'clf__max_samples':[1.0,0.8],
                        },
                        'AdaBoostClassifier': {
                        'clf__n_estimators': [50,100],
                        'clf__learning_rate':[1.,0.9,0.8],
                        'clf__algorithm': ['SAMME.R']
                        }
                    }
    if setting == 'detailed':
      search_space = {
                        'LinearSVC': {
                        'clf__C': [1e-2,1e-1,1,1e1,1e2,1e3],
                        'clf__max_iter':[800,1000,1200,1500,2000],
                        },
                        'SVC': {
                        'clf__C':[1.,1e-2,1e-1,1,1e1],
                        'clf__gamma': [0.5, 0.1, 0.01, 0.001, 0.0001],
                        'clf__kernel': ['rbf', 'linear', 'sigmoid', 'poly'],
                        },
                        'DecisionTreeClassifier': {
                        'clf__criterion':['gini','entropy'], 
                        'clf__splitter': ['best','random'],
                        'clf__min_samples_split': [2,3,4],
                        'clf__min_samples_leaf': [1,2,3],
                        },
                        'MLPClassifier': {
                        'clf__hidden_layer_sizes': [(100,),(100,64),(100,64,16)],
                        'clf__solver': ['adam','lbfgs', 'sgd', 'adam'],
                        'clf__alpha': [1e-2,1e-3],
                        'clf__batch_size': ['auto'],
                        'clf__learning_rate': ['adaptive', 'invscaling', 'constant'],
                        'clf__max_iter': [1000]
                        },
                        'LogisticRegression': {
                        'clf__solver': ['newton-cg', 'lbfgs', 'sag'],
                        'clf__C': np.logspace(0,4,10)
                        },
                        'PassiveAggressiveClassifier': {
                        'clf__C': [0., 0.01, 0.1, 1.],
                        'clf__loss': ['hinge', 'squared_hinge'],
                        },
                        'RidgeClassifier': {
                        'clf__alpha': [0.,1e-3,1.,1e-4,1e-3,1e-2,1e-1,1.],
                        'clf__normalize': [False,True]
                        },
                        'SGDClassifier': {
                        'clf__loss': ['squared_hinge', 'hinge', 'log'],
                        'clf__alpha': [1e-3,1e-2],
                        },
                        'ExtraTreesClassifier': {
                        'clf__n_estimators': [200,500],
                        'clf__criterion':['gini','entropy'], 
                        'clf__min_samples_split': [2,3,4],
                        'clf__min_samples_leaf': [1,2,3]
                        },
                        'RandomForestClassifier': {
                        'clf__n_estimators': [200,500],
                        'clf__criterion':['gini','entropy'], 
                        'clf__min_samples_split': [2,3,4],
                        'clf__min_samples_leaf': [1,2,3]
                        },
                        'BaggingClassifier': {
                        'clf__n_estimators': [10,30,50,100,200],
                        'clf__max_samples':[1.,0.8,0.4,0.2], 
                        },
                        'AdaBoostClassifier': {
                        'clf__n_estimators': [30,50,100,150,300],
                        'clf__learning_rate':[1.0,0.9,0.8,0.4],
                        'clf__algorithm': ['SAMME.R', 'SAMME']
                        }
                    }

    return search_space, classes

def supervised_regressor_defs(setting):
    classes = {
            'SVR': SVR(),
            'LinearSVR': LinearSVR(),
            'ARDRegression': ARDRegression(),
            'LinearRegression': LinearRegression()
    }

    search_space = {
                    'SVR': {
                        'clf__kernel': ['poly', 'rbf'],
                        'clf__degree':[2,3],
                    },
                    'LinearSVR': {
                    },
                    'ARDRegression': {
                    },
                    'LinearRegression': {
                    }
    }
    return search_space, classes

def classification_deployer(X,y, n_splits, classifiers, search_space, pipeline_steps, scoring, verbose=3, greater_is_better=True):

    rows = []

    if issparse(X):
        print('Converting passed data to dense array...')
        X =  X.toarray()

    # get position of 'clf' in pipeline_steps
    idx_clf_pipeline = [i for i,it in enumerate(pipeline_steps) if it[0]=='clf'][0]

    best_gs_instance = None

    if greater_is_better:
        best_model_score = 0.0
    else:
        best_model_score = 9e20 # not good code, michael...

    for iter_idx, key in enumerate(classifiers.keys()):
        clas = classifiers[key]
        space = search_space[key]

        iter_pipeline_steps = copy.deepcopy(pipeline_steps)
        iter_pipeline_steps[idx_clf_pipeline] = ('clf',clas)
        pipe = Pipeline(iter_pipeline_steps)

        gs_clf = GridSearchCV(pipe, space, scoring = scoring, cv = n_splits, 
                                    n_jobs = 1, return_train_score = True, verbose = verbose)
        gs_clf.fit(X, y)
        params = gs_clf.cv_results_['params']
        scores = []
        for i in range(n_splits):
            r1 = gs_clf.cv_results_[f"split{i}_test_score"]
            scores.append(r1.reshape(len(params),1))

        r2 = gs_clf.cv_results_["mean_fit_time"]
        
        all_scores = np.hstack(scores)
        for param, score, time in zip(params, all_scores, r2):
            param['mean_fit_time'] = time
            d = {
                'estimator': key,
                'min_score': min(score),
                'max_score': max(score),
                'mean_score': np.mean(score),
                'std_score': np.std(score),
                }
            rows.append((pd.Series({**param,**d})))

        if greater_is_better:
            logic = gs_clf.best_score_ > best_model_score
        else:
            logic = gs_clf.best_score_ < best_model_score
            print(logic, f"{gs_clf.best_score_} < {best_model_score}")

        if logic:
            print('Better score ({:.3f}) found on classifier: {}'.format(gs_clf.best_score_,key))
            best_model_score = gs_clf.best_score_
            best_gs_instance = gs_clf

    return pd.concat(rows, axis=1).T, best_gs_instance.best_estimator_

class ConfidenceInferencer:
    def __init__(self):
        self.train_ys = None
        self.test_ys = None
        self.train_Xs = None
        self.test_Xs = None
    
    def prep_data(self, Xs, ys, badges=None, dates=None, random_state=0, type_ML = 'classifier'):
        self.type_ML = type_ML
        if type_ML == 'classifier':
            if badges == None:
                self.train_ys, self.test_ys, self.train_Xs, self.test_Xs = train_test_split(ys, Xs, badges, train_size = 0.9, shuffle=True, stratify=ys, random_state=random_state)
            else:
                self.train_ys, self.test_ys, self.train_Xs, self.test_Xs, self.train_badges, self.test_badges, self.train_dates, self.test_dates = train_test_split(ys, Xs, badges, dates, train_size = 0.9, shuffle=True, stratify=ys, random_state=random_state)

        elif type_ML == 'regressor':
            self.train_ys = ys
            self.train_Xs = Xs


    def _classify(self,
                    embedding,
                    pipeline_steps, 
                    scoring, 
                    search_space, 
                    classes,
                    n_cv_splits = 5,
                    verbose=0,
        ):

        try:
            X = self.train_Xs
            y = self.train_ys

            results_df, best_model = classification_deployer(
                                                            X,y, 
                                                            n_cv_splits, 
                                                            classes, 
                                                            search_space, 
                                                            pipeline_steps, 
                                                            scoring,
                                                            verbose=verbose,
                                                            greater_is_better=self.greater_is_better
                                                                )
            # organize columns
            cols = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score', 'mean_fit_time']
            cols += [c for c in results_df.columns if c not in cols]
            results_df = results_df[cols]
            # sort values
            if self.greater_is_better:
                order_ascending = False
            else:
                order_ascending = True
            results_df = results_df.sort_values(['mean_score'], ascending=order_ascending)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
        return results_df, best_model

    def classify_supervised(self, embedding='tfidf', n_cv_splits = 5, subset_example_classifiers = None, setting = 'normal', user_defined_classes = None, user_defined_search_space = None):

        pipeline_steps = [('clf', None)]

        if self.type_ML == 'classifier':
            scoring = make_scorer(f1_score, average = 'weighted')
            self.greater_is_better = True
        else:
            scoring = make_scorer(mean_squared_error, squared=True)
            self.greater_is_better = False

        if user_defined_classes == None or user_defined_search_space == None:
            # Utilize a subset of the pre-determined classifiers

            if self.type_ML == 'classifier':
                search_space, classes = supervised_classifier_defs(setting)

            elif self.type_ML == 'regressor':
                search_space, classes = supervised_regressor_defs(setting)

            if subset_example_classifiers != None:
                for clf_str in subset_example_classifiers:
                    if clf_str not in classes:
                        del classes[clf_str]
                        del search_space[clf_str]
                    else:
                        raise Exception("All components of subset_example_classifiers must be keys in ")
        else:
            search_space = user_defined_search_space
            classes = user_defined_classes

        self.supervised_results, self.supervised_best_model =  self._classify(
                                                                        embedding,
                                                                        pipeline_steps, 
                                                                        scoring, 
                                                                        search_space, 
                                                                        classes,
                                                                        n_cv_splits = n_cv_splits,
                                                                )
        return self.supervised_results, self.supervised_best_model

    def predict_best_model(self, eval_func = None, *eval_aargs):

        X = self.test_Xs
        y = self.test_ys

        print('Best algorithm found:\n',self.supervised_best_model)
        pred_y = self.supervised_best_model.predict(X)

        #pred_y = np.argmax(prob_y, axis=1)

        score = f1_score(y, pred_y, average = 'weighted')
        print(f"Score: {score}")
        
        return pred_y, self.test_ys, self.test_badges, self.test_dates #, prob_y

    def predict_best_regr_model(self, Xs):
        return self.supervised_best_model.predict(Xs)
