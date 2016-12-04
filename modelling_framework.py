
# coding: utf-8

# # Modelling Framework

# In[161]:

import itertools as it
import matplotlib
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import scipy as sp
import sklearn.preprocessing as Preprocessing
import datetime

from itertools import combinations
from sklearn.cross_validation import KFold as kfold
from sklearn.decomposition import TruncatedSVD as tSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression as Lin_Reg
from sklearn.linear_model import LogisticRegression as Log_Reg
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from scipy.io import mmread

get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot') 
from IPython.display import display, HTML


# # Start evaluating models

# In[ ]:

def calc_expected_profit(profit_data_test, test_y_hat):
    interest_revenue = model_loan_term * profit_data_test.installment[test_y_hat == True].sum()
    recoveries = recoveries_avg * (test_y_hat == False).sum()
    principal_losses = profit_data_test.loan_amount[test_y_hat == False].sum()
    profit_mm = round(interest_revenue + recoveries - principal_losses) / float(10 ** 6)
    return profit_mm


# In[2]:

model_performance = {}

def eval_model_all_years(model_factory, 
                         columns = None, 
                         poly_degree = None, 
                         prob_threshold = 0.5, 
                         x = x_expanded, 
                         x_test = x_test_expanded, 
                         y = y, 
                         y_test = y_test,
                         years = years, 
                         years_test = years_test, 
                         profit_data_test = profit_data_test,
                         model_name = None):
    k = 5
    np.random.seed(1729)
    
    if columns is None:
        x_local = x
        x_local_test = x_test
    else:
        x_local = x[columns]
        x_local_test = x_test[columns]
        
    if poly_degree is not None:
        poly_xform = Preprocessing.PolynomialFeatures(degree=poly_degree, include_bias=False)
        x_local = pd.DataFrame(poly_xform.fit_transform(x_local))
        x_local_test = pd.DataFrame(poly_xform.fit_transform(x_local_test))
        
    if True: # for yr in [2011, 2012, 2013]: # set(years.values):
        indexes = range(len(years))
        np.random.shuffle(indexes)

        cm_accum = np.zeros((2, 2))
        f1_accum = 0
        score = 0
        weighted_score = 0

        # k-fold cross-validation
        for i in range(k):
            train_indexes = list(indexes[0:len(indexes)*i/k]) + list(indexes[len(indexes)*(i+1)/k:])
            test_indexes = indexes[len(indexes)*i/k:len(indexes)*(i+1)/k]
        
            #print "TRAIN ", train_indexes
            #print 'TEST', test_indexes
            #print "Y", y.iloc[test_indexes]
            
            # model = model_factory().fit(x_expanded[years==yr], y[years==yr])
            # score = model.score(x_expanded[years==yr], y[years==yr]) / k
            model = model_factory().fit(x_local.iloc[train_indexes,:], y.iloc[train_indexes])
            y_hat = model.predict(x_local)
            score += model.score(x_local.iloc[test_indexes], y.iloc[test_indexes]) / k
            y_hat_weighted = (model.predict_proba(x_local)[:,0] > prob_threshold)[test_indexes]
            weighted_score += (y.iloc[test_indexes][y_hat_weighted]).mean() / k
            cm_accum += confusion_matrix(y.iloc[test_indexes], y_hat[test_indexes])
            f1_accum += f1_score(y.iloc[test_indexes], y_hat[test_indexes], pos_label = 1) / k

            if model_name is None:
                model_name = type(model).__name__
        
        # but also test against the x_test
        test_y_hat = (model.predict_proba(x_local_test)[:,0] > prob_threshold)
        test_score = (y_test == test_y_hat).mean()
        test_precision = 1- y_test[test_y_hat].mean()
        test_f1 = f1_score(y_test, test_y_hat, pos_label = 1)

        # expected profit
        profit_mm = calc_expected_profit(profit_data_test, test_y_hat)
        
        print "all   score: %.3f  baseline: %.3f   1-prec: %.3f   f1: %.3f  | test score %.3f  1-prec %.3f f1 %.3f  GP %dMM" % (
            score, 1-y.mean(), 1-weighted_score, f1_accum, test_score, test_precision, test_f1, profit_mm)

        model_performance[model_name] = {
            'score': score,
            'baseline' : 1-y.mean(),
            'prec' : 1-weighted_score,
            'f1': f1_accum,
            'test_score': test_score,
            'test_prec': test_precision,
            'test_f1': test_f1,
            'test_profit': profit_mm,
        }

# TODO: Confusion matrix (right now, we're not doing well enough to worry about that)


# In[194]:

def eval_model_by_year(model_factory, columns = None, prob_threshold = 0.5,
                       x = x_expanded, x_test = x_test_expanded, y = y, y_test = y_test,
                       years = years, years_test = years_test, profit_data_test = profit_data_test,
                       model_name = None):
    eval_model_all_years(model_factory, columns, None, prob_threshold, x, x_test, y, y_test, years, years_test,
                         profit_data_test, model_name = model_name)
    k = 5
    np.random.seed(1729)
    
    if columns is None:
        x_local = x
        x_local_test = x_test
    else:
        x_local = x[columns]
        x_local_test = x_test[columns]
        
    for yr in [2011, 2012, 2013]: # set(years.values):
        indexes = np.where(years == yr)[0]
        np.random.shuffle(indexes)

        cm_accum = np.zeros((2, 2))
        f1_accum = 0
        score = 0
        weighted_score = 0

        # k-fold cross-validation
        for i in range(k):
            train_indexes = list(indexes[0:len(indexes)*i/k]) + list(indexes[len(indexes)*(i+1)/k:])
            test_indexes = indexes[len(indexes)*i/k:len(indexes)*(i+1)/k]
        
            #print "TRAIN ", train_indexes
            #print 'TEST', test_indexes
            #print "Y", y.iloc[test_indexes]
            
            # model = model_factory().fit(x_expanded[years==yr], y[years==yr])
            # score = model.score(x_expanded[years==yr], y[years==yr]) / k
            model = model_factory().fit(x_local.iloc[train_indexes,:], y.iloc[train_indexes])
            y_hat = model.predict(x_local)
            score += model.score(x_local.iloc[test_indexes], y.iloc[test_indexes]) / k
            y_hat_weighted = (model.predict_proba(x_local)[:,0] > prob_threshold)[test_indexes]
            weighted_score += (y.iloc[test_indexes][y_hat_weighted]).mean() / k
            cm_accum += confusion_matrix(y.iloc[test_indexes], y_hat[test_indexes])
            f1_accum += f1_score(y.iloc[test_indexes], y_hat[test_indexes], pos_label = 1) / k
        
        # but also test against the x_test
        test_score = model.score(x_local_test[years_test == yr], y_test[years_test == yr])
        test_y_hat = (model.predict_proba(x_local_test[years_test == yr])[:,0] > prob_threshold)
        test_precision = 1- y_test[years_test == yr][test_y_hat].mean()

        print "%d  score: %.3f  baseline: %.3f   wscore: %.3f   f1: %.3f  | test score %.3f  1-prec %.3f"  % (
            yr, score, 1-y[years==yr].mean(), 1-weighted_score, f1_accum, test_score, test_precision)

# TODO: Confusion matrix (right now, we're not doing well enough to worry about that)
# TODO: Pretty-print
# TODO: Store results to allow side-by-side


# In[215]:

def eval_model_with_threshold(model_factory, columns=None):
    k = 5
    np.random.seed(1729)
    if columns is None:
        x_local = x_expanded
    else:
        x_local = x_expanded[columns]

    if True: # because old indent for loop
        indexes = range(len(y))
        np.random.shuffle(indexes)

        probs = np.ones_like(y) * -1

        for i in range(k):
            train_indexes = list(indexes[0:len(indexes)*i/k]) + list(indexes[len(indexes)*(i+1)/k:])
            test_indexes = indexes[len(indexes)*i/k:len(indexes)*(i+1)/k]
        
            model = model_factory().fit(x_local.iloc[train_indexes,:], y.iloc[train_indexes])
            probs_test = (model.predict_proba(x_local)[:,0]) #[test_indexes]
            probs = np.where([ii in test_indexes for ii in range(len(y))],  # slow but the only one I've found that works!
                             probs_test, probs)
            # print i, (probs == -1).sum(), (probs > 0).sum()
            
    thresholds = np.arange(0, 1, 0.05)
    plt.plot(thresholds,
             [1-y[probs > t].mean() for t in thresholds])
    plt.show()

    return probs

