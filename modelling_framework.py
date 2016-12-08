
# coding: utf-8

# # Modelling Framework

# In[1]:

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
from sklearn.linear_model import LogisticRegressionCV as Log_Reg_CV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from scipy.io import mmread

get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot') 
from IPython.display import display, HTML


# # Start evaluating models

# In[2]:

def calc_expected_profit(profit_data_test, test_y_hat):
    interest_revenue = model_loan_term * profit_data_test.installment[test_y_hat == True].sum()
    recoveries = recoveries_avg * (test_y_hat == False).sum()
    principal_losses = profit_data_test.loan_amount[test_y_hat == False].sum()
    profit_mm = round(interest_revenue + recoveries - principal_losses) / float(10 ** 6)
    return profit_mm


# In[3]:

def ROC_plot(model, X, Y, model_name):
    # Plot the ROC curve for the given model
    roc_data = []
    # Note that the values actually start in the upper right and move to the lower left as p increases
    # so we need to initialize these at (1, 1) not at (0, 0) for the numeric intergation
    prev_false_positive = 1
    prev_true_positive = 1
    auc = 0  # rough integral
    predicted_prob = model.predict_proba(X)[:,1]

    # Draw ROC curve and use numeric integration to compute AUC
    for p in np.arange(0, 1, 0.01):
        yhat = predicted_prob >= p
        false_positive_rate = ((yhat == 1) & (Y == 0)).sum() * 1.0 / ((Y == 0).sum())
        true_positive_rate = ((yhat == 1) & (Y == 1)).sum() * 1.0 / ((Y == 1).sum())
        roc_data.append((false_positive_rate, true_positive_rate))
        # mark the key thresholds that we might use
        if p in (0.5, 0.6, 0.85):
            plt.scatter(false_positive_rate, true_positive_rate)  
        # Use midpoint rectangle method to approximate AUC
        auc += (true_positive_rate + prev_true_positive) / 2.0 * (prev_false_positive - false_positive_rate)
        prev_false_positive = false_positive_rate
        prev_true_positive = true_positive_rate


    # Close off the curve by ending at (0, 0) regardless of what the last point was
    roc_data.append((0, 0))
    # Use midpoint rectangle method to approximate AUC
    auc += (0 + prev_true_positive) / 2.0 * (prev_false_positive - 0)

    plt.plot([roc[0] for roc in roc_data],
             [roc[1] for roc in roc_data],
            )
    plt.xlim(-.01, 1.0)
    plt.ylim(0, 1.01)
    plt.title("ROC Curve for " + model_name)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.gca().text(0.99, 0.1, "AUC = %.3f" % (auc,),
                   ha = 'right', va = 'bottom'
    )
    plt.savefig('docs/images/roc_' + model_name.replace('/', '_') + '.png',
                bbox_inches='tight'
    )
    plt.show()

    return auc

def cross_terms(x):
    # compute cross terms -- but not two one-hots against each other, because memory
    x_out = x[x.columns]
    for c1, c2 in combinations(x.columns, 2):
        if '__' in c1 and '__' in c2:
            continue
        x_out[c1 + ' x ' + c2] = x[c1] * x[c2]
    return x_out

# In[4]:

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
        # expand column names for factors
        columns = [c for c in x.columns
                   if (c in columns
                       or c.split('__')[0] in columns)]
        x_local = x[columns]
        x_local_test = x_test[columns]
        
    if poly_degree == 2:
        x_local = cross_terms(x_local)
        x_local_test = cross_terms(x_local_test)
    elif poly_degree is not None:
        poly_xform = Preprocessing.PolynomialFeatures(degree=poly_degree, include_bias=False)
        x_local = pd.DataFrame(poly_xform.fit_transform(x_local))
        x_local_test = pd.DataFrame(poly_xform.fit_transform(x_local_test))
        
    indexes = range(len(years))
    np.random.shuffle(indexes)

    cm_accum = np.zeros((2, 2))
    f1_accum = 0
    score = 0
    precision = 0

    # k-fold cross-validation
    for i in range(k):
        train_indexes = list(indexes[0:len(indexes)*i/k]) + list(indexes[len(indexes)*(i+1)/k:])
        test_indexes = indexes[len(indexes)*i/k:len(indexes)*(i+1)/k]

        model = model_factory().fit(x_local.iloc[train_indexes,:], y.iloc[train_indexes])
        y_hat = model.predict(x_local)
        score += model.score(x_local.iloc[test_indexes], y.iloc[test_indexes]) / k
        y_hat_weighted = (model.predict_proba(x_local)[:,0] > prob_threshold)[test_indexes]
        precision += (y.iloc[test_indexes][y_hat_weighted]).mean() / k
        cm_accum += confusion_matrix(y.iloc[test_indexes], y_hat[test_indexes])
        f1_accum += f1_score(y.iloc[test_indexes], y_hat[test_indexes], pos_label = 1) / k

        if model_name is None:
            model_name = type(model).__name__

    # but also test against the x_test
    model = model_factory().fit(x_local, y)
    test_y_hat = (model.predict_proba(x_local_test)[:,0] > prob_threshold)
    test_score = model.score(x_local_test, y_test)
    test_precision = y_test[test_y_hat].mean()
    test_f1 = f1_score(y_test, test_y_hat, pos_label = 1)

    # expected profit
    profit_mm = calc_expected_profit(profit_data_test, test_y_hat)

    area_under_curve = ROC_plot(model, x_local_test, y_test, model_name)

    print "all   score: %.3f  baseline: %.3f   prec: %.3f   f1: %.3f  | test score %.3f  prec %.3f f1 %.3f  GP %dMM" % (
        score, y.mean(), precision, f1_accum, test_score, test_precision, test_f1, profit_mm)

    model_performance[model_name] = {
        'score': score,
        'baseline' : y.mean(),
        'prec' : precision,
        'f1': f1_accum,
        'test_score': test_score,
        'test_prec': test_precision,
        'test_f1': test_f1,
        'test_profit': profit_mm,
        'auc': area_under_curve,
    }

    return model

# In[ ]:

def eval_model_by_year(model_factory, 
                       columns = None, 
                       prob_threshold = 0.5,
                       x = x_expanded, 
                       x_test = x_test_expanded, 
                       y = y, 
                       y_test = y_test,
                       years = years, 
                       years_test = years_test,
                       profit_data_test = profit_data_test,
                       model_name = None):

    # Start with an overview
    all_years_model = eval_model_all_years(
                         model_factory, columns, None, prob_threshold, 
                         x, x_test, y, y_test, years, years_test,
                         profit_data_test, 
                         model_name = model_name)
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
        precision = 0

        # k-fold cross-validation
        for i in range(k):
            train_indexes = list(indexes[0:len(indexes)*i/k]) + list(indexes[len(indexes)*(i+1)/k:])
            test_indexes = indexes[len(indexes)*i/k:len(indexes)*(i+1)/k]
        
            model = model_factory().fit(x_local.iloc[train_indexes,:], y.iloc[train_indexes])
            y_hat = model.predict(x_local)
            score += model.score(x_local.iloc[test_indexes], y.iloc[test_indexes]) / k
            y_hat_weighted = (model.predict_proba(x_local)[:,0] > prob_threshold)[test_indexes]
            precision += (y.iloc[test_indexes][y_hat_weighted]).mean() / k
            cm_accum += confusion_matrix(y.iloc[test_indexes], y_hat[test_indexes])
            f1_accum += f1_score(y.iloc[test_indexes], y_hat[test_indexes], pos_label = 1) / k
        
        # but also test against the x_test
        test_score = model.score(x_local_test[years_test == yr], y_test[years_test == yr])
        test_y_hat = (model.predict_proba(x_local_test[years_test == yr])[:,0] > prob_threshold)
        test_precision = y_test[years_test == yr][test_y_hat].mean()

        print "%d  score: %.3f  baseline: %.3f   prec: %.3f   f1: %.3f  | test score %.3f  prec %.3f"  % (
            yr, score, y[years==yr].mean(), precision, f1_accum, test_score, test_precision)

    return all_years_model;
