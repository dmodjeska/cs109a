
# coding: utf-8

# # Data Loading
# David Modjeska and Andrew Greene

# This notebook isolates the code we use to load the data and do some last-minute pre-processing on it.

# In[27]:

# Deal with our necessary importds

import datetime
import itertools as it
import numpy as np
import os.path as op
import pandas as pd
import scipy as sp
import sklearn.preprocessing as Preprocessing

from itertools import combinations
from scipy.io import mmread
from sklearn.decomposition import TruncatedSVD as tSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from IPython.display import display, HTML


# In[28]:

### specify processed data files to generate - full/partial, partial %, and train/test
### Note: this cell is present in both notebooks

# load and clean full dataset?
load_full = True

# if not loading and cleaning full dataset, what sample percentage?
sample_percent = 10

if load_full:
    pct_str = ""
else: # not load_full
    pct_str = str(sample_percent) + "_pct"
    
### set intermediate file names
dir_str = "./intermediate_files/"

processed_data_file = dir_str + "processed_data_" + pct_str + ".json"

nlp_data_file = dir_str + "nlp_data_" + pct_str + ".json"
term_freqs_file = dir_str + "term_freqs_" + pct_str + ".mtx"
diff_terms_file = dir_str + "diff_terms_" + pct_str + ".json"


# In[29]:

### load processed data
data = pd.read_json(processed_data_file)
data_nlp = pd.read_json(nlp_data_file)
desc_matrix_coo = mmread(term_freqs_file)
desc_matrix = sp.sparse.csr_matrix(desc_matrix_coo)
count_cols_df = pd.read_json(diff_terms_file)

count_cols_bool = count_cols_df.values > 0.0


# In[30]:

print len(data) # Confirm that the number looks reasonable


# We want to do a little filtering here. As discussed in our writeup, we will limit ourselves to 36-month loans issued in the years 2011, 2012, and 2013

# In[31]:

model_loan_term = 36
data_filtered = data[data.loan_term == model_loan_term]
data_filtered = data_filtered[pd.to_datetime(data_filtered.issue_date).dt.year.isin([2011,2012,2013])]
print len(data_filtered)


# In[32]:

# Make sure we have a reasonable distribution of issue dates
pd.to_datetime(data_filtered.issue_date).dt.year.value_counts()


# A quick top-level sanity-check of the data, using Pandas `describe` method (and transposing it so it first on screen better)

# In[33]:

data_filtered.describe().T


# We need to make a couple of adjustments to columns

# In[34]:

# earliest_credit is not really a good indicator -- we want to know how long has elapsed since then
# See http://stackoverflow.com/questions/17414130/pandas-datetime-calculate-number-of-weeks-between-dates-in-two-columns
data_filtered['months_since_earliest_credit'] = (
    (pd.to_datetime(data_filtered.issue_date) - pd.to_datetime(data_filtered.earliest_credit))/np.timedelta64(1,'M')
).round()


# In[ ]:

# fix up empoyment title (which is "employer name" more often than not

def cleanup_emp_title(s):
    s = unicode(s).strip()
    if s == 'nan':
        return ''
    s = s.lower()
    s = s.replace('united states', 'us')
    s = s.replace(' llc', '')
    s = s.replace('.', '')
    s = s.replace(',', '')
    s = s.replace('-', '')
    if s.endswith(' inc'):
        s = s[:-4]
    s = s.replace(' ', '')
    if s == 'self':
        s = 'selfemployed'
    if s == 'usps':
        s = 'uspostalservice'
    if s == 'ups':
        s = 'unitedparcelservice'
    if s == 'usaf':
        s = 'usairforce'
    if s == 'rn':
        s = 'registerednurse'
    if s.endswith('bank'):
        s = s[:-4]
    if s.endswith('corp'):
        s = s[:-4]
    return s

data_filtered['emp_cleaned'] = data_filtered.employ_title.apply(cleanup_emp_title)


# In[35]:

# Separate the predictors (everything except "loan status") and the outcome
data_filtered_x = data_filtered.drop('loan_status', axis = 1)
data_filtered_y = data_filtered['loan_status']


# In[36]:

# copy unstandardized columns for later profit calculation
profit_data = data_filtered_x[['installment', 'loan_amount', 'recoveries', 'total_rec_int', 'total_rec_late_fee',
                              'total_rec_prncp']]
recoveries_avg = profit_data.recoveries.sum() / float(np.count_nonzero(profit_data.recoveries))


# Standardize the data. (See the comment below for a detailed explanation of what that means)

# In[37]:

# Certain columns in the raw data should not be in our model
columns_not_to_expand = [
    'description',        # free-text, so don't one-hot encode (NLP is separate)
    'employ_title',       # replaced by cleaned-up version
    'loan_subgrade',      # tainted predictor
    'id',                 # unique to each row
    'installment',        # tainted predictor
    'interest_rate',      # tainted predictor
    'index',              # unique to each row
    'ipr',                # tainted predictor
    'issue_date',         # not useful in future, using economic indicators instead
    'earliest_credit',    # has been converted to months_since_earliest_credit
    'recoveries',         # post hoc for profit calculation only
    'total_rec_int',      # post hoc for profit calculation only
    'total_rec_late_fee', # post hoc for profit calculation only
    'total_rec_prncp',    # post hoc for profit calculation only    
]


# In[38]:

# Given an input matrix X and the equivalent matrix X from the training set,
#
# (1) impute missing values (as "MISSING" for categorical, since the fact that 
# the value is missing may itself be significant; and using the median value
# for continuous predictors)
#
# (2) expand categorical predictors into a set of one-hot-encoded columns --
# using 0 and 1, and limiting ourselves to the 50 most common values in the
# training set, provided they have at least 10 instances (to prevent overfitting)
#
# (3) standardize continuous predictors using the mean and stdev of the
# training set

def expand_x(x, x_orig):
    x_expanded = pd.DataFrame()
    for colname in x_orig.columns:
        if colname in columns_not_to_expand:
            continue
        print colname, x_orig[colname].dtype
        if x_orig[colname].dtype == 'object':
            values = x[colname].fillna('MISSING')
            value_column_counts = x_orig[colname].fillna('MISSING').value_counts()
            value_columns = value_column_counts[value_column_counts > 10].index
            if len(value_columns) > 50:
                value_columns = value_columns[:50]
            for val in value_columns:
                x_expanded[colname + '__' + val.replace(' ', '_')] = (values == val).astype(int)
        else:
            values = x[colname].fillna(x[colname].median())
            sd = np.nanstd(x_orig[colname])
            if sd < 1e-10:
                sd = 1
            x_expanded[colname] = (values - np.nanmean(x_orig[colname]))/sd
    return x_expanded


# ### Split Data

# In[55]:

# Get a more manageable sample
np.random.seed(1729)
sample_flags = np.random.random(len(data_filtered)) <= 0.25
print "Indexes computed\n" 

# train set
x_train = data_filtered_x.iloc[sample_flags, :]
x_expanded = expand_x(x_train, x_train)
print "(Training set has %d rows)\n" % (len(x_expanded),)

# test set
x_test_expanded = expand_x(data_filtered_x.iloc[~sample_flags, :], x_train)
print "(Test set has %d rows)" % (len(x_test_expanded),)


# In[40]:

# split response column
y = data_filtered_y.iloc[sample_flags]
y_test = data_filtered_y.iloc[~sample_flags]


# In[41]:

# split profit data
profit_data_train = profit_data.iloc[sample_flags, :]
profit_data_test = profit_data.iloc[~sample_flags, :]


# In[42]:

### filter NLP data
filter_flags = data_nlp.loan_term.values == model_loan_term
data_nlp_filtered = data_nlp.iloc[filter_flags]

x_nlp_filtered = data_nlp_filtered.drop('loan_status', 1)
y_nlp_filtered = data_nlp_filtered.loan_status

desc_matrix_filtered = desc_matrix[filter_flags]
count_cols_bool_filtered = count_cols_bool[filter_flags]


# In[43]:

### split NLP data into training and testing sets
np.random.seed(1729)
train_flags = np.random.random(data_nlp_filtered.shape[0]) < 0.7

x_nlp_train = x_nlp_filtered.iloc[train_flags, :]
y_nlp_train = y_nlp_filtered.iloc[train_flags]

x_nlp_test = x_nlp_filtered.iloc[~train_flags, :]
y_nlp_test = y_nlp_filtered.iloc[~train_flags]

desc_matrix_train = pd.DataFrame(desc_matrix_filtered[train_flags, :].toarray())
desc_matrix_test = pd.DataFrame(desc_matrix_filtered[~train_flags, :].toarray())

count_cols_bool_train = pd.DataFrame(count_cols_bool_filtered[train_flags, :])
count_cols_bool_test = pd.DataFrame(count_cols_bool_filtered[~train_flags, :])

years_nlp = pd.to_datetime(x_nlp_train.issue_date).dt.year
years_nlp_test = pd.to_datetime(x_nlp_test.issue_date).dt.year


# In[44]:

### match indexes

desc_matrix_train.index = x_nlp_train.index
desc_matrix_test.index = x_nlp_test.index

count_cols_bool_train.index = x_nlp_train.index
count_cols_bool_test.index = x_nlp_test.index


# In[45]:

# inspect test proportion of good/bad loans
y_test.value_counts()


# In[46]:

# verify size of train set
np.count_nonzero(x_expanded.loan_amount)


# In[47]:

# be prepared to split stuff up by year of issue
years = pd.to_datetime(data_filtered_x.issue_date.iloc[sample_flags]).dt.year
years_test = pd.to_datetime(data_filtered_x.issue_date.iloc[~sample_flags]).dt.year


# ### Apply PCA to predictors

# In[48]:

tsvd = tSVD(n_components = 100, random_state=1729)
tsvd.fit(x_expanded)
data_filtered_expanded_x_pca = pd.DataFrame(tsvd.transform(expand_x(data_filtered_x, x_train)))
data_filtered_expanded_x_pca.index = data_filtered_x.index
pca_cum_var_expl = np.cumsum(np.round(tsvd.explained_variance_ratio_, 4) * 100)


# In[49]:

print "max variance explained", pca_cum_var_expl.max()
print "PCA: first and last columns where % variance explained >= 99:",             np.where(pca_cum_var_expl >= 99)[0][[0, -1]]

x_expanded_pca = data_filtered_expanded_x_pca.iloc[sample_flags, :73]
x_test_expanded_pca = data_filtered_expanded_x_pca.iloc[~sample_flags, :73]


# In[50]:

tsvd = tSVD(n_components = 500, random_state=1729)
desc_matrix_filtered_pca = pd.DataFrame(tsvd.fit_transform(desc_matrix_filtered))
desc_matrix_filtered_pca.index = x_nlp_filtered.index
pca_cum_var_expl = np.cumsum(np.round(tsvd.explained_variance_ratio_, 4) * 100)


# In[51]:

print "max variance explained", pca_cum_var_expl.max()
print "PCA: first and last columns where % variance explained >= 84:",             np.where(pca_cum_var_expl >= 84)[0][[0, -1]]


# In[52]:

desc_matrix_pca = desc_matrix_filtered_pca.iloc[train_flags, :]
desc_matrix_test_pca = desc_matrix_filtered_pca.iloc[~train_flags, :]


# In[ ]:



