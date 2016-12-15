# Turns out we need to rerun everything at the last minute to compensate 
# for a bug in how we were computing precision

# Running it from the ipynb is not stable, is single-threaded, and is
# slower then running directly from the command line.

# So we pulled out the key pieces. Each block can be copied-and-pasted
# into one of several dozen instances of the Python interpreter (running
# in the Anaconda virtual environment) to take advantage of multiple
# cores on multiple computers.

### CORE 1

execfile('emergency-head.py')
logfile_name='model_performance_2a.txt'

for learning_rate in (0.1, 0.01):
    for n_est in (10, 50, 100, 200, 500):
        for max_depth in range(2, 10):
            eval_model_all_years(lambda: GBC(n_estimators = n_est, max_depth = max_depth, learning_rate = learning_rate),
                                 model_group='GBC',
                                 model_name='GBC %d/%d/%f' % (n_est, max_depth, learning_rate,))

# PCA
eval_model_all_years(lambda: Log_Reg(),
                     x = x_expanded_pca, x_test = x_test_expanded_pca,
                     model_name='PCA LogReg')

eval_model_all_years(lambda: Log_Reg(class_weight='balanced'),
                     x = x_expanded_pca, x_test = x_test_expanded_pca,
                     model_name='PCA LogReg balanced')

eval_model_all_years(lambda: QDA(),
                     x = x_expanded_pca, x_test = x_test_expanded_pca,
                     model_name='PCA QDA')

eval_model_all_years(lambda: GBC(n_estimators = 10, max_depth = 2, learning_rate = 0.1),
                     x = x_expanded_pca, x_test = x_test_expanded_pca,
                     model_name='PCA GBC 10/2/0.1')


### CORE 2

execfile('emergency-head.py')
logfile_name='model_performance_2b.txt'

from sklearn.ensemble import AdaBoostClassifier
for num_est, l_rate in ((50, 1.0), (100, 0.5), (200, 0.1), (1000, 0.01), (10000, 0.001)):
    eval_model_all_years(lambda: AdaBoostClassifier(random_state=1729, n_estimators=num_est, learning_rate = l_rate),
                         model_group='AdaBoost',
                         model_name = 'AdaBoost LR=%f num_est=%d' % (l_rate, num_est,))            


### CORE 3

execfile('emergency-head.py')
logfile_name='model_performance_2c.txt'
for max_depth in range(2, 10):
  for max_features in ['sqrt']: # + list(np.arange(0.1, 0.91, 0.5)):
    for class_weight in ('balanced',): # None
      eval_model_all_years(lambda: RFC(random_state=1729, 
                                     max_depth=max_depth, 
                                     max_features=max_features,
                                     class_weight = class_weight,
                                     n_estimators=200),
                           model_group='RFC',
                           model_name='RFC ' + (class_weight or 'unbalanced') + 
                                 ' ' + str(max_depth) + '/' + str(max_features) + '/200')  # TODO: other hyperparams

### CORE 4

execfile('emergency-head.py')
logfile_name='model_performance_2d.txt'
for max_depth in range(4, 10):
    for poly_degree in range(2, 5):
        eval_model_all_years(lambda: RFC(random_state=1729, 
                                     max_depth=max_depth, 
                                     class_weight = 'balanced',
                                     n_estimators=200),
                             x = cross_terms(x_expanded, poly_degree),
                             x_test = cross_terms(x_test_expanded, poly_degree),
                             model_name='RFC balanced ^' + str(poly_degree) + " to " 
                                 ' ' + str(max_depth) + '/200')      

### CORE 5

execfile('emergency-head.py')

class Stacking(object):
    
    def __init__(self, unfitted_combiner, component_type = 'mixed'):
        self.combiner = unfitted_combiner
        self.model_stack = None   
        
        stack = []
        
        # logistic regression component models with various class weights
        if component_type == 'log_reg':
            for diff in np.arange(0, 5, 0.5):
                weight_dict = { 0 : 1, 1 : (1 + diff) }
                stack.append([Log_Reg(C = 1, class_weight = weight_dict), 'Log_Reg ' + str(1 + diff)])
                
        # default - mixed component models
        else:
            stack.append((Log_Reg(C = 10 ** -5), 'Log_Reg'))
            stack.append((Log_Reg(C = 10 ** -5, class_weight = 'balanced'), 'Log_Reg balanced'))
            stack.append((LDA(shrinkage = 1, solver = 'eigen'), 'LDA'))
            stack.append((QDA(reg_param = 1), 'QDA'))
            stack.append((RFC(random_state=1729, n_estimators = 10, max_depth = 4), 'RF'))
            stack.append((RFC(random_state=1729, n_estimators = 300, max_depth = 5, class_weight = 'balanced'),
                        'RF balanced'))
            stack.append((GBC(n_estimators = 10, max_depth = 2, learning_rate = 0.1), 'Boost'))
            stack.append((DTC(max_depth = 10, max_features = 'log2', criterion = 'gini'), 'Tree 1'))
            stack.append((DTC(max_depth = 20, max_features = 'sqrt', criterion = 'entropy'), 'Tree 2'))
            stack.append((DTC(max_depth = 20, max_features = None, criterion = 'entropy'), 'Tree 3'))
            stack.append((DTC(max_depth = 30, max_features = None, criterion = 'gini'), 'Tree 4'))      
            #stack.append(SVC(class_weight = 'balanced')) # slow
                         
        self.model_stack = pd.DataFrame(stack, columns = ['Model_Params', 'Model'])
        self.num_models = len(self.model_stack)
    
    def _predict_component_models(self, X):
        n = X.shape[0]
        y_hat_stack = np.zeros((n, self.num_models))
        for index in range(self.num_models):
            y_hat_stack[:, index] = self.model_stack.iloc[index, 0].predict(X)
        return y_hat_stack
        
    def fit(self, X, y):
        for index in range(self.num_models):
            self.model_stack.iloc[index, 0].fit(X, y)
        y_hat_stack = self._predict_component_models(X)
        self.combiner = self.combiner.fit(y_hat_stack, y)
        return self
        
    def score(self, X, y):
        y_hat_stack = self._predict_component_models(X)
        score = self.combiner.score(y_hat_stack, y)
        return score
              
    def predict(self, X):
        y_hat_stack = self._predict_component_models(X)
        y_hat = self.combiner.predict(y_hat_stack)
        return y_hat
    
    def predict_proba(self, X):
        y_hat_stack = self._predict_component_models(X)
        y_hat_proba = self.combiner.predict_proba(y_hat_stack)
        return y_hat_proba
    
    def confusion_matrix(self, y, y_hat):
        return confusion_matrix(y, y_hat)
     
    def f1_score(self, y, y_hat):
        return f1_score(y, y_hat, pos_label = 1)
    
    def get_features(self):
        long_name = str(type(self.combiner))
        short_name = re.sub('.*\.', '', long_name)
        short_name = re.sub('\'>', '', short_name)
        
        index = index = self.model_stack.iloc[:, 1]
        if short_name == 'LogisticRegression':
            return pd.DataFrame(self.combiner.coef_.T, index = index)
        else:
            return pd.DataFrame(self.combiner.feature_importances_.T, index = index)        

def print_stacking_features(x_train, y_train, x_test, y_test, combiner, component_type = 'mixed'):
    model = Stacking(combiner, component_type)
    model.fit(x_train, y_train)
    
    features = model.get_features()
    features.columns = ['Importance']
    features['Abs'] = np.abs(features.Importance)
    features = features.sort_values('Abs', ascending = False).drop('Abs', axis = 1)
    display(features)

logfile_name='model_performance_2e.txt'

combiner = Log_Reg(C = 1, class_weight = 'balanced', penalty = 'l2', solver = 'liblinear')
eval_model_all_years(lambda: Stacking(combiner), model_name = "Stack LogReg balanced (mixed)")

combiner = Log_Reg(C = 1, class_weight = 'balanced', penalty = 'l2', solver = 'liblinear')
eval_model_all_years(lambda: Stacking(combiner, 'log_reg'), model_name = "Stack LogReg balanced (logreg)")

combiner = RFC(random_state=1729, n_estimators = 10, max_depth = 4, max_features = 'sqrt', 
               criterion = 'gini', class_weight = 'balanced')
eval_model_all_years(lambda: Stacking(combiner), model_name = "Stack RF balanced (mixed)")

combiner = RFC(random_state=1729, n_estimators = 10, max_depth = 4, max_features = 'sqrt', 
               criterion = 'gini', class_weight = 'balanced')
eval_model_all_years(lambda: Stacking(combiner, 'log_reg'), model_name = "Stack RF balanced (logreg)")

combiner = DTC(max_depth = 10, max_features = 'log2', class_weight = 'balanced', 
                           criterion = 'gini')
eval_model_all_years(lambda: Stacking(combiner), model_name = "Stack Tree balanced (mixed)")

combiner = DTC(max_depth = 10, max_features = 'log2', class_weight = 'balanced', 
                           criterion = 'gini')
eval_model_all_years(lambda: Stacking(combiner, 'log_reg'), model_name = "Stack Tree balanced (logreg)")

# SVC

execfile('emergency-head.py')
logfile_name='model_performance_2f.txt'

for gamma in (0.001, 0.01, 0.1):
    for C in 10 ** np.arange(-2.0, 2.1, 2.0):
        print C, gamma, datetime.datetime.now()
        eval_model_all_years(lambda: SVC(class_weight='balanced',
                                         probability=True,
                                         C=C, gamma=gamma),
                             model_group='SVC',
                             model_name="SVC C=" + str(C) + " g=" + str(gamma))


# More SVC - core g

execfile('emergency-head.py')
logfile_name='model_performance_2g.txt'

for gamma in (0.001, 0.01, 0.1):
    for C in 10 ** np.arange(-2.0, 2.1, 2.0):
        print C, gamma, datetime.datetime.now()
        eval_model_all_years(lambda: SVC(class_weight='balanced',
                                         probability=True,
                                         C=C, gamma=gamma),
                             model_group='SVC',
                             model_name="SVC C=" + str(C) + " g=" + str(gamma))

# SVC - core h
execfile('emergency-head.py')
logfile_name='model_performance_2h.txt'
        
for gamma in (0.0001,):
    for C in 10 ** np.arange(-2.0, 2.1, 2.0):
        print C, gamma, datetime.datetime.now()
        eval_model_all_years(lambda: SVC(class_weight='balanced',
                                         probability=True,
                                         C=C, gamma=gamma),
                             model_group='SVC',
                             model_name="SVC C=" + str(C) + " g=" + str(gamma))
        
# SVC - core i
execfile('emergency-head.py')
logfile_name='model_performance_2i.txt'
        
for gamma in (0.5,):
    for C in 10 ** np.arange(-2.0, 2.1, 2.0):
        print C, gamma, datetime.datetime.now()
        eval_model_all_years(lambda: SVC(class_weight='balanced',
                                         probability=True,
                                         C=C, gamma=gamma),
                             model_group='SVC',
                             model_name="SVC C=" + str(C) + " g=" + str(gamma))

# SVC - core j
execfile('emergency-head.py')
logfile_name='model_performance_2j.txt'
        
for gamma in (0.1,):
    for C in 10 ** np.arange(-2.0, 2.1, 2.0):
        print C, gamma, datetime.datetime.now()
        eval_model_all_years(lambda: SVC(class_weight='balanced',
                                         probability=True,
                                         C=C, gamma=gamma),
                             model_group='SVC',
                             model_name="SVC C=" + str(C) + " g=" + str(gamma))

# SVC - core k
execfile('emergency-head.py')
logfile_name='model_performance_2k.txt'
        
for gamma in (0.01,):
    for C in 10 ** np.arange(-2.0, 2.1, 2.0):
        print C, gamma, datetime.datetime.now()
        eval_model_all_years(lambda: SVC(class_weight='balanced',
                                         probability=True,
                                         C=C, gamma=gamma),
                             model_group='SVC',
                             model_name="SVC C=" + str(C) + " g=" + str(gamma))


# Expand hyperparam search on GBC
# Core l
execfile('emergency-head.py')
logfile_name='model_performance_2l.txt'

for learning_rate in (0.1,):
    for n_est in (1000, 5000):
        for max_depth in range(2, 6):
            eval_model_all_years(lambda: GBC(n_estimators = n_est, max_depth = max_depth, learning_rate = learning_rate),
                                 model_group='GBC',
                                 model_name='GBC %d/%d/%f' % (n_est, max_depth, learning_rate,))

# SVC core m
execfile('emergency-head.py')
logfile_name='model_performance_2m.txt'
        
for gamma in (1/120, 1/110, 1/130):
    for C in (1.0, 10.0, 100.0):
        print C, gamma, datetime.datetime.now()
        eval_model_all_years(lambda: SVC(class_weight='balanced',
                                         probability=True,
                                         C=C, gamma=gamma),
                             model_group='SVC',
                             model_name="SVC C=" + str(C) + " g=" + str(gamma))

# SVC core n
execfile('emergency-head.py')
logfile_name='model_performance_2n.txt'
        
for gamma in (1/110,):
    for C in (1.0, 10.0, 100.0):
        print C, gamma, datetime.datetime.now()
        eval_model_all_years(lambda: SVC(class_weight='balanced',
                                         probability=True,
                                         C=C, gamma=gamma),
                             model_group='SVC',
                             model_name="SVC C=" + str(C) + " g=" + str(gamma))

# SVC Core o
execfile('emergency-head.py')
logfile_name='model_performance_2o.txt'
        
for gamma in (1/130,):
    for C in (1.0, 10.0, 100.0):
        print C, gamma, datetime.datetime.now()
        eval_model_all_years(lambda: SVC(class_weight='balanced',
                                         probability=True,
                                         C=C, gamma=gamma),
                             model_group='SVC',
                             model_name="SVC C=" + str(C) + " g=" + str(gamma))

# SVC Core p
execfile('emergency-head.py')
logfile_name='model_performance_2p.txt'
        
for gamma in (10,):
    for C in (1.0, 10.0, 100.0):
        print C, gamma, datetime.datetime.now()
        eval_model_all_years(lambda: SVC(class_weight='balanced',
                                         probability=True,
                                         C=C, gamma=gamma),
                             model_group='SVC',
                             model_name="SVC C=" + str(C) + " g=" + str(gamma))

# SVC core q
execfile('emergency-head.py')
logfile_name='model_performance_2q.txt'
        
for gamma in (100,):
    for C in (10.0,):
        print C, gamma, datetime.datetime.now()
        eval_model_all_years(lambda: SVC(class_weight='balanced',
                                         probability=True,
                                         C=C, gamma=gamma),
                             model_group='SVC',
                             model_name="SVC C=" + str(C) + " g=" + str(gamma))
        
# SVC Core r
execfile('emergency-head.py')
logfile_name='model_performance_2r.txt'
        
for gamma in (100,):
    for C in (100.0,):
        print C, gamma, datetime.datetime.now()
        eval_model_all_years(lambda: SVC(class_weight='balanced',
                                         probability=True,
                                         C=C, gamma=gamma),
                             model_group='SVC',
                             model_name="SVC C=" + str(C) + " g=" + str(gamma))
        
# SVC core s
execfile('emergency-head.py')
logfile_name='model_performance_2s.txt'
        
for gamma in (.01,):
    for C in (100.0,):
        print C, gamma, datetime.datetime.now()
        eval_model_all_years(lambda: SVC(class_weight='balanced',
                                         probability=True,
                                         C=C, gamma=gamma),
                             model_group='SVC',
                             model_name="SVC C=" + str(C) + " g=" + str(gamma))

# SVC core t
execfile('emergency-head.py')
logfile_name='model_performance_2t.txt'
        
for gamma in (.01,):
    for C in (1.0,):
        print C, gamma, datetime.datetime.now()
        eval_model_all_years(lambda: SVC(class_weight='balanced',
                                         probability=True,
                                         C=C, gamma=gamma),
                             model_group='SVC',
                             model_name="SVC C=" + str(C) + " g=" + str(gamma))


# SVC core i got missed -- redo it here, broken into 3
execfile('emergency-head.py')
logfile_name='model_performance_2i.txt'
        
gamma=0.5
C=.01
print C, gamma, datetime.datetime.now()
eval_model_all_years(lambda: SVC(class_weight='balanced',
                                 probability=True,
                                 C=C, gamma=gamma),
                     model_group='SVC',
                     model_name="SVC C=" + str(C) + " g=" + str(gamma))


# SVC core i2
execfile('emergency-head.py')
logfile_name='model_performance_2i2.txt'
        
gamma=0.5
C=1
print C, gamma, datetime.datetime.now()
eval_model_all_years(lambda: SVC(class_weight='balanced',
                                 probability=True,
                                 C=C, gamma=gamma),
                     model_group='SVC',
                     model_name="SVC C=" + str(C) + " g=" + str(gamma))


# SVC core i3
execfile('emergency-head.py')
logfile_name='model_performance_2i3.txt'
        
gamma=0.5
C=100
print C, gamma, datetime.datetime.now()
eval_model_all_years(lambda: SVC(class_weight='balanced',
                                 probability=True,
                                 C=C, gamma=gamma),
                     model_group='SVC',
                     model_name="SVC C=" + str(C) + " g=" + str(gamma))


# Keep track of our status
'''
h: gamma=0.0001, C=10^-2 (10:48-13:09)
   gamma=0.0001, C=10^0  (13:09-15:36)
   gamma=0.0001, C=10^+2 (15:36-...)

i: gamma=0.5, C=10^-2 (10:52-13:47)
              C=10^0  (13:47-...)
              C=10^+2 (...)

g: gamma=0.001,  C=10^-2 (10:46-13:08)
   gamma=0.001,  C=10^0  (13:08-15:28)
   gamma=0.001,  C=10^+2 (15:28-...)
Can kill when: gamma=0.01,   C=10^-2 (...)

j: gamma=0.1,    C=10^-2 (...)
   gamma=0.1,    C=10^0  (...)
   gamma=0.1,    C=10^+2 (...) ==>t

k: gamma=0.01,    C=10^-2 (13:30-...)
   gamma=0.01,    C=10^0  (...)
   gamma=0.01,    C=10^+2 (...) ==>s

l: GBC (rate=0.1, est=1000,5000; max_depth 2-5)

m: gamma 1/120,  C=1 
   gamma 1/120,  C=10
   gamma 1/120,  C=100
then kill

n: gamma 1/110,  C=1
   gamma 1/110,  C=10
   gamma 1/110,  C=100

o: gamma 1/130,  C=1
   gamma 1/130,  C=10
   gamma 1/130,  C=100

p: gamma 10,  C=1
   gamma 10,  C=10
   gamma 10,  C=100

q: gamma 100, C=10

r: gamma 100, C=100

s: gamma .01, C=100

t: 0.01, 1

'''
