''' Render final scores of all models
'''

import json
import matplotlib.pyplot as plt
import os
import pandas as pd

plt.style.use('ggplot')

model_performance = {}
for filename in (os.listdir('.')):
    if filename.startswith('model_performance_2') and filename.endswith('.txt'):
        print "loading", filename
        for line in open(filename, 'r'):
            d = json.loads(line)
            model_performance[d['name']] = d['perf']
            

baseline_models = {
    'auc': 'LogReg_CV balanced',
    'test_f1_inv': 'Always 1',
    'test_score': 'Always 1',
    'test_prec_inv': 'Always 1',
}

baseline_values = {}
for k,v in baseline_models.items():
    baseline_values[k] = model_performance[v][k]

# get max value of each score per group
best_in_group = {}
for perf in model_performance.values():
    group_name = perf['group']
    if group_name in best_in_group:
        for k2 in ('auc', 'test_f1_inv', 'test_prec_inv', 'test_score',) :
            if perf[k2] > best_in_group[group_name][k2]:
                best_in_group[group_name][k2] = perf[k2]
    else:
        best_in_group[group_name] = {}
        for k2 in ('auc', 'test_f1_inv', 'test_prec_inv', 'test_score',) :
            best_in_group[group_name][k2] = perf[k2]

for perf in model_performance.values():
    group_name = perf['group']
    perf['is_best_in_group'] = False
    for k2 in ('auc', 'test_f1_inv', 'test_prec_inv', 'test_score',) :
        if perf[k2] == best_in_group[group_name][k2]:
            perf['is_best_in_group'] = True

print "html"
h = open('docs/model_performance_table.js', 'w')
h2 = open('docs/model_performance_table_unfiltered.js', 'w')
for k, d in sorted(model_performance.items()):
    s = "<tr><th>%s</th>" % (k, )
    is_best_in_group = False
    for k2 in ('auc', 'test_f1_inv', 'test_prec_inv', 'test_score',) :
        v = d[k2]
        
        if k2 in ('baseline', 'test_profit', 'elapsed', 'timestamp', 'model_group'):
            continue
        try:
            base_value = baseline_values[k2]
            base_value = round(base_value, 3)
        except:
            pass
            
        if k2 in ('auc', 'test_f1_inv', 'test_prec_inv', 'test_score') and round(v, 3) > base_value:
            s += "<td class='better'>%.3f</td>" % (v, )
        else:
            s += "<td>%.3f</td>" % (v, )
    s += "</tr>"
    
    if d['is_best_in_group']:
        h.write('document.write("' + s + '")\n')
    h2.write('document.write("' + s + '")\n')
h.close()
h2.close()

model_performance_df = pd.DataFrame(model_performance).T
for col in ('auc', 'test_score', 'test_f1_inv', 'test_prec_inv'):
    print col
    base_value = baseline_values[col]

    plt.figure(figsize=(15, .25*len(model_performance_df)))
    plt.axvline(x=base_value, color='darkgrey')
    sorted_values = model_performance_df[col].sort_values()
    sorted_values.plot(kind='barh', 
                       color=[['darkblue', 'orange'][i] for i in list(sorted_values.apply(lambda x: round(x, 3)) > round(base_value, 3))],
                       width=0.85)
    plt.title(col.replace('_inv', ''))
    plt.savefig('docs/images/score_' + col.replace('_inv', '') + '_all.png',
                bbox_inches='tight')

    plt.figure(figsize=(15, .25*sum(model_performance_df.is_best_in_group)))
    plt.axvline(x=base_value, color='darkgrey')
    sorted_values = model_performance_df[col][model_performance_df.is_best_in_group].sort_values()
    sorted_values.plot(kind='barh', 
                       color=[['darkblue', 'orange'][i] for i in list(sorted_values.apply(lambda x: round(x, 3)) > round(base_value, 3))],
                       width=0.85)
    plt.title(col.replace('_inv', ''))
    plt.savefig('docs/images/score_' + col.replace('_inv', '') + '.png',
                bbox_inches='tight')

    
