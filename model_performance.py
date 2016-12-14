''' Render final scores of all models
'''

import json
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

model_performance = {}
for filename in ('model_performance_2.txt',
                 'model_performance_2a.txt',
                 'model_performance_2b.txt',
                 'model_performance_2c.txt',
                 'model_performance_2d.txt',
                 'model_performance_2e.txt',
                 'model_performance_2f.txt',
                 'model_performance_2g.txt',
                 'model_performance_2h.txt',
#                 'model_performance_2i.txt',
                 'model_performance_2j.txt',
                 'model_performance_2k.txt',
                 'model_performance_2l.txt',
             ):
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

print "html"
h = open('docs/model_performance_table.js', 'w')
for k, d in sorted(model_performance.items()):
    s = "<tr><th>%s</th>" % (k, )
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
    h.write('document.write("' + s + '")\n')
h.close()

model_performance_df = pd.DataFrame(model_performance).T
for col in ('auc', 'test_score', 'test_f1_inv', 'test_prec_inv'):
    print col
    base_value = baseline_values[col]
    plt.figure(figsize=(15, .25*len(model_performance_df)))
    plt.axvline(x=base_value, color='darkgrey')
    sorted_values = model_performance_df[col].fillna(0).sort_values()
    sorted_values.plot(kind='barh', 
                       color=[['darkblue', 'orange'][i] for i in list(sorted_values.apply(lambda x: round(x, 3)) > round(base_value, 3))],
                       width=0.85)
    plt.title(col.replace('_inv', ''))
    plt.savefig('docs/images/score_' + col.replace('_inv', '') + '.png',
                bbox_inches='tight')

    
