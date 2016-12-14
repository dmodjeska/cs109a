''' Render final scores of all models
'''

import json
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

print "loading"
model_performance = {}
for line in open('model_performance.txt', 'r'):
    d = json.loads(line)
    model_performance[d['name']] = d['perf']

baseline_values = model_performance['LogReg_CV balanced']

print "html"
h = open('docs/model_performance.html', 'w')
for k, d in sorted(model_performance.items()):
    s = "<tr><th>%s</th>" % (k, )
    for k2, v in sorted(d.items()):
        if k2 in ('baseline', 'test_profit'):
            continue
        base_value = round(baseline_values[k2], 3)
        if k2 == 'test_score':
            base_value = 0.848
            
        if k2 in ('auc', 'test_f1', 'test_prec', 'test_score') and v != 'nan' and round(v, 3) > base_value:
            s += "<td class='better'>%.3f</td>" % (v, )
        else:
            s += "<td>%.3f</td>" % (v, )
    s += "</tr>\n"
    h.write(s)
h.close()

model_performance_df = pd.DataFrame(model_performance).T
for col in model_performance_df.columns:
    print col
    base_value = baseline_values[col]
    if col == 'test_score':
        base_value = 0.848
    plt.figure(figsize=(15, .25*len(model_performance_df)))
    sorted_values = model_performance_df[col].fillna(0).sort_values()
    sorted_values.plot(kind='barh', 
                       color=['kr'[i] for i in list(sorted_values > base_value)],
                       width=0.85)
    plt.title(col)
    plt.savefig('docs/images/score_' + col + '.png',
                bbox_inches='tight')

    
