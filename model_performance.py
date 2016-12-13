''' Render final scores of all models
'''

import json
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

model_performance = {}
for line in open('model_performance.txt', 'r'):
    d = json.loads(line)
    model_performance[d['name']] = d['perf']

baseline_values = model_performance['LogReg_CV balanced']

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
model_performance_df.fillna(0)
for col in model_performance_df.columns:
    print col
    plt.figure(figsize=(15, .25*len(model_performance_df)))
    model_performance_df[col].sort_values().plot(kind='barh', width=0.85)
    plt.title(col)
    plt.savefig('docs/images/score_' + col + '.png',
                bbox_inches='tight')

    
