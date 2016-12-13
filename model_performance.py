''' Render final scores of all models
'''

import json
import matplotlib.pyplot as plt
import pandas as pd

model_performance = {}
for line in open('model_performance.txt', 'r'):
    d = json.loads(line)
    model_performance[d['name']] = d['perf']

h = open('docs/model_performance.html', 'w')
for k, d in sorted(model_performance.items()):
    s = "<tr><th>%s</th>" % (k, )
    for k2, v in sorted(d.items()):
        s += "<td>%.3f</td>" % (v,)
    s += "</tr>\n"
    h.write(s)
h.close()

model_performance_df = pd.DataFrame(model_performance).T
for col in model_performance_df.columns:
    plt.figure(figsize=(15, .25*len(model_performance_df)))
    model_performance_df[col].sort_values().plot(kind='barh')
    plt.title(col)
    plt.savefig('docs/images/score_' + col + '.png',
                bbox_inches='tight')

    
