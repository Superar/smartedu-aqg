from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

results_path = Path('evaluation')
data = list()
for method in results_path.iterdir():
    for result in method.iterdir():
        method_name = method.name
        text_name = result.stem
        df = pd.read_json(result, orient='records')
        rouge = pd.json_normalize(df['ROUGE']).applymap(lambda x: x[1])
        df[['ROUGE-1', 'ROUGE-L']] = rouge
        df['Text'] = text_name
        df['Method'] = method_name
        data.append(df)

df = pd.concat(data)
df.index = df['Text'] + '_' + df.index.astype(str)
df = df.reset_index()

fig, ax = plt.subplots()
ax.set_ylabel('BLEU', c='b')
plt.xticks(rotation=45, ha='right')

ax2 = ax.twinx()
ax2.set_ylabel('ROUGE-L (Recall)', c='r')

groups = df.groupby('Method')
for idx, gp in groups:
    marker = 10 if idx == 'sim_bert' else 11
    ax.scatter(x=gp['index'], y=gp['BLEU'],
               c='b', marker=marker, alpha=0.6,
               label=idx)
    ax2.scatter(x=gp['index'], y=gp['ROUGE-L'],
                c='r', marker=marker, alpha=0.6,
                label=idx)
ax.legend()
plt.tight_layout()
plt.show()
