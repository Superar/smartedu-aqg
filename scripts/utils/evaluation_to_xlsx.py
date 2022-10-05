from pathlib import Path
import pandas as pd

results_path = Path('evaluation')
data = list()
for method in results_path.iterdir():
    if not method.is_dir():
        continue
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
df.drop(columns=['Gold Question', 'Automatic Question',
                 'Answer', 'Context', 'ROUGE'],
        inplace=True)
# df = df.reset_index()
df.to_excel(results_path / 'results.xlsx')
