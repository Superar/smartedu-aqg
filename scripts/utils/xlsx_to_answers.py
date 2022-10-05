'''
Script to convert a XLSX file (with columns Number, Topic, Question, Answer)
to a TXT with one answer per line.
'''

import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f',
                    type=Path, required=True,
                    help='Input XLSX file.')
parser.add_argument('--output', '-o',
                    type=Path, required=True,
                    help='Output TXT file to be written.')
args = parser.parse_args()

df = pd.read_excel(args.file, header=1)

answer_cols = df.columns[df.columns.str.startswith('Correct Answer')]


def merge_cols(row):
    return [x for x in row[answer_cols].dropna()]


df['Answer'] = df.apply(merge_cols, axis='columns')
df = df.drop(answer_cols, axis='columns')
df = df.explode('Answer')
df = df[['Question', 'Answer']]
df.to_json(args.output, orient='records', indent=4, force_ascii=False)
