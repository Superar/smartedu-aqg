import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from ...similarity import bert, tfidf
from .generate import create_model, generate_question

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f',
                    type=Path, required=True,
                    help='Input text from which the questions are going to be produced.')
parser.add_argument('--answers', '-a',
                    type=Path, required=True,
                    help='File with the answers for the questions to be generated. One answer per line.')
parser.add_argument('--similarity', '-s', type=str,
                    required=True, choices=['tfidf', 'bert'],
                    help='Similarity method to be used')
parser.add_argument('--output', '-o',
                    type=Path, required=True,
                    help='File to save the results')
parser.add_argument('--tfidf', '-t', type=Path,
                    help='Path to the directory containing documents for calculating IDF values.')
parser.add_argument('--model', '-m', type=str,
                    default='mrm8488/bert2bert_shared-portuguese-question-generation',
                    help='Fine-tuned BERT model for AQG.')
parser.add_argument('--simmodel', '-M', type=str,
                    default='ricardo-filho/bert-portuguese-cased-nli-assin-assin-2',
                    help='Fine-tuned BERT model for sentence similarity.')
args = parser.parse_args()

with args.file.open('rU', encoding='utf-8') as file_:
    text = [l.rstrip() for l in file_.readlines() if l.rstrip()]

with args.answers.open('rU', encoding='utf-8') as file_:
    answer_df = pd.read_json(file_, orient='records')
    answers = answer_df['Answer'].tolist()

if args.similarity == 'tfidf':
    context_indices = tfidf.create_context(args.tfidf, text, answers)
elif args.similarity == 'bert':
    context_indices = bert.create_context(text, answers, args.simmodel)

logger = list()
tokenizer, aqg = create_model(args.model)
for i, idx in enumerate(context_indices):
    context = [text[j] for j in idx]
    ctext = ' '.join(context)
    answer = answers[i]
    auto_question, _ = generate_question(ctext, answer,
                                         tokenizer, aqg,
                                         device)
    gold_question = answer_df.iloc[i]['Question']

    logger.append({'Gold Question': gold_question,
                   'Automatic Question': auto_question,
                   'Answer': answer,
                   'Context': context})
with args.output.open('w', encoding='utf-8') as file_:
    json.dump(logger, file_, indent=4, ensure_ascii=False)
