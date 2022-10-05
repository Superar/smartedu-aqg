import argparse
import json
from pathlib import Path

from .bleu import score as bleu_score
from .rouge import score as rouge_score


parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f',
                    type=Path, required=True,
                    help='JSON file containing both gold and generated questions. ' +
                    'Follows the same format created by the smartedu-aqg.generate module.')
parser.add_argument('--output', '-o',
                    type=Path, required=True,
                    help='Path for the output file to be saved. Will be saved as a JSON ' +
                    'with the same information as `file`, but with the additional evaluation metrics.')
args = parser.parse_args()

with args.file.open('rU', encoding='utf-8') as file_:
    data = json.load(file_)

for instance in data:
    instance['BLEU'] = bleu_score(instance['Gold Question'],
                                  instance['Automatic Question'])
    instance['ROUGE'] = rouge_score(instance['Gold Question'],
                                      instance['Automatic Question'])

with args.output.open('w', encoding='utf-8') as file_:
    json.dump(data, file_, ensure_ascii=False, indent=4)
