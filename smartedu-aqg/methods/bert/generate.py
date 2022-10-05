import torch
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, BertTokenizer,
                          EncoderDecoderModel)


def create_model(model_name: str) -> tuple[BertTokenizer, EncoderDecoderModel]:
    return (AutoTokenizer.from_pretrained(model_name, use_fast=False),
            AutoModelForSeq2SeqLM.from_pretrained(model_name))


def generate_question(text: str, answer: str,
                      tokenizer: BertTokenizer,
                      model: EncoderDecoderModel,
                      device: torch.device) -> tuple[str, str]:
    model.to(device)

    qg_input = f'answer: {answer} question: {text}'

    model.eval()
    encoded_input = tokenizer(
        qg_input,
        padding='max_length',
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model.generate(input_ids=encoded_input["input_ids"])

    question = tokenizer.decode(output[0], skip_special_tokens=True)
    return question, answer
