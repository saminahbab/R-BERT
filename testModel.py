from model import BertForSequenceClassification

from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers import BertConfig

import torch

MODEL = "bert-large-uncased"
PATH = "./models/relation-weights/"
TOKENIZER_PATH = "./models/tokenizer/"

ADDITIONAL_SPECIAL_TOKENS = ["[E11]", "[E12]", "[E21]", "[E22]"]
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
CLS_TOKEN_SEGMENT_ID = 1
MAX_LEN = 128
SENTENCE = {
    "text": "[E11] Apple [E12] shares rose, in what was seen as a tumultuous day for the [E21] FTSE 250 [E22].",
    "e1": {"start": 0, "end": 5, "span": "Samin"},
    "e2": {"start": 35, "end": 45, "span": "AMICORP"},
}

TASK_NAME = "semeval"

SEMEVAL_RELATION_LABELS = [
    "Other",
    "Message-Topic(e1,e2)",
    "Message-Topic(e2,e1)",
    "Product-Producer(e1,e2)",
    "Product-Producer(e2,e1)",
    "Instrument-Agency(e1,e2)",
    "Instrument-Agency(e2,e1)",
    "Entity-Destination(e1,e2)",
    "Entity-Destination(e2,e1)",
    "Cause-Effect(e1,e2)",
    "Cause-Effect(e2,e1)",
    "Component-Whole(e1,e2)",
    "Component-Whole(e2,e1)",
    "Entity-Origin(e1,e2)",
    "Entity-Origin(e2,e1)",
    "Member-Collection(e1,e2)",
    "Member-Collection(e2,e1)",
    "Content-Container(e1,e2)",
    "Content-Container(e2,e1)",
]
LABEL_MAP = {label: i for i, label in enumerate(SEMEVAL_RELATION_LABELS)}


def get_model():

    bertconfig = BertConfig.from_pretrained(
        MODEL,
        num_labels=len(SEMEVAL_RELATION_LABELS),
        finetuning_task=TASK_NAME,
    )

    model = BertForSequenceClassification.from_pretrained(PATH)
    tokenizer = BertTokenizer.from_pretrained(
        TOKENIZER_PATH,
        do_lower_case=True,
        additional_special_tokens=ADDITIONAL_SPECIAL_TOKENS,
    )

    # had to do this during training.
    model.resize_token_embeddings(len(tokenizer))
    model.to("cpu")
    #     model.eval()
    return model, tokenizer


def format_input(tokenizer, sentence):

    tokens = tokenizer.tokenize(sentence)
    tokens += [SEP_TOKEN]
    segment_ids = [0] * len(tokens)

    tokens = [CLS_TOKEN] + tokens

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    if "[E22]" not in tokens or "[E12]" not in tokens:
        # represents a broken sentences with no complete entity pairs.
        print("Sentence supplied is not a proper sentence.")
        return None

    else:
        e11_p = tokens.index("[E11]") + 1
        e12_p = tokens.index("[E12]")

        e21_p = tokens.index("[E21]") + 1
        e22_p = tokens.index("[E22]")

    # assume you will send only one sentence at a time.
    input_mask = [1] * len(input_ids)

    padding_length = MAX_LEN - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)

    input_mask = input_mask + ([0] * padding_length)

    segment_ids = segment_ids + ([0] * padding_length)
    segment_ids = [1] + segment_ids

    e1_mask = [0 for i in range(len(input_mask))]
    e2_mask = [0 for i in range(len(input_mask))]

    for i in range(e11_p, e12_p):
        e1_mask[i] = 1
    for i in range(e21_p, e22_p):
        e2_mask[i] = 1

    assert len(input_ids) == MAX_LEN
    assert len(input_mask) == MAX_LEN
    assert len(segment_ids) == MAX_LEN

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    input_mask = torch.tensor([input_mask], dtype=torch.long)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long)
    e1_mask = torch.tensor([e1_mask], dtype=torch.long)
    e2_mask = torch.tensor([e2_mask], dtype=torch.long)

    # not really sure why we need to add labels in the input?
    labels = torch.tensor([1], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "token_type_ids": segment_ids,
        "labels": labels,
        "e1_mask": e1_mask,
        "e2_mask": e2_mask,
    }


def get_prediction(model, sentence):
    with torch.no_grad():
        output = model(**sentence)
        logits = output[-1]
        results = list(zip(logits[0].tolist(), SEMEVAL_RELATION_LABELS))
        for result in results:
            print(result)


def main():

    model, tokenizer = get_model()
    sentence = format_input(tokenizer, SENTENCE["text"])
    get_prediction(model, sentence)


if __name__ == "__main__":

    main()
