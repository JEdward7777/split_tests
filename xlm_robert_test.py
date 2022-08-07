from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from datasets import load_dataset
import torch
import evaluate
import numpy as np

model_checkpoint = './xlm-r-finetuned-ner/checkpoint-3'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


raw_datasets = load_dataset( "masakhaner", 'swa' )

ner_feature = raw_datasets["train"].features["ner_tags"]
label_names = ner_feature.feature.names

id2label = {str(i): label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

metric = evaluate.load("seqeval")
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)


model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

#see https://lewtun.github.io/blog/til/nlp/huggingface/transformers/2021/01/01/til-data-collator.html
# "Finally, we can calculate the loss per example with the following function:1"

for test_sequence in tokenized_datasets["test"]:
    input_ids = torch.tensor( [test_sequence["input_ids"]] )
    attention_mask = torch.tensor( [test_sequence["attention_mask"]] )



    with torch.no_grad():
        output = model(input_ids, attention_mask)
        #predicted_label = torch.argmax(output.logits,axis=2)
        pass

    metrics = compute_metrics( (output.logits,[test_sequence["labels"]]))

    print( metrics )
    pass

    # result = model(**tokenized_datasets)
    # #result = model(raw_datasets["test"])

    # metrics = compute_metrics(result)

    # print( metrics )