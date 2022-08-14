from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
import torch
from datasets import load_dataset
import evaluate
import numpy as np

# #following along https://huggingface.co/course/chapter7/3?fw=pt

model_checkpoint = 'xlm-roberta-base' #"distilbert-base-uncased"
# # model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

# # distilbert_num_parameters = model.num_parameters() / 1_000_000
# # print(f"'>>> number of parameters: {round(distilbert_num_parameters)}M'")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# #text = "This is a great <mask>."


# #inputs = tokenizer(text, return_tensors="pt")
# # token_logits = model(**inputs).logits
# # # Find the location of [MASK] and extract its logits
# # mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
# # mask_token_logits = token_logits[0, mask_token_index, :]
# # # Pick the [MASK] candidates with the highest logits
# # top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

# # for token in top_5_tokens:
# #     print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

raw_datasets = load_dataset( "masakhaner", 'swa' )

# # sample = raw_datasets["train"].shuffle(seed=42).select(range(3))

# # for row in sample:
# #     print(f"\n'>>> ID: {row['id']}'")
# #     print(f"'>>> Tokens: {row['tokens']}'")
# #     print(f"'>>> ner_tags: {row['ner_tags']}'")

ner_feature = raw_datasets["train"].features["ner_tags"]
label_names = ner_feature.feature.names


# # words = raw_datasets["train"][0]["tokens"]
# # labels = raw_datasets["train"][0]["ner_tags"]
# # line1 = ""
# # line2 = ""
# # for word, label in zip(words, labels):
# #     full_label = label_names[label]
# #     max_length = max(len(word), len(full_label))
# #     line1 += word + " " * (max_length - len(word) + 1)
# #     line2 += full_label + " " * (max_length - len(full_label) + 1)

# # print(line1)
# # print(line2)


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

# inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)

# labels = raw_datasets["train"][0]["ner_tags"]
# word_ids = inputs.word_ids()
# print(labels)
# print(align_labels_with_tokens(labels, word_ids))


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

tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)


# #that was loading the datasets.  Now for the training.

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])

metric = evaluate.load("seqeval")

# # labels = raw_datasets["train"][0]["ner_tags"]
# # labels = [label_names[i] for i in labels]

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

id2label = {str(i): label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

args = TrainingArguments(
    "finetuned-xlm-r-masakhaner-swa-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=20,
    weight_decay=0.01,
    push_to_hub=True,
    save_total_limit=3,
    #per_device_train_batch_size=1,
    #gradient_accumulation_steps=4*2*2*2*2*2*2*2*2*2*2*2,
    #gradient_checkpointing=True, 
    #fp16=True,
)

#good?
#4*2*2*2*2*2*2*2*2*2*2,

#bad
#4*2*2*2*2*2*2*2*2*2,

#seemed to make one iteration
#4*2*2*2*2*2*2*2*2*2*2,

#4=734.00

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

from pynvml import *
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

print_gpu_utilization()

trainer.train()

pass
