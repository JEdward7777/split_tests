from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
import sys

epoch_flag = False
epochs = 20
learning_rate_flag = False
learning_rate = 2e-5
model_checkpoint = 'xlm-roberta-base'
model_checkpoint_flag = False
just_tokenizer = False
save_total_limit = 3
save_total_limit_flag = False
push_checkpoint = "finetuned-xlm-r-masakhaner-swa-whole-word-phonetic-2"
push_checkpoint_flag = False
for arg in sys.argv[1:]:
    if epoch_flag:
        epochs = int(arg)
        print( f"epochs set to {epochs}")
        epoch_flag = False
    elif learning_rate_flag:
        learning_rate = float(arg)
        print( f"learning rate set to {learning_rate}")
        learning_rate_flag = False
    elif model_checkpoint_flag:
        model_checkpoint = arg
        model_checkpoint_flag = False
    elif save_total_limit_flag:
        save_total_limit = int(arg)
        save_total_limit_flag = False
    elif push_checkpoint_flag:
        push_checkpoint = arg
        push_checkpoint_flag = False
    elif arg == "--epoch":
        epoch_flag = True
    elif arg == "--just-tokenizer":
        just_tokenizer = True
    elif arg == "--learning-rate":
        learning_rate_flag = True
    elif arg == "--model-checkpoint":
        model_checkpoint_flag = True
    elif arg == "--save-total-limit":
        save_total_limit_flag = True
    elif arg == '--push-checkpoint':
        push_checkpoint_flag = True
    else:
        print( f"eh? {arg}" )


train_data_csv = 'swahili_cross_6000_trimmed.csv'

selected_column = 'allosaurus_transcript_spaced'

# #model_checkpoint = './xlm-r-finetuned-ner/checkpoint-3'
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False )

#https://huggingface.co/docs/datasets/loading#csv

phonetic_dataset = load_dataset("csv", data_files=train_data_csv)
phonetic_dataset = phonetic_dataset['train'].train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)


#collect the new tokens by just figuring out what the new words are.
new_tokens = []
for train_test in phonetic_dataset:
    for example in phonetic_dataset[train_test]:
        for word in example[selected_column].split( " " ):

            # #by messing around I discovered that adding a space on the front of the token helps it work the same as the
            # #natrual tokens otherwise it kind of gets interperated as a subword.
            # word = ' ' + word

            if word not in new_tokens:
                new_tokens.append(word)
            #     print( f"Adding{word}")
            # else:
            #     print( f"Dup{word}")
        #print(example[selected_column])

actual_added_tokens = tokenizer.add_tokens( new_tokens )


#also need to push tokenizer.
tokenizer.push_to_hub(push_checkpoint)

if not just_tokenizer:

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    def preprocess_function(examples):
        return tokenizer([s for s in examples[selected_column] if s is not None and len(s)], truncation=True)





    tokenized_phonetic_dataset = phonetic_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=phonetic_dataset["train"].column_names,
    )


    block_size = 128

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


    lm_dataset = tokenized_phonetic_dataset.map(group_texts, batched=True, num_proc=4)

    # pass

    #Load up the split data

    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    #resize to include new tokens.
    model.resize_token_embeddings(len(tokenizer))

    #train on downstream task of masked langauge modeling.
    #https://huggingface.co/docs/transformers/v4.21.1/en/tasks/language_modeling

    training_args = TrainingArguments(
        push_checkpoint,
        #output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        weight_decay=0.01,
        push_to_hub=True,
        save_total_limit=save_total_limit,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
    )

    train_results = trainer.train()
    # rest is optional but nice to have
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    #how to push https://huggingface.co/docs/transformers/model_sharing
    trainer.push_to_hub()