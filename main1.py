import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, AutoModel, GenerationConfig
from peft import LoraConfig, get_peft_model, TaskType
from utils import load_dataset, tokenize_dataset
from datasets import load_dataset
import time
from peft import PeftModel
import pandas as pd
import evaluate


# Loading config
with open("config.json") as f:
    config = json.load(f)

    model_name = "D:/D/db/try1"

#Loading Model
try:
    model = AutoModel.from_pretrained(model_name)
    print(f"modelLoadedSuccessfully!")
except Exception as e:
    print(f"ErrorLoadingModel: {e}")

dataset = load_dataset("csv", data_files={"train": "train.csv", "validation": "val.csv", "test": "test.csv"})

dataset

#This line creates an instance of a sequence-to-sequence model
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f'trainable model parameters: {trainable_model_params}\n \
            all model parameters: {all_model_params} \n \
            percentage of trainable model parameters: {(trainable_model_params / all_model_params) * 100} %'

print(print_number_of_trainable_model_parameters(original_model))

try:
    def tokenize_function(example):
        # Combine instruction and input into a prompt
        prompt = [inst + "\n\n" + inp for inst, inp in zip(example["instruction"], example["question"])]
    
        # Tokenize the input prompt
        example['input_ids'] = tokenizer(prompt, padding='max_length', truncation=True,
                                        return_tensors='pt').input_ids
    
        # Tokenize the expected output (target)
        example['labels'] = tokenizer(example['answer'], padding='max_length', truncation=True,
                                    return_tensors='pt').input_ids
        return example

    tokenize_datasets = dataset.map(tokenize_function, batched=True)
    tokenize_datasets = tokenize_datasets.remove_columns(['instruction', 'question', 'answer'])
    print(f"datasetProcessed")
except Exception as e:
    print(f"ErrorInPreprocessing: {e}")

output_dir = f'D:/D/db/try1/dialogue-summary-training-{str(int(time.time()))}'

tokenize_datasets = tokenize_datasets.filter(lambda exmaple, index: index % 100 == 0, with_indices=True)

lora_config = LoraConfig(r=32,
                         lora_alpha=32, ## LoRA Scaling factor 
                         target_modules=['q', 'v'], ## Inject LoRA adapters into specific submodules of the model.
                         lora_dropout = 0.05, #During training, randomly turn off 20% of the neurons in this layer each time the model sees data.
                         bias='none', #Specifies whether to train bias terms in the model.
                         task_type=TaskType.SEQ_2_SEQ_LM ## flan-t5 | Tells PEFT that we're fine-tuning a Sequence-to-Sequence Language Model
)
peft_model = get_peft_model(original_model, lora_config)

print(print_number_of_trainable_model_parameters(peft_model))

try:
    output_dir = f'D:/D/db/try1/dialogue-summary-training-{str(int(time.time()))}'
    ## this is we are again back to the hugging face trainer module
    peft_training_args = TrainingArguments(output_dir=output_dir,
                                           auto_find_batch_size=True,
                                           learning_rate=1e-3,
                                           num_train_epochs=1,
                                           logging_steps=1,
                                           max_steps=1,
                                           report_to='none' ## can be wandb, but we are reporint to noe
                    )

    ## this is same except we are using PEFT model instead of regular
    peft_trainer = Trainer(model=peft_model, 
                          args=peft_training_args,
                          train_dataset=tokenize_datasets['train']
                     )


    peft_trainer.train()

    peft_model_path = 'D:/D/db/try1/peft-dialogue-summary-checkpoint-local'

    peft_trainer.model.save_pretrained(peft_model_path)
    tokenizer.save_pretrained(peft_model_path)
    print(f"PEFT Adapter Trained")
except Exception as e:
    print(f"unable to train: {e}")

try:
    peft_model_base = AutoModelForSeq2SeqLM.from_pretrained('D:/D/db/try1', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained('D:/D/db/try1')

    peft_model = PeftModel.from_pretrained(peft_model_base, 
                                          'D:/D/db/try1/peft-dialogue-summary-checkpoint-local',
                                          torch_dtype=torch.bfloat16,
                                          is_trainable=False) ## is_trainable mean just a forward pass jsut to get a sumamry

    index = 200 ## randomly pick index
    question = dataset['test'][index]['question']
    human_baseline_summary = dataset['test'][index]['answer']

    prompt = f"""
    Answer the following question based on the context.

    {question}

    answer:
    """

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids

    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)


    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

    print(f'Human Baseline summary: \n{human_baseline_summary}\n')
    print(f'Original Model Output \n{original_model_text_output}\n')
    print(f'Peft Model Output \n{peft_model_text_output}\n')
except Exception as e:
    print(f"noInferencing: {e}")

try:
    question = dataset['test'][0:10]['question']
    human_baseline_summaries = dataset['test'][0:10]['answer']

    original_model_summaries = []
    peft_model_summaries = []

    for _, dialogue in enumerate(question):
        prompt = f"""
        Answer the following question based on the context. 

        {question}

        Answer: """

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids

        original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
        original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
        original_model_summaries.append(original_model_text_output)

        peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
        peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
        peft_model_summaries.append(peft_model_text_output)


    zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries,
                               peft_model_summaries))

    df = pd.DataFrame(zipped_summaries, columns=['human_baseline_summaries', 'original_model_summaries', 'peft_model_summaries'])
    df
    print(f"evaluated")
except Exception as e:
    print(f"ErrorInEvaluation: {e}")

rouge = evaluate.load('rouge')

original_model_results = rouge.compute(predictions=original_model_summaries, 
                                       references=human_baseline_summaries[0: len(original_model_summaries)],
                                      use_aggregator=True,
                                      use_stemmer=True)

peft_model_results = rouge.compute(predictions=peft_model_summaries, 
                                    references=human_baseline_summaries[0: len(peft_model_summaries)],
                                    use_aggregator=True,
                                    use_stemmer=True)

print(f'Original Model: \n{original_model_results}\n') 
print(f'PEFT Model: \n{peft_model_results}\n')
