import sys
import torch
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import ExecutionTraceObserver
os.environ["TOKENIZERS_PARALLELISM"] = "false"
bsinput=int(sys.argv[1])
num_epochs = 1
num_iters = 46
listmodel=[]
bslist=[bsinput]
# Clear CUDA cache
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_rank = torch.cuda.current_device()
print(f"Starting on rank {_rank}.")

# Load a dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_dataset = dataset['train']
print("Dataset loaded.")
# namelist = ['meta-llama/Llama-3.2-1B']
namelist = ['NousResearch/Llama-3.2-1B']


for batch_size in bslist:
    for name in namelist:
        try:
            print("training model now:", name)
            # Load the model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(name)
            tokenizer = AutoTokenizer.from_pretrained(name)
            
            # Add padding token to the tokenizer if it doesn't have one
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
            model = model.to(device)
            # Tokenize the dataset
            def tokenize_function(examples):
                return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)

            tokenized_datasets = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
            tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            first_batch_subset = Subset(tokenized_datasets, list(range(batch_size)))
            train_dataloader = DataLoader(first_batch_subset, batch_size=batch_size, shuffle=False,num_workers=8)
            
            optimizer = optim.AdamW(model.parameters(), lr=5e-5)
            total_steps = len(train_dataloader) * num_epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

            model.train()
            print("batches num",len(train_dataloader))
            for epoch in range(num_epochs):
                for batch in train_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    total_time1 = 0
                    total_time2 = 0
                    for i in range(num_iters):
                        if 40<i<=50:
                            eg = ExecutionTraceObserver()
                            eg.register_callback("./llama_standard_profiler/graph_"+name.replace("/","-")+"-iter"+str(i)+".json")
                            eg.start()
                            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,with_stack=True,profile_memory=True) as prof:
                                torch.cuda.synchronize()
                                starter = torch.cuda.Event(enable_timing=True)
                                ender = torch.cuda.Event(enable_timing=True)
                                starter.record()
                                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
                                loss = outputs.loss
                                loss.backward()
                                optimizer.step()
                                scheduler.step()
                                optimizer.zero_grad()
                                ender.record()
                                torch.cuda.synchronize()
                                curr_time = starter.elapsed_time(ender)
                                total_time1 += curr_time
                            prof.export_chrome_trace("./llama_standard_profiler/profiler_"+name.replace("/","-")+"-iter"+str(i)+".json")
                            eg.stop()
                            eg.unregister_callback()
                            print("Save Exeution Trace")
                            print(name,"gpu time",curr_time)
                        elif 30<i<=40:
                            torch.cuda.synchronize()
                            starter = torch.cuda.Event(enable_timing=True)
                            ender = torch.cuda.Event(enable_timing=True)
                            starter.record()
                            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
                            loss = outputs.loss
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad() #check code here
                            
                            ender.record()
                            torch.cuda.synchronize()
                            curr_time = starter.elapsed_time(ender)
                            total_time2 += curr_time
                            print(name, "gpu time 2", curr_time)
                        else:
                            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
                            loss = outputs.loss
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            print(name,"iter",i,"done")
                    print("avg profiler Total time1:", total_time1/5)
                    print("avg Total time2:", total_time2/10)
                                
        except Exception as e:
            print(e)
            sys.exit(1)
            # pass