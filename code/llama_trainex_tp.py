import sys
import torch
import os
import time
import numpy as np
import tensor_parallel as tp
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import ExecutionTraceObserver
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import socket

os.environ["TOKENIZERS_PARALLELISM"] = "false"
bsinput=int(sys.argv[1])
distcase=int(sys.argv[2])
num_epochs = 1
num_iters = 46
listmodel=[]
bslist=[bsinput]

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
rank = dist.get_rank()
world_size = dist.get_world_size()
hostname = socket.gethostname()

# Check GPU availability and set device. Eearly detect CUDA/node problem.
try:
    num_gpus = torch.cuda.device_count()
    print(f"Hostname: {hostname}, Local Rank: {local_rank}, Global Rank: {global_rank}, NUM_GPUS: {num_gpus}")
    torch.cuda.set_device(local_rank)
    print(f"[{hostname}] Rank {rank}, Local Rank {local_rank}: CUDA device set to {local_rank}")
    if local_rank == 0:
        print(f"[{hostname}] NumPy version: , {np.__version__}")
        print(f"[{hostname}] Torch version:, {torch.__version__}")
    torch.cuda.empty_cache()
except Exception as e:
    print(f"[{hostname}] Rank {rank}, Local Rank {local_rank}: Error setting CUDA device: {e}")
    sys.exit(1)  # Exit if there's a critical failure in setting the device


# Load a dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_dataset = dataset['train']
print("Dataset loaded.")
# namelist = ['meta-llama/Llama-3.2-1B']
namelist = ['NousResearch/Llama-3.2-1B']


for batch_size in bslist:
    for name in namelist:
        try:
            print(f"[{hostname}] Rank {rank}, Local Rank {local_rank}: training model now: {name}")
            # Load the model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(name)
            tokenizer = AutoTokenizer.from_pretrained(name)
            
            # Add padding token to the tokenizer if it doesn't have one
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
            
            # Tokenize the dataset
            def tokenize_function(examples):
                return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)

            tokenized_datasets = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
            tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            first_batch_subset = Subset(tokenized_datasets, list(range(batch_size)))
            train_dataloader = DataLoader(first_batch_subset, batch_size=batch_size, shuffle=False,num_workers=8)
            sampler = DistributedSampler(first_batch_subset, shuffle=False)
            
            optimizer = optim.AdamW(model.parameters(), lr=5e-5)
            total_steps = len(train_dataloader) * num_epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
            torch.cuda.set_device(local_rank)
            torch.cuda.empty_cache()
            model = model.to('cuda:' + str(local_rank))
            if distcase == 1:
                model,distributed_info = tp.tensor_parallel(model, device_ids=[local_rank],distributed=True) 
            else:
                model = tp.tensor_parallel(model, device_ids=[local_rank],distributed=False) 
            # print("batches num",len(train_dataloader))
            for epoch in range(num_epochs):
                model.train()
                sampler.set_epoch(epoch)
                for batch in train_dataloader:
                    batch = {k: v.to(local_rank) for k, v in batch.items()}
                    total_time1 = 0
                    total_time2 = 0
                    for i in range(num_iters):
                        if 40<i<=50:
                            eg = ExecutionTraceObserver()
                            eg.register_callback("./llama_profiler/tp_"+str(distcase)+"_graph_"+name.replace("/","-")+"-"+hostname+"-rank"+str(rank)+"-iter"+str(i)+".json")
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
                            prof.export_chrome_trace("./llama_profiler/tp_"+str(distcase)+"_profiler_"+name.replace("/","-")+"-"+hostname+"-rank"+str(rank)+"-iter"+str(i)+".json")
                            eg.stop()
                            eg.unregister_callback()
                            print("Save Exeution Trace")
                            print(f"[{hostname}] Rank {rank}, Local Rank {local_rank}: {name}, gpu time: {curr_time}")
                            prof = None  # Release profiler reference to free memory
                            del eg  # Remove the observer to ensure no memory retention
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
                            print(f"[{hostname}] Rank {rank}, Local Rank {local_rank}: {name}, gpu time 2: {curr_time}")

                        else:
                            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
                            loss = outputs.loss
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            # print(name,"iter",i,"done")
                    print(f"[{hostname}] Rank {rank}, Local Rank {local_rank}: avg profiler Total time1: {total_time1/5}")
                    print(f"[{hostname}] Rank {rank}, Local Rank {local_rank}: avg Total time2: {total_time2/10}")   
                                
        except Exception as e:
            print(f"---first error\n[{hostname}] Rank {rank}, Local Rank {local_rank}, {e}")
            sys.exit(1)  # stop executing when there is an error    
            

dist.destroy_process_group()