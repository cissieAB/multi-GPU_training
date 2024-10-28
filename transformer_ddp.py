import sys
import socket
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import ExecutionTraceObserver
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

os.environ["TOKENIZERS_PARALLELISM"] = "false"
bsinput=int(sys.argv[1])
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

# Check GPU availability and set device
try:
    num_gpus = torch.cuda.device_count()
    print(f"Hostname: {hostname}, Rank: {rank}, Local Rank: {local_rank}, Global Rank: {global_rank}, NUM_GPS: {num_gpus}")
    torch.cuda.set_device(local_rank)
    print(f"[{hostname}] Rank {rank}, Local Rank {local_rank}: CUDA device set to {local_rank}")
    torch.cuda.empty_cache()
except Exception as e:
    print(f"[{hostname}] Rank {rank}, Local Rank {local_rank}: Error setting CUDA device: {e}")
    sys.exit(1)  # Exit if there's a critical failure in setting the device

# Load a dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_dataset = dataset['train']
print("Dataset loaded.")
namelist = ['gpt2','gpt2-medium','bert-base-uncased','bert-large-cased',"google-t5/t5-small","google/flan-t5-small","google-t5/t5-large","google/flan-t5-large"]
namelist = ['gpt2','gpt2-medium','bert-base-uncased','bert-large-cased',"google-t5/t5-small","google/flan-t5-small"]
namelist = ['gpt2']

total_time1 = 0
total_time2 = 0
for batch_size in bslist:
    for name in namelist:
        try:
            print("training model now:", name)
            # Load the model and tokenizer
            if 't5' in name:
                tokenizer = AutoTokenizer.from_pretrained(name)
                model = T5ForConditionalGeneration.from_pretrained(name)
            else:
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
            # tokenized_datasets.set_format(type='torch', columns=['input_ids'])
            tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            first_batch_subset = Subset(tokenized_datasets, list(range(batch_size)))
            sampler = DistributedSampler(first_batch_subset, shuffle=False)
            train_dataloader = DataLoader(first_batch_subset, batch_size=batch_size, sampler=sampler, shuffle=False,num_workers=4)
            
            torch.cuda.set_device(local_rank)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            optimizer = optim.AdamW(model.parameters(), lr=5e-5)
            model = model.to('cuda:' + str(local_rank))
            model = DDP(model, device_ids=[local_rank])
            for epoch in range(num_epochs):
                model.train()
                sampler.set_epoch(epoch)
                for batch in train_dataloader:
                    batch = {k: v.to(local_rank) for k, v in batch.items()} #'cuda:' + str(local_rank)
                    total_time1 = 0
                    total_time2 = 0
                    for i in range(num_iters):
                        if 40<i<=45:
                            eg = ExecutionTraceObserver()
                            eg.register_callback("./graph_"+name.replace("/","-")+".json")
                            # eg.register_callback("./transformer_ddp_profiler/graph_"+name.replace("/","-")+".json")
                            eg.start()
                            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,with_stack=True,profile_memory=True,
                                on_trace_ready=torch.profiler.tensorboard_trace_handler("./transformer_ddp_profiler/profiler_"+name.replace("/","-")+"-iter"+str(i)+".json")) as prof:
                                torch.cuda.synchronize()
                                starter = torch.cuda.Event(enable_timing=True)
                                ender = torch.cuda.Event(enable_timing=True)
                                starter.record()
                                
                                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
                                loss = outputs.loss
                                loss.backward()
                                optimizer.step()
                                optimizer.zero_grad()
                                
                                ender.record()
                                torch.cuda.synchronize()
                                curr_time = starter.elapsed_time(ender)
                                total_time1 += curr_time
                            # prof.export_chrome_trace("./transformer_ddp_profiler/profiler_"+name.replace("/","-")+"-iter"+str(i)+".json")
                            eg.stop()
                            eg.unregister_callback()
                            # print("Save Exeution Trace")
                            print(f"[{hostname}] Rank {rank}, Local Rank {local_rank}: {name}, gpu time: {curr_time}")
                        elif 30<i<=40:
                            torch.cuda.synchronize()
                            starter = torch.cuda.Event(enable_timing=True)
                            ender = torch.cuda.Event(enable_timing=True)
                            starter.record()
                            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
                            loss = outputs.loss
                            loss.backward()
                            optimizer.step()
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
                            optimizer.zero_grad()
                    print(f"[{hostname}] Rank {rank}, Local Rank {local_rank}: avg profiler Total time1: {total_time1/5}")
                    print(f"[{hostname}] Rank {rank}, Local Rank {local_rank}: avg Total time2:", total_time2/5)
        except Exception as e:
            print(f"---first error\n[{hostname}] Rank {rank}, Local Rank {local_rank}\n",e)
            # pass
dist.destroy_process_group()

