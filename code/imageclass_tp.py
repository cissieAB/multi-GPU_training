import os
import socket
import time
import torch
import torchvision
import sys
import numpy as np
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
import tensor_parallel as tp
import torch.distributed as dist

from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import ExecutionTraceObserver
import torch.distributed as dist

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

num_iters = 46
bsinput=int(sys.argv[1])
distcase=int(sys.argv[2])
listmodel=[]

# 加载model
vgg11 = models.vgg11(weights=None) #weights=None
vgg13 = models.vgg13(weights=None)
vgg16 = models.vgg16(weights=None)
vgg19 = models.vgg19(weights=None)

resnet18 = models.resnet18(weights=None)
resnet34 = models.resnet34(weights=None)
resnet50 = models.resnet50(weights=None)
resnet101 = models.resnet101(weights=None)
resnet152 = models.resnet152(weights=None)

# alexnet = models.alexnet(weights=None)
# squeezenet1_1 = models.squeezenet1_1(weights=None)
# squeezenet1_0 = models.squeezenet1_0(weights=None)
densenet161 = models.densenet161(weights=None)
densenet169 = models.densenet169(weights=None)
densenet121 = models.densenet121(weights=None)
densenet201 = models.densenet201(weights=None)

bslist=[bsinput]    
data_transforms = {
    'predict': transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])}

for bs in bslist:
    
    BATCH_SIZE = bs
    EPOCHS = 1
    WORKERS = 4
    IMG_DIMS = (336, 336)
    CLASSES = 10
    
    listmodel=[vgg11,vgg13,vgg16,vgg19,resnet18,resnet34,resnet50,resnet101,resnet152,
              densenet161,densenet169,densenet121,densenet201]
    namelist = ['vgg11','vgg13','vgg16','vgg19','resnet18','resnet34','resnet50','resnet101','resnet152',
              'densenet161','densenet169','densenet121','densenet201']
    
    j = 0
    for model, name in zip(listmodel, namelist):
        dataset = {'predict' : torchvision.datasets.ImageFolder("./ILSVRC2012_img_val", data_transforms['predict'])}
        dataset_subset=dataset['predict']
        dataset_subset = torch.utils.data.Subset(dataset['predict'],range(BATCH_SIZE))
        data_loader = {'predict': torch.utils.data.DataLoader(dataset_subset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False,
                                            num_workers=WORKERS)}
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        try: 
            model = model.to('cuda:' + str(local_rank))
            if distcase == 1:
                model,distributed_info = tp.tensor_parallel(model, device_ids=[local_rank],distributed=True) 
                # for param in model.parameters():
                #     param.requires_grad = True
                # print(type(model), model)
            else:
                model = tp.tensor_parallel(model, device_ids=[local_rank],distributed=False) 

            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            for epoch in range(EPOCHS): # 1 epoch
                    model.train()
                    for features, labels in tqdm(data_loader['predict'], total=len(data_loader['predict'])):
                        features, labels = features.to(local_rank), labels.to(local_rank)
                        print("Batchsize:", features.size())
                        total_time1 = 0
                        total_time2 = 0
                        for i in range(num_iters): # 30 iterations
                            if 40<i<=num_iters-1:
                                eg = ExecutionTraceObserver()
                                eg.register_callback("./imageclass_tp_profiler/graph_"+name+"-"+hostname+"-rank"+str(rank)+"-iter"+str(i)+".json")
                                eg.start()
                                with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,with_stack=True,profile_memory=True) as prof:
                                    torch.cuda.synchronize()
                                    starter = torch.cuda.Event(enable_timing=True)
                                    ender = torch.cuda.Event(enable_timing=True)
                                    starter.record()
                                    optimizer.zero_grad()
                                    preds = model(features)
                                    if distcase == 1:
                                        preds.requires_grad_(True)
                                    loss = loss_fn(preds, labels)
                                    loss.backward()
                                    optimizer.step()
                                    
                                    ender.record()
                                    torch.cuda.synchronize()
                                    curr_time = starter.elapsed_time(ender)
                                    total_time1 +=curr_time                    
                            
                                prof.export_chrome_trace("./imageclass_tp_profiler/profiler_"+name+"-"+hostname+"-rank"+str(rank)+"-iter"+str(i)+".json")
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
                                
                                optimizer.zero_grad()
                                preds = model(features)
                                if distcase == 1:
                                    preds.requires_grad_(True)
                                loss = loss_fn(preds, labels)
                                loss.backward()
                                optimizer.step()
                                
                                ender.record()
                                torch.cuda.synchronize()
                                curr_time = starter.elapsed_time(ender)
                                total_time2 += curr_time
                                print(f"[{hostname}] Rank {rank}, Local Rank {local_rank}: {name}, gpu time 2: {curr_time}")
                            else:
                                optimizer.zero_grad()
                                preds = model(features)
                                if distcase == 1:
                                    preds.requires_grad_(True)
                                loss = loss_fn(preds, labels)
                                loss.backward()
                                optimizer.step()
                        print(f"[{hostname}] Rank {rank}, Local Rank {local_rank}: avg profiler Total time1: {total_time1/5}")
                        print(f"[{hostname}] Rank {rank}, Local Rank {local_rank}: avg Total time2:", total_time2/10)
        except Exception as e:
            print(f"---first error\n[{hostname}] Rank {rank}, Local Rank {local_rank}, {e}")
            sys.exit(1)  # stop executing when there is an error    
            

dist.destroy_process_group()