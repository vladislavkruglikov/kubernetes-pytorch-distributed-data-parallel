import os
import time
import torch


torch_distributed_backend = os.getenv(key="TORCH_DISTRIBUTED_BACKEND")
torch_distributed_world_size = int(os.getenv(key="TORCH_DISTRIBUTED_WORLD_SIZE"))
torch_distributed_rank = int(os.getenv(key="TORCH_DISTRIBUTED_RANK"))
torch_distributed_master_address = os.getenv(key="TORCH_DISTRIBUTED_MASTER_ADDRESS")
torch_distributed_master_port = os.getenv(key="TORCH_DISTRIBUTED_MASTER_PORT")

os.environ['MASTER_ADDR'] = torch_distributed_master_address
os.environ['MASTER_PORT'] = torch_distributed_master_port

print(torch_distributed_backend)
print(torch_distributed_world_size)
print(torch_distributed_rank)
print(torch_distributed_master_address)
print(torch_distributed_master_port)

torch.distributed.init_process_group(
    backend=torch_distributed_backend,
    world_size=torch_distributed_world_size,
    rank=torch_distributed_rank
)


class MyTrainDataset(torch.utils.data.Dataset):
    def __init__(self, size: int):
        self._size = size
        self._dataset = [(torch.rand(2048), torch.rand(32)) for _ in range(size)]

    def __len__(self):
        return self._size
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._dataset[index]


model = torch.nn.Linear(in_features=2048, out_features=32)

distributed_data_parallel_model = torch.nn.parallel.DistributedDataParallel(module=model)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

dataset = MyTrainDataset(size=16384)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    pin_memory=True,
    shuffle=False,
    sampler=torch.utils.data.distributed.DistributedSampler(dataset=dataset)
)

start_time = time.time()

for epoch in range(16):
    batch_size = len(next(iter(data_loader))[0])
    data_loader.sampler.set_epoch(epoch=epoch)
    steps = len(data_loader)
    message_template = "Rank {rank} start epoch {epoch} with batch_size {batch_size} with steps {steps}✨"
    message = message_template.format(
        rank=torch_distributed_rank, 
        epoch=epoch,
        batch_size=batch_size,
        steps=steps
    )

    print(message)

    for source, targets in data_loader:
        optimizer.zero_grad()
        output = model(input=source)
        loss = torch.nn.functional.cross_entropy(input=output, target=targets)
        loss.backward()
        optimizer.step()

end_time = time.time()

print(end_time - start_time)

torch.distributed.destroy_process_group()
