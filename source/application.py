import os
import torch
import datetime


backend = os.getenv(key="BACKEND")
world_size = int(os.getenv(key="WORLD_SIZE"))
rank = int(os.getenv(key="RANK"))
master_address = os.getenv(key="MASTER_ADDRESS")
master_port = int(os.getenv(key="MASTER_PORT"))
time_out = int(os.getenv(key="TIME_OUT"))

os.environ['MASTER_ADDR'] = master_address
os.environ['MASTER_PORT'] = str(master_port)

# The machine with rank 0 will be used to set up all connections

print("📦 Backend is {backend}".format(backend=backend))
print("🌍 World size is {world_size}".format(world_size=world_size))
print("🏆 Rank is {rank}".format(rank=rank))
print("🏤 Master address is {master_address}".format(master_address=master_address))
print("🏤 Master port is {master_port}".format(master_port=master_port))
print("⏰ Time out is {time_out}".format(time_out=time_out))

torch.distributed.init_process_group(
    backend=backend,
    timeout=datetime.timedelta(seconds=time_out),
    world_size=world_size,
    rank=rank,
)

print("🌱 Successfully created distributed context")

torch.distributed.destroy_process_group()
