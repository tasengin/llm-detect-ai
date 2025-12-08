import os
import socket
import torch
import torch.distributed as dist

def run(rank, world_size):
    if rank == 0:
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
        print(f"Master: {master_addr}:{master_port}")
        store = dist.TCPStore(master_addr, int(master_port), world_size, True)
        print("Master is ready.")
        store.set("message", "Hello")
        print("Master sent message.")
        print(f"Master received: {store.get('response')}")
    else:
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
        print(f"Worker {rank}: Connecting to {master_addr}:{master_port}")
        store = dist.TCPStore(master_addr, int(master_port), world_size, False)
        print(f"Worker {rank} is ready.")
        print(f"Worker {rank} received: {store.get('message')}")
        store.set("response", f"Hello from worker {rank}")
        print(f"Worker {rank} sent response.")

if __name__ == "__main__":
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    run(rank, world_size)
