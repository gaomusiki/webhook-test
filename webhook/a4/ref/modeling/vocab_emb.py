from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup


class ParallelVocabEmbedding(nn.Module):
    """Parallel Vocab Embedding module
    This is a simplified version of the practical one, \
        which shards the vocabulary embedding table into `world_size` partitions in a process group, and \
        each rank in that process group only handles one partition of it, thus \
        in pratical, we can apply the large vocabulary embedding in parallel and then reduce them together.
    However, for this simpler module, you only need to implement the jobs for any single rank, and \
        don't have to handle the reduce operation or any other parallelism stuff.
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        emb_size: int,
        rank: int = 0,
        world_size: int = 1,
        process_group: Optional[ProcessGroup] = None,
        init_mean: float = 0.0,
        init_std: float = 1.0,
        init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        """Initialize Parallel Vocab Embedding module
        
        Args:
            vocab_size(int): vocabulary size
            emb_size(int): embedding size
            rank(int, default = 0): rank
            world_size(int, default = 1): world size
            process_group(Optional[ProcessGroup], default = None): the process group (which will not be used for this simpler module yet)
            init_mean(float, default = 0.0): mean of the normal distribution
            init_std(float, default = 1.0): standard deviation of the normal distribution
            init_base_seed(int, default = 42): the base seed for the initialization (the real seed will be base seed + rank)
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        # raise NotImplementedError("TODO: Assignment1 - Task2")
        
        assert vocab_size % world_size == 0, f"vocab_size {vocab_size} should be divisible by world_size {world_size}"
        
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group
        self.init_mean = init_mean
        self.init_std = init_std
        self.init_seed = init_base_seed + rank
        self.dtype = dtype
        self.device = device
        
        # get the start and end index in the vocabulary for this partition
        self.vocab_size_per_partition = vocab_size // world_size
        self.vocab_start_idx = rank * self.vocab_size_per_partition
        self.vocab_end_idx = self.vocab_start_idx + self.vocab_size_per_partition
        
        # init the embedding weight for this partition
        self.weight = nn.Parameter(
            torch.empty(
                self.vocab_size_per_partition,
                self.emb_size,
                dtype=self.dtype,
                device=self.device,
            )
        )
        self.reset_parameters()
        
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """The forward pass for Parallel Vocab Embedding module
        
        Args:
            input_ids(torch.LongTensor): input ids, with shape: (batch_size, seq_len)
        
        Returns:
            output(torch.Tensor): output embedding tensor, with shape: (batch_size, seq_len, emb_size)
        """
        # raise NotImplementedError("TODO: Assignment1 - Task2")
        
        # process the input ids
        input_ids_device = input_ids.device
        input_ids = input_ids.clone().to(self.weight.device)
        if self.world_size > 1:
            # build the input ids mask
            input_mask = (input_ids < self.vocab_start_idx) | (input_ids >= self.vocab_end_idx)
            
            # shift the input ids
            input_ids -= self.vocab_start_idx
            
            # mask the input ids
            input_ids[input_mask] = 0
            
        # get the output embeddings for this partition
        output = F.embedding(
            input_ids,
            self.weight,
        ).to(input_ids_device)
        
        # mask the output embeddings.
        if self.world_size > 1:
            output[input_mask, :] = 0.0
            
        # TODO: reduce the output embeddings among the process group
        
        return output
        
    def reset_parameters(self) -> None:
        """Initialize learnable embedding parameters for Vocab Embedding from a normal distribution"""
        # raise NotImplementedError("TODO: Assignment1 - Task2")
        
        torch.manual_seed(self.init_seed)
        nn.init.normal_(self.weight, self.init_mean, self.init_std)
        