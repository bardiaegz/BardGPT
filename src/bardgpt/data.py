import tiktoken
import torch
import os
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:
    def __init__(self, B, T, split='train', device='cpu'):
        self.B, self.T, self.split, self.device = B, T, split, device
        assert split in {'train', 'val'}

        data_root = 'edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f'no shards found for split {split}'

        enc = tiktoken.get_encoding('gpt2')

        tokens = enc.encode_ordinary(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        print(f'loaded {len(shards)} shards for split {split}')

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        buf = buf.to(self.device)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T

        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0

        return x, y
        