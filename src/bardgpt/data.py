import tiktoken
import torch

class DataLoader:
    def __init__(self, B, T, device='cpu'):
        self.B, self.T, self.device = B, T, device

        enc = tiktoken.get_encoding('gpt2')
        with open('input.txt', mode='r') as f:
            text = f.read()

        tokens = enc.encode_ordinary(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        print(f'loaded {len(self.tokens)} tokens.')
        print(f'each epoch contains {len(self.tokens) // (B * T)} batches.')

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        buf = buf.to(self.device)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T

        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y
        