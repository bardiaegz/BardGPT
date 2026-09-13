from .model import GPT
from .config import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


def main() -> None:
    enc = tiktoken.get_encoding('gpt2')
    prompt = enc.encode("Hello, I'm a language model,")
    prompt = torch.tensor(prompt, dtype=torch.long)
    prompt = prompt.unsqueeze(0)
    prompt = prompt.repeat(num_return_sequenecs, 1)
    prompt = prompt.to('mps')
    model = GPT(GPTConfig(vocab_size=50_304))
    model.to('mps') # TODO: add auto-detect device

    model.eval()
    with torch.inference_mode():
        while prompt.size(1) < max_length:
            logits, _ = model(prompt)
            logits = logits[:, -1, :]
            probs = F.softmax(input=logits, dim=-1)
            topk_probs, topk_indicies = torch.topk(input=probs, k=k, dim=-1)
            ix = torch.multinomial(input=topk_probs, num_samples=1)
            xcol = torch.gather(input=topk_indicies, dim=-1, index=ix)
            prompt = torch.cat((prompt, xcol), dim=1)

    for i in range(num_return_sequenecs):
        decoded = enc.decode(prompt[i, :max_length].tolist())
        print(f'> {decoded}')