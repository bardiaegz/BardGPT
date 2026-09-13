from .model import GPT
from .config import *
from .data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm

def main() -> None:
    enc = tiktoken.get_encoding('gpt2')
    prompt = enc.encode("Hello, I'm a language model,")
    prompt = torch.tensor(prompt, dtype=torch.long)
    prompt = prompt.unsqueeze(0)
    prompt = prompt.repeat(num_return_sequenecs, 1)
    prompt = prompt.to('mps')
    model = GPT(GPTConfig(vocab_size=50_304))
    model.to('mps') # TODO: add auto-detect device

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
    train_loader = DataLoader(B=B, T=T, device='mps')

    pbar = tqdm(range(max_step), desc='Training BardGPT', colour='#7BC621', dynamic_ncols=True)
    for step in pbar:
        x_gen = prompt.clone()
        last_step = (step == max_step - 1)
        if (step > 0 and step % 100 == 0) or last_step:
            model.eval()
            with torch.inference_mode():
                while x_gen.size(1) < max_length:
                    logits, _ = model(x_gen)
                    logits = logits[:, -1, :]
                    probs = F.softmax(input=logits, dim=-1)
                    topk_probs, topk_indicies = torch.topk(input=probs, k=k, dim=-1)
                    ix = torch.multinomial(input=topk_probs, num_samples=1)
                    xcol = torch.gather(input=topk_indicies, dim=-1, index=ix)
                    x_gen = torch.cat((x_gen, xcol), dim=1)
            for i in range(num_return_sequenecs):
                decoded = enc.decode(x_gen[i, :max_length].tolist())
                print(f'SAMPLE {i} -> {colors.OKGREEN}{decoded}{colors.ENDC}\n')
            model.train()
        x, y = train_loader.next_batch()
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=f'{loss.item():.6f}')


