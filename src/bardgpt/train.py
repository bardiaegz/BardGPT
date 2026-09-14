from .model import GPT
from .config import *
from .data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm
import time
import os

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'log')
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, 'log.txt')

BANNER = r"""
  ____                _  _____ _____ _______ 
 |  _ \              | |/ ____|  __ \__   __|
 | |_) | __ _ _ __ __| | |  __| |__) | | |   
 |  _ < / _` | '__/ _` | | |_ |  ___/  | |   
 | |_) | (_| | | | (_| | |__| | |      | |   
 |____/ \__,_|_|  \__,_|\_____|_|      |_|   
"""

def main() -> None:
    with open(log_path, mode='w') as log_file:
        print(BANNER)
        enc = tiktoken.get_encoding('gpt2')
        prompt = enc.encode("Hello, I'm a language model,")
        prompt = torch.tensor(prompt, dtype=torch.long)
        prompt = prompt.unsqueeze(0)
        prompt = prompt.repeat(num_return_sequences, 1)
        prompt = prompt.to(device)

        # https://arxiv.org/pdf/2203.03341 - Recovering single precision accuracy from Tensor Cores while surpassing the FP32 theoretical peak performance
        # TensorFloat-32 (1 sign bits - 8 exponent bits - 10 mantissa bits (same as float16)) -> 19 bits
        torch.set_float32_matmul_precision('high')
        model = GPT(GPTConfig(vocab_size=50_304))
        model.to(device)
        raw_model = model
        use_compile = False
        if use_compile:
            model = torch.compile(model)
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=6e-4)
        train_loader = DataLoader(B=B, T=T, device=device)

        pbar = tqdm(range(max_step), desc='Training BardGPT', colour='#7BC621', dynamic_ncols=True)
        for step in pbar:
            last_step = (step == max_step - 1)
            if (step > 0 and step % 100 == 0) or last_step:
                x_gen = prompt.clone()
                log_file.write(f'\n\n{'=' * 24} GENERATION {'=' * 24}\n')
                raw_model.eval()
                with torch.inference_mode():
                    while x_gen.size(1) < max_length:
                        # https://arxiv.org/pdf/1905.12322 - A Study of BFLOAT16 for Deep Learning Training
                        # https://docs.cloud.google.com/tpu/docs/bfloat16
                        # BrainFloat-16 (1 sign bits - 8 exponent bits (same as float32) - 7 mantissa bits ) -> 16 bits
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, _ = model(x_gen)
                        logits = logits[:, -1, :].float()
                        logits[:, enc.n_vocab:] = -float('inf')
                        probs = F.softmax(input=logits, dim=-1)
                        topk_probs, topk_indices = torch.topk(input=probs, k=k, dim=-1)
                        ix = torch.multinomial(input=topk_probs, num_samples=1)
                        xcol = torch.gather(input=topk_indices, dim=-1, index=ix)
                        x_gen = torch.cat((x_gen, xcol), dim=1)
                for i in range(num_return_sequences):
                    decoded = enc.decode(x_gen[i, :max_length].tolist())
                    log_file.write(f'\nSAMPLE {i} -> {decoded}\n')
                    print(f'SAMPLE {i} -> {colors.OKGREEN}{decoded}{colors.ENDC}\n')
                log_file.write(f'\n{'=' * 60}')
                raw_model.train()
            t0 = time.time()
            x, y = train_loader.next_batch()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = raw_model(x, y)
            optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
            optimizer.step()
            if device_type == 'cuda':
                torch.cuda.synchronize()
            elif device_type == 'mps':
                torch.mps.synchronize()
            t1 = time.time()
            dt = t1 - t0
            token_processed = train_loader.B * train_loader.T
            toksec = token_processed / dt
            log_file.write(f'\nSTEP: {step:05d} | NORM: {norm:.4f} | LOSS: {loss.item():.6f}')
            pbar.set_postfix(loss=f'{loss.item():.6f}', norm=f'{norm:.4f}', dt=f'{dt*1000:.0f}ms', toksec=f'{toksec:.2f}')
            log_file.flush()