import tiktoken
import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
from datasets import load_dataset

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)

# these stay at module level so the worker processes (which re-import this file) can use them
enc = tiktoken.get_encoding('gpt2')
eot = enc.eot_token

def tokenize(doc):
  tokens = [eot]
  tokens.extend(enc.encode_ordinary(doc['text']))
  tokens_np = np.array(tokens)
  assert (0 <= tokens_np).all() and (tokens_np < 2 ** 16).all(), 'the tokens either too small or too large'
  tokens_np_uint16 = tokens_np.astype(np.uint16)
  return tokens_np_uint16

def write_datafile(filename, tokens_np):
  np.save(filename, tokens_np)

def main():
  fw = load_dataset(path='HuggingFaceFW/fineweb-edu', name=remote_name, split='train')
  os.makedirs(DATA_CACHE_DIR, exist_ok=True)

  nprocs = max(1, os.cpu_count() // 2)

  with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, fw, chunksize=16):
      if token_count + len(tokens) < shard_size:
        all_tokens_np[token_count:token_count+len(tokens)] = tokens
        token_count += len(tokens)

        if progress_bar is None:
          progress_bar = tqdm(total=shard_size, unit='tokens', desc=f'Shard {shard_index}')
        progress_bar.update(len(tokens))
      else:
        split = 'val' if shard_index == 0 else 'train'
        filename = os.path.join(DATA_CACHE_DIR, f'edufineweb_{split}_{shard_index:06d}')
        remainder = shard_size - token_count
        all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
        if progress_bar is not None:
          progress_bar.update(remainder)
          progress_bar.close()
        write_datafile(filename, all_tokens_np)
        shard_index += 1
        progress_bar = None
        all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
        token_count = len(tokens) - remainder

    if token_count != 0:
      split = 'val' if shard_index == 0 else 'train'
      filename = os.path.join(DATA_CACHE_DIR, f'edufineweb_{split}_{shard_index:06d}')
      write_datafile(filename, all_tokens_np[:token_count])

if __name__ == '__main__':
  main()
