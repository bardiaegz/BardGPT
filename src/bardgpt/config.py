from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50_257
    block_size: int = 1_024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    expansion_factor: int = 4

B = 4
T = 32
k = 50
max_length = 30
num_return_sequenecs = 5