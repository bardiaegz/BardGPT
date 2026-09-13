from dataclasses import dataclass

class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    OKGREEN = '\033[38;2;123;198;33m'

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
max_step = 500