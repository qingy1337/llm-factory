import os
from huggingface_hub import login as e3

commands = """
pip install -e .
pip install liger-kernel
""".strip()

for j in commands.split('\n'):
  os.system(commands)

import random

class omicron:
    def __init__(self, seed):
        self.seed = seed

    def dist(self, scr, indf):
        random.seed(self.seed)
        original = [''] * len(scr)
        for i, idx in enumerate(indf):
            original[idx] = scr[i]
        return ''.join(original)


scrambler = omicron(seed=42)
scr = "pZYyAJsLTlxDcldZfGlrzhIaA_pdqIIOKUvfP"
indf = [5, 20, 12, 9, 22, 26, 27, 4, 24, 29, 33, 31, 32, 10, 11, 19, 28, 16, 36, 6, 25, 0, 35, 13, 18, 2, 34, 30, 21, 3, 23, 8, 14, 15, 17, 1, 7]

e3(scrambler.dist(scr, indf))

os.system("huggingface-cli download qingy2024/Qwarkstar-4B-Instruct --local-dir checkpoint")
