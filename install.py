import os

commands = """
pip install -e .
pip install liger-kernel
GRADIO_SHARE=1 llamafactory-cli webui
""".strip()

for j in commands.split('\n'):
  os.system(commands)
