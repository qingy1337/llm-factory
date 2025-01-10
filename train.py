import subprocess

# Command 1: llamafactory-cli train
command1 = [
    "llamafactory-cli", "train",
    "--stage", "sft",
    "--do_train", "True",
    "--model_name_or_path", "./checkpoint",
    "--preprocessing_num_workers", "16",
    "--finetuning_type", "full",
    "--template", "default",
    "--rope_scaling", "linear",
    "--flash_attn", "auto",
    "--enable_liger_kernel", "True",
    "--dataset_dir", "data",
    "--dataset", "smoltalk-1M",
    "--cutoff_len", "8192",
    "--learning_rate", "0.0002",
    "--num_train_epochs", "1.0",
    "--max_samples", "2000000",
    "--per_device_train_batch_size", "16",
    "--gradient_accumulation_steps", "2",
    "--lr_scheduler_type", "cosine",
    "--max_grad_norm", "1.0",
    "--logging_steps", "3",
    "--save_steps", "50",
    "--warmup_steps", "10",
    "--packing", "True",
    "--report_to", "none",
    "--output_dir", "saves",
    "--pure_bf16", "True",
    "--plot_loss", "True",
    "--trust_remote_code", "True",
    "--ddp_timeout", "180000000",
    "--optim", "adamw_torch"
]

# Command 2: python3 upload.py
command2 = ["python", "upload.py"]

# Run both commands
process1 = subprocess.Popen(command1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
process2 = subprocess.Popen(command2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd="checkpoint")

# Capture and display the output of the first command
for line in iter(process1.stdout.readline, ""):
    print(line, end="")

# Wait for both processes to finish
process1.wait()
process2.wait()
