import torch
class Config:
    # Data
    batch_size = 64 
    block_size = 256 
    
    # Training
    resume_path = None
    max_iters = 5000
    eval_interval = 500
    eval_iters = 200
    learning_rate = 3e-4 # Learning rate standard per Transformer
    weight_decay = 1e-1
    device = 'cpu' 
    if torch.backends.mps.is_available():
        device = 'mps'
    if torch.cuda.is_available():
        device = 'cuda'    
    seed = 1337
    
    # Logging & Checkpoints
    use_wandb = False 
    project_name = "nano-gpt-rehab"
    run_name = "bigram-baseline"
    checkpoint_dir = "checkpoints"
    checkpoint_interval = 1000

