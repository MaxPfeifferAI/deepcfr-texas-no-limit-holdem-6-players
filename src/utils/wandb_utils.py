"""Weights & Biases (wandb) utilities for logging experiments."""

import wandb

def init_wandb(project_name="deepcfr-poker", mode="disabled"):
    """Initialize wandb with the given project name."""
    wandb.init(project=project_name, mode=mode)

def log_metrics(metrics, step=None):
    """Log metrics to wandb."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)

def log_model(model, name):
    """Log a model to wandb."""
    if wandb.run is not None:
        wandb.save(name)

def finish():
    """Finish the current wandb run."""
    if wandb.run is not None:
        wandb.finish() 