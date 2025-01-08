from src.trainer import Trainer
from src.validator import Validator

DEBUG_TRAIN: bool = False # Option for testing the training code without validation phase
DEBUG_VALIDATION: bool = False # Option for testing the validation code without training beforehand

def train_cycle(
    trainer: Trainer,
    validator: Validator,
    start_epoch: int,
    num_epochs: int,
    skip_valid: bool,
):
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        # Training
        if not DEBUG_VALIDATION:
            trainer.train(checkpoint_name=f'checkpoint_{epoch:03d}')

        # Validation
        if not skip_valid and not DEBUG_TRAIN:
            validator.eval(checkpoint_name=f'checkpoint_{epoch:03d}')
        
