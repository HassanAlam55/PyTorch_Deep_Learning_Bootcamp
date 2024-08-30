'''
function for creating dataloader
'''
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def create_dataloaders(
            train_dir: str,
            test_dir: str,
            transform: transforms.Compose,
            batch_size: int,
            num_workers: int= 0,
            pin_memory = True
            ):
    
    """_summary_

    Returns:
        _type_: _description_
    """    """_summary_
    """    

    train_data = datasets.ImageFolder(train_dir, transform = transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    class_names = train_data.classses

    train_dataloader = DataLoader(
        train_data,
        batch_size=bath_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = pin_memory
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory=pin_memory
    )

    return train_dataloader, test_dataloader, class_names
