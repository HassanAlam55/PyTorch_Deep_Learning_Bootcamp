'''
utils form odel
'''
from pathlib import Path
import torch

def save_model (model: torch.nn.Module,
            target_dir: str,
            model_name: str):
    """_summary_

    Args:
        model (torch.nn.Module): _description_
        target_dir (str): _description_
        model_name (str): _description_
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok = True)

    assert model_name.endswith('.pth') or model_name.endswith('pt'), 'model_name shoule end with ".pt", or ".pth"'
    model_save_path = target_dir_path/model_name

    print (f'[INFO] Saving model to: {model_save_path}')
    torch.save(obj = model.state(),
                f=model_save_path)
