import os
import sys
import torch
from torchvision import transforms
import importlib

# from going_modular import data_loaders, engine, hassan_TinyVGG, utils
import data_loaders, engine, hassan_TinyVGG, utils
# sys.path.append('./going_modular') 
# importlib.reload(data_loaders)
# importlib.reload(engine)
# importlib.reload(hassan_TinyVGG)
# importlib.reload(utils)
# from going_modular importlib.reload(ibkr_helper)
# importlib.reload(general_helper)

torch.manual_seed(42)
torch.cuda.manual_seed(42)


NUM_EPOCS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARING_RATE = 0.001

train_dir = '../data/pizza_steak_sushi_20_percent/train'
test_dir = '../data/pizza_steak_sushi_20_percent/test'

# train_dir = '../../data/pizza_steak_sushi_20_percent/train'
# test_dir = '../../data/pizza_steak_sushi_20_percent/test'

print(os.path.abspath(train_dir))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_transform = transforms.Compose([
    transforms.Resize(size = (224, 224)),
                      transforms.ToTensor()])

train_dataloader, test_datalaoder, class_names = data_loaders.create_dataloaders(
    train_dir=train_dir,
    test_dir = test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE

)

hassan_custom_model  = hassan_TinyVGG.HassanFood(
                        input_shape = 3,
                        hidden_units = 9,
                        output_shape= len(class_names)).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=hassan_custom_model.parameters(), lr = 0.001)

from timeit import default_timer as timer 
start_time = timer ()

model_0_results = engine.train_model(model=hassan_custom_model,
                                     train_dataloader=train_dataloader,
                                     test_dataloader=test_datalaoder,
                                     loss_fn = loss_fn,
                                     optimizer=optimizer,
                                     epochs=NUM_EPOCS,
                                     device=device)

end_time = timer()
print(f'[INFO] total train time : {end_time - start_time:.3f} seconds')

utils.save_model (model = hassan_custom_model,
           target_dir = 'models',
            model_name = '05_hassan_model_0.pth' )
