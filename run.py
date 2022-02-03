import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms as T
from torchvision.transforms import ToTensor, Lambda
import numpy as np
from tqdm import tqdm
from torchvision.models import resnet18

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt

from utils import LitModel

import tensorboard

##################
learning_rate = 1e-3
batch_size = 64
epochs = 100
num_tta = 10
progress_bar = False
checkpoint_verbose = False
earlystopping_verbose = False
##################

augmentation = T.Compose([T.ToTensor(),
                                T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
                                T.RandomHorizontalFlip(),
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

no_augmentation = T.Compose([T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=augmentation
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=no_augmentation
)

test_data_tta = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=augmentation
)

train_dataset, val_dataset = random_split(training_data, [45000, 5000])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=20)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=20)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=20)
test_dataloader_tta = DataLoader(test_data_tta, batch_size=batch_size, num_workers=20)

logger = TensorBoardLogger("tb_logs", name="my_model")
early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=earlystopping_verbose, mode="min")
checkpoint_callback = ModelCheckpoint('models', save_top_k=1, monitor='val_loss', verbose=checkpoint_verbose, mode='min')

architecture = resnet18(pretrained=True)
model = LitModel(architecture, learning_rate)
trainer = Trainer(max_epochs=epochs, 
                gpus=1,
                enable_progress_bar=progress_bar,
                logger=logger, 
                callbacks=[early_stop_callback, checkpoint_callback])

trainer.fit(model, train_dataloader, val_dataloader)

trainer.test(test_dataloaders=test_dataloader)

tta_pred_list = []
for _ in tqdm(range(num_tta)):
    y_hat = torch.vstack(trainer.predict(model=model, dataloaders=test_dataloader_tta))
    tta_pred_list.append(y_hat)
tta_pred_mean = torch.stack(tta_pred_list).mean(0)

tta_acc = np.mean(tta_pred_mean.argmax(1).numpy() == np.array(test_data.targets))
print(f"TTA accuracy: {tta_acc}")