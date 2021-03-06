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
import argparse

import tensorboard

# argpaser
parser = argparse.ArgumentParser(description='PyTorch Lightning Example')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--num_tta', type=int, default=10)
parser.add_argument('--es_patience', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--progress_bar', type=bool, default=False)
parser.add_argument('--checkpoint_verbose', type=bool, default=False)
parser.add_argument('--earlystopping_verbose', type=bool, default=False)
args = parser.parse_args()

seed = args.seed

architecture = resnet18(pretrained=True)

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

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
test_dataloader_tta = DataLoader(test_data_tta, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

logger = TensorBoardLogger("tb_logs", name="my_model")
early_stop_callback = EarlyStopping(monitor="val_loss", patience=args.es_patience, verbose=args.earlystopping_verbose, mode="min")
checkpoint_callback = ModelCheckpoint('models', save_top_k=1, monitor='val_loss', verbose=args.checkpoint_verbose, mode='min')


model = LitModel(architecture, args.learning_rate)
trainer = Trainer(max_epochs=args.epochs, 
                gpus=1,
                enable_progress_bar=args.progress_bar,
                logger=logger, 
                callbacks=[early_stop_callback, checkpoint_callback])

trainer.fit(model, train_dataloader, val_dataloader)

trainer.test(test_dataloaders=test_dataloader)

tta_pred_list = []
for _ in tqdm(range(args.num_tta)):
    y_hat = torch.vstack(trainer.predict(model=model, dataloaders=test_dataloader_tta))
    tta_pred_list.append(y_hat)
tta_pred_mean = torch.stack(tta_pred_list).mean(0)

tta_acc = np.mean(tta_pred_mean.argmax(1).numpy() == np.array(test_data.targets))
print(f"TTA accuracy: {tta_acc}")