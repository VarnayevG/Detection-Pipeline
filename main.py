import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from detection.data import ClassificationDataset
from detection.model import ResNet50
from detection.utils import get_predicts, plot_learning_curves, train


TRAIN_STAGE = 'train'
VALIDATION_STAGE = 'val'
TEST_STAGE = 'test'
BATCH_SIZE, NUM_EPOCHS = 32, 50
LR, FACTOR = 1e-05, 0.3
SCH_TYPE = 'plateao'
Y_LIM = (0, 1.5)

logging.basicConfig(level=logging.DEBUG)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-trp', '--train_img', type=str, help='Path to train images', required=True)
    parser.add_argument('-vp', '--val_img', type=str, help='Path to validation images', required=True)
    parser.add_argument('-tp', '--test_img', type=str, help='Path to test images', required=True)
    parser.add_argument('-tl', '--train_lbl', type=str, help='Path to train labels', required=True)
    parser.add_argument('-vl', '--val_lbl', type=str, help='Path to val labels', required=True)
    parser.add_argument('-o', '--output_path', type=str, help='Path where output is to be stored', required=True)

    args = parser.parse_args()
    logging.info('Parsed data')

    train_data = ClassificationDataset(Path(args.train_img),  Path(args.train_lbl), TRAIN_STAGE)
    val_data = ClassificationDataset(Path(args.val_img), Path(args.val_lbl), VALIDATION_STAGE)
    test_data = ClassificationDataset(Path(args.test_img), None, TEST_STAGE)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    logging.info('Created train/val/test DataLoaders')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = ResNet50()
    net = net.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=FACTOR)
    logging.info('Initialize NN and params')

    net, history = train(
        net, device, criterion, optimizer, scheduler,  train_loader, val_loader, sch_type=SCH_TYPE, num_epochs=NUM_EPOCHS, ylim=Y_LIM)
    plot_learning_curves(history, y_lim=Y_LIM)
    logging.info('Trained NN')

    preds = get_predicts(net, test_loader)
    probs = 1 / (1 + np.exp(-np.array(preds)))
    logging.info('Calculated probabilities')

    output = pd.DataFrame(columns=['id', 'target_people'])
    output['id'] = np.arange(1, probs.shape[0] + 1, 1)
    output['target_people'] = probs
    output.to_csv(Path(args.output_path), index=False)
    logging.info('Output is created, terminating')

if __name__ == '__main__':
    main()
