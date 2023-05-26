from collections import defaultdict
import logging
from pathlib import Path
from typing import List

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def eval_epoch(model,
               batch_gen,
               device,
               criterion,
               optimizer,
               is_train=False):
    '''
    Run single train and eval epoch.
    '''

    epoch_loss = 0
    model.train(is_train)

    preds = []
    labels = []
    for X_batch, y_batch in tqdm(batch_gen):

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits.reshape(-1), y_batch.float().reshape(-1).to(device))

        if is_train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += np.sum(loss.detach().cpu().numpy())
        preds += logits.detach().cpu().numpy().tolist()
        labels += y_batch.detach().cpu().numpy().tolist()

    epoch_loss /= len(batch_gen)
    epoch_auc = roc_auc_score(labels, preds)

    return epoch_loss, epoch_auc


def train(model,
          device,
          criterion,
          optimizer,
          scheduler,
          train_batch_gen,
          val_batch_gen,
          sch_type,
          num_epochs=50,
          ylim=(2, 4)):
    '''
    Train model and log loss values.
    '''

    history = defaultdict(lambda: defaultdict(list))

    for epoch in range(num_epochs):

        train_loss, train_auc = eval_epoch(model, train_batch_gen, device, criterion, optimizer, is_train=True)
        history['loss']['train'].append(train_loss)
        history['auc']['train'].append(train_auc)

        val_loss, val_auc = eval_epoch(model, val_batch_gen, device, criterion, optimizer, is_train=False)
        history['loss']['val'].append(val_loss)
        history['auc']['val'].append(val_auc)

        if sch_type == 'plateao':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        if epoch == 1 or epoch + 1 % 5 == 0:
            logging.info(f'Epoch {epoch + 1} of {num_epochs}:\n')
            logging.info(f'\train loss: {train_loss}\nval loss: {val_loss}')
            logging.info(f'\train auc: {train_auc}\nval loss: {val_auc}')

    return model, history


def get_predicts(model, batch_gen, device):
    model.train(False)
    preds = []
    for X_batch in tqdm(batch_gen):
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        y_pred = logits.detach().cpu().numpy().reshape(-1)
        preds += y_pred.tolist()

    return np.array(preds)


def plot_learning_curves(path: Path, history: List, y_lim=(2, 4)):
    '''
    Функция для обучения модели и вывода лосса и метрики во время обучения.

    :param history: (dict)
        accuracy и loss на обучении и валидации
    '''
    sns.set_style(style='whitegrid')
    plt.figure(figsize=(20, 7))

    plt.subplot(1, 2, 1)
    plt.title('Loss vs. epoch', fontsize=15)
    plt.plot(history['loss']['train'], label='train')
    plt.plot(history['loss']['val'], label='val')
    plt.ylabel('Loss ', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.legend()
    plt.ylim(y_lim)

    plt.subplot(1, 2, 2)
    plt.title('ROC AUC score vs. epoch', fontsize=15)
    plt.plot(history['auc']['train'], label='train')
    plt.plot(history['auc']['val'], label='val')
    plt.ylabel('ROC AUC score', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.savefig(path)
    plt.legend()
    plt.show()
