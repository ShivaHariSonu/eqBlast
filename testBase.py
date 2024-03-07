#!/usr/bin/env python3
# Purpose: Applies a fitted base model from trainBase.py to the testing data.
# Author: Ben Baker,Shiva Hari (University of Utah) distributed under the MIT license.
import torch
import h5py
import argparse
import logging
import numpy as np
from numpyDataset import NumpyDataset
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from cnn import CNN
import os

if __name__ == "__main__":
    model_path = '/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/machineLearning/eqBlast/trainBase/model_23823/model_59.pt'
    test_path = 'test.h5'
    device = 'cpu'
    batch_size = 32

    log_level = logging.INFO
    """
    # Create the logger
    log_level = logging.CRITICAL
    if (args.verbosity == 2): 
        log_level = logging.INFO
    elif (args.verbosity >= 3): 
        log_level = logging.DEBUG
    else:
        log_level = logging.CRITICAL
    """

    logging.basicConfig(format = '%(asctime)s %(levelname)-8s %(message)s',
                        level = log_level,
                        datefmt='%Y-%m-%d %H:%M:%S')

    if (not os.path.exists(model_path)):
        sys.exit("Trained model {} does not exist".format(model_path))
    if (not os.path.exists(test_path)):
        sys.exit("Test data {} does not exist".format(model_path))
 
    logging.info("Loading model...")
    model = CNN()
    model.load_from_jit(model_path)
    #model.load_state_dict(torch.jit.load(model_path).state_dict())
    model.to(device)
    model.eval()

    logging.info("Loading test data in {}".format(test_path))
    h5_handle = h5py.File(test_path, 'r')
    X_test = h5_handle['X'][:]
    y_test = h5_handle['y'][:]

    logging.info("Putting test data into data loader...")
    test_dataset = NumpyDataset(X_test, y_test)
 
    assert len(X_test) == len(y_test)

    logging.info("Applying model to {} examples...".format(len(y_test)))
    y_predicted_probability = np.zeros(len(y_test))
    for i in range(0, len(test_dataset), batch_size):
       logging.debug("Processed {} of {}".format(i, len(test_dataset)))
       i1 = min(i + batch_size, len(test_dataset))
       n_examples = i1 - i
       X_example, y_example = test_dataset[i:i1]
       y_predicted_probability[i:i1] = model.forward(X_example).cpu().detach().numpy()[0:n_examples, 0]
    # Loop
    y_predicted = (y_predicted_probability > 0.5)*1
    logging.info("Tabulating metrics for {}...".format(model_path))
    test_accuracy  = accuracy_score(y_test, y_predicted)
    test_roc_score = roc_auc_score(y_test, y_predicted_probability)
    [test_precision, test_recall, test_f_score, test_support] = precision_recall_fscore_support(y_test, y_predicted)
    precision, recall, thresholds = precision_recall_curve(y_test, y_predicted_probability)
    test_auc_precision_recall = auc(recall, precision)
    logging.info("Test results: precision-recall auc {}, accuracy {}, precision {}, recall {}, f-score {}, auc {}".format(
                 test_auc_precision_recall,
                 test_accuracy,
                 test_precision, test_recall, test_f_score, 
                 test_roc_score))

