#!/usr/bin/env python3
# Purpose: Trains the base models prior to (Multi)SWAG.
# Author: Ben Baker,SHiva Hari (University of Utah) distributed under the MIT license.
from cnn import CNN
from numpyDataset import NumpyDataset
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import numpy as np
import h5py
import pandas as pd
import argparse
import logging
import sys
import os

class Model():
    def __init__(self,
                 network,
                 optimizer,
                 device,
                 output_directory = 'models',
                 logger = None):
        if (not os.path.exists(output_directory)):
            raise Exception("{} does not exist".format(output_directory))
    
        self.device = device
        self.optimizer = optimizer
        self.network = network
        self.logger = logger
        if (self.logger is None):
            self.logger = logging.basicConfig(format = '%(asctime)s %(levelname)-8s %(message)s',
                                              level = logging.INFO,
                                              datefmt='%Y-%m-%d %H:%M:%S')
        self.output_directory = output_directory 

    def train(self, training_loader, validation_loader, n_epochs = 50, terminate_early = 5):
        if (training_loader is None):
            raise Exception("Training loader cannot be None")
        from torch.autograd import Variable
        self.network.to(device)
        self.network.train()
        ds = []
        loss = torch.nn.BCEWithLogitsLoss()
        n_training_examples = len(training_loader.dataset) 
        n_training_batches = len(training_loader)
        n_validation_examples = 0
        if (validation_loader is not None):
            n_validation_examples = len(validation_loader.dataset)
        d_best_epoch = {'epoch' :-1, 'score' : 0}

        for epoch in range(0, n_epochs + 1):
            logging.info("Starting epoch {}".format(epoch)) 

            # Loop
            self.network.train()
            y_observed = np.zeros(n_training_examples, dtype = 'int')
            y_predicted = np.zeros(n_training_examples, dtype = 'int')
            y_predicted_probability = np.zeros(n_training_examples, dtype = 'float') 
            example = 0
            for i, data in enumerate(training_loader, 0):
                if (i%500 == 0 and epoch > 0):
                    logging.debug("Beginning batch {} of {} for epoch {}...".format(i, len(training_loader), epoch))
                # Get inputs/outputs and wrap in variable object
                inputs_host, y_true = data
                inputs_device = Variable(inputs_host.to(self.device))
                n_batch_examples = len(inputs_device)
                y_true_device = Variable(y_true.to(self.device)) 

                # On the first go we establish the baseline
                if (epoch > 0):
                    # Set gradients for all parameters to zero
                    self.optimizer.zero_grad()

                    # Forward pass
                    outputs_device = self.network(inputs_device)

                    # Backward pass
                    loss_value = loss(outputs_device, y_true_device)
                    loss_value.backward()

                    # Update parameters
                    self.optimizer.step()
                else:
                    outputs_device = self.network(inputs_device)

                # Update statistics
                outputs_host = torch.sigmoid(outputs_device).cpu().detach().numpy()
                i1 = example
                i2 = example + n_batch_examples
                y_observed[i1:i2] = (y_true.numpy()[:, 0] > 0.5)*1
                y_predicted[i1:i2] = (outputs_host[:, 0] > 0.5)*1
                y_predicted_probability[i1:i2] = outputs_host[:, 0]
                example = example + n_batch_examples
                #if (i > 600):
                #    break
            # Loop on batches in training
            logging.debug("Tabulating training metrics...")
            training_accuracy  = accuracy_score(y_observed, y_predicted)
            training_roc_score = roc_auc_score(y_observed, y_predicted_probability)
            precision, recall, thresholds = precision_recall_curve(y_observed, y_predicted_probability)
            training_auc_precision_recall = auc(recall, precision)
            [training_precision, training_recall, training_f_score, training_support] \
                 = precision_recall_fscore_support(y_observed, y_predicted)
            logging.info("Epoch {}; training accuracy {}".format(epoch, training_accuracy))

            # Tabulate validation scores
            self.network.eval()
            validation_accuracy = None
            validation_roc_score = None
            validation_recall = None
            validation_precision = None
            validation_f_score = None
            validation_auc_precision_recall = None
            if (validation_loader is not None):
                logging.info("Evaluating model on validation data...")
                self.network.eval()
                y_observed = np.zeros(n_validation_examples, dtype = 'int')
                y_predicted = np.zeros(n_validation_examples, dtype = 'int')
                y_predicted_probability = np.zeros(n_validation_examples, dtype = 'float')
                example = 0
                for i, data in enumerate(validation_loader, 0):
                    # Get inputs/outputs and wrap in variable object
                    inputs, y_true = data
                    inputs_device = Variable(inputs.to(self.device))
                    n_batch_examples = len(inputs_device)
                    y_true_device = Variable(y_true.to(self.device)) 
 
                    # Forward pass
                    outputs_device = self.network(inputs_device)

                    # Extract results into arrays
                    outputs_host = outputs_device.cpu().detach().numpy()
                    i1 = example
                    i2 = example + n_batch_examples
                    y_observed[i1:i2] = (y_true.numpy()[:, 0] > 0.5)*1
                    y_predicted[i1:i2] = (outputs_host[:, 0] > 0.5)*1
                    y_predicted_probability[i1:i2] = outputs_host[:, 0]
                    example = example + n_batch_examples
                # Loop on validation batches
                logging.debug("Tabulating validation metrics...")
                validation_accuracy  = accuracy_score(y_observed, y_predicted)
                validation_roc_score = roc_auc_score(y_observed, y_predicted_probability)
                [validation_precision, validation_recall, validation_f_score, validation_support] \
                    = precision_recall_fscore_support(y_observed, y_predicted)
                precision, recall, thresholds = precision_recall_curve(y_observed, y_predicted_probability)
                validation_auc_precision_recall = auc(recall, precision)
                logging.info("Validation result for epoch {}; precision-recall auc {}, accuracy {}, precision {}, recall {}, f-score {}, auc {}".format(
                             epoch,
                             validation_auc_precision_recall,
                             validation_accuracy,
                             validation_precision, validation_recall, validation_f_score, 
                             validation_roc_score))
            # End check on validation

            # Get model off device and onto disk
            if (epoch > 0):
                file_name = os.path.join(self.output_directory,
                                         'model_%d.pt'%(epoch))
                logging.info("Writing torch script file {}".format(file_name))
                jit_module = torch.jit.trace(self.network.to('cpu').forward,
                                             torch.zeros(1,
                                                         self.network.get_input_channels(),
                                                         self.network.get_input_height(),
                                                         self.network.get_input_width()))
                jit_module.save(file_name)
                self.network.to(self.device)
 
            # Update the convergence dictionary
            d = {'epoch': epoch,
                 'training_accuracy': training_accuracy,
                 'training_roc': training_roc_score,
                 'training_blast_precision': training_precision[0],
                 'training_blast_recall': training_recall[0],
                 'training_blast_f_score': training_f_score[0],
                 'training_earthquake_precision': training_precision[1],
                 'training_earthquake_recall': training_recall[1],
                 'training_earthquake_f_score': training_f_score[1],
                 'training_precision_recall_auc': training_auc_precision_recall,
                 'validation_accuracy' : validation_accuracy,
                 'validation_blast_roc': validation_roc_score,
                 'validation_blast_precision': validation_precision[0],
                 'validation_blast_recall': validation_recall[0],
                 'validation_blast_f_score': validation_f_score[0],
                 'validation_earthquake_precision': validation_precision[1],
                 'validation_earthquake_recall': validation_recall[1],
                 'validation_earthquake_f_score': validation_f_score[1],
                 'validation_precision_recall_auc': validation_auc_precision_recall
                }
            ds.append(d) 
           
            # Note the best model
            if (validation_auc_precision_recall is not None):
                if (validation_auc_precision_recall > d_best_epoch['score']):
                    d_best_epoch = {'epoch' : epoch, 'score' : validation_auc_precision_recall}
                if (epoch - d_best_epoch['epoch'] > terminate_early):
                    logging.info("Model no longer making progress on validation dataset; terminating early...")
            else:
                if (training_accuracy > d_best_epoch['score']):
                    d_best_epoch = {'epoch' : epoch, 'score' : training_accuracy}
        # Loop on epochs
        logging.info("Training finished.  Best epoch was {}".format(d_best_epoch['epoch']))
        convergence_df = pd.DataFrame(ds)
        convergence_df.to_csv(os.path.join(self.output_directory, 'convergence.csv'), index = False)
        
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = '''
Trains the base CNN model for earthquake/blast discrimination.\n

''',
                                   formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--device_name',
                        type = str,
                        help = 'the name of the device on which to train - e.g., cpu or cuda:0',
                        default = 'cuda:0')
    parser.add_argument('--learning_rate',
                        type = float,
                        help = 'the learning rate',
                        default = 4.e-5)
    parser.add_argument('--epochs',
                        type = int,
                        help = 'the number of epochs for training',
                        default = 20)
    parser.add_argument('--training_file',
                        type = str,
                        help = 'the name of the HDF5 archive with the training data',
                        default = os.path.join('data', 'train.h5')) 
    parser.add_argument('--validation_file',
                        type = str,
                        help = 'the name of the HDF5 archive with the validation data',
                        default = os.path.join('data', 'validate.h5'))
    parser.add_argument('--output_directory',
                        type = str,
                        help = 'the directory to which torch script model files will be written',
                        default = 'models')
    parser.add_argument('--seed',
                        type = int,
                        help = 'the random seed so for reproducible results',
                        default = 82323)
    parser.add_argument('--terminate_early',
                        type = int,
                        help = 'the optimization will terminate early if the best validation ROC score does not decrease after this many epochs',
                        default = 5)
    parser.add_argument('-b', '--batch_size',
                        type = int,
                        help = 'the batch size',
                        default = 16)
    parser.add_argument('-v', '--verbosity',
                        type = int,
                        help = 'controls the verbosity (1 errors/warnings, 2 errors/warnings/info, 3 errors/warnings/info/debug)',
                        default = 2)
    parser.add_argument('--log_file',
                        type = str,
                        help = 'the name of the log file; by default this is standard out',
                        default = None)
    args = parser.parse_args()

    # Create the logger
    log_level = logging.CRITICAL
    if (args.verbosity == 2):
        log_level = logging.INFO
    elif (args.verbosity >= 3):
        log_level = logging.DEBUG
    else:
        log_level = logging.CRITICAL
    if (args.log_file is None):
        logging.basicConfig(format = '%(asctime)s %(levelname)-8s %(message)s',
                            level = log_level,
                            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(filename = args.log_file,
                            filemode = 'w',
                            format = '%(asctime)s %(levelname)-8s %(message)s',
                            level = log_level,
                            datefmt='%Y-%m-%d %H:%M:%S')
    # Unpack the roptions
    device_name = args.device_name
    n_epochs = args.epochs
    if (n_epochs < 1):
        sys.exit("Number of epochs must be positive")
    learning_rate = args.learning_rate
    if (learning_rate <= 0):
        sys.exit("Learning rate must be positive")
    batch_size = args.batch_size
    if (batch_size < 2):
        sys.exit("Batch size must be at least 2 because of batch-normalization")
    terminate_early = args.terminate_early
    if (terminate_early < 1):
        sys.exit("Early termination must be positive")
    output_directory = args.output_directory
    if (not os.path.exists(output_directory)):
        os.makedirs(output_directory)
    assert os.path.exists(output_directory)
    training_file = args.training_file
    if (not os.path.exists(training_file)):
        sys.exit("Training file {} does not exist".format(training_file))
    validation_file = args.validation_file
    if (not os.path.exists(validation_file)):
        logging.warning("Validation file {} does not exist".format(validation_file))

    try:
        logging.debug("Will train on {}".format(device_name))
        device = torch.device(device_name)
    except:
        sys.exit("Could not load device {}".format(device_name))
    np.random.seed(args.seed)

    # Initialize convolutional neural network stock parameters
    logging.info("Initializing CNN...")
    cnn = CNN(random_seed = args.seed)
    logging.info("Number of trainable parameters: {}".format(cnn.get_total_number_of_parameters()))

    logging.info("Initializing Adam optimizer with learning rate {}".format(learning_rate))
    optimizer = torch.optim.Adam(cnn.parameters(), lr = learning_rate)

    logging.info("Loading data from training file {}".format(training_file))
    with h5py.File(training_file, 'r') as f:
        X_train = f['X'][:]
        y_train = f['y'][:]
        assert len(X_train) == len(y_train)
    logging.info("Number of training examples {}".format(len(X_train)))
    logging.info("There {} training earthquakes and {} training blasts".format(
                 np.sum(y_train == 1), np.sum(y_train == 0)))
    training_dataset = NumpyDataset(X_train, y_train)
    # Ensure there's no pathological ordering 
    training_indices = np.random.choice(np.arange(0, len(X_train)),
                                        size = len(X_train),
                                        replace = False)
    training_sampler = torch.utils.data.sampler.SubsetRandomSampler(training_indices)
    training_loader = torch.utils.data.DataLoader(training_dataset,
                                                  batch_size = batch_size,
                                                  shuffle = False,
                                                  sampler = training_sampler)

 
    validation_loader = None
    if (os.path.exists(validation_file)):
        logging.info("Loading data from validation file {}".format(validation_file))
        with h5py.File(validation_file, 'r') as f:
            X_validate = f['X'][:]
            y_validate = f['y'][:]
            assert len(X_validate) == len(y_validate)
        logging.info("Number of validation examples {}".format(len(X_validate)))
        logging.info("There {} validation earthquakes and {} validation blasts".format( 
                     np.sum(y_validate == 1), np.sum(y_validate == 0)))
        validation_dataset = NumpyDataset(X_validate, y_validate)
        validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                        batch_size = batch_size,
                                                        shuffle = False)

   
    logging.info("Initializing model for training...")
    model = Model(network = cnn,
                  optimizer = optimizer,
                  device = device,
                  output_directory = output_directory,
                  logger = logging)
    model.train(training_loader = training_loader,
                validation_loader = validation_loader,
                n_epochs = n_epochs,
                terminate_early = terminate_early) 
