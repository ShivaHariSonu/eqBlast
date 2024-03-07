#!/usr/bin/env python3
# Purpose: Splits the larger data into a training, validation, and testing dataset.
# Author: Ben Baker, Shiva Harigit  (UUSS) distributed under the MIT license.
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import argparse
import h5py
import os
import sys

def downsample_majority(events : np.array, df : pd.DataFrame):
    print("Downsampling majority class...")
    # Get the indices of earthquakes
    blast_example_rows = []
    earthquake_example_rows = []
    for event in events:
        temp_df = df[ (df['evid'] == event) ]
        if (temp_df.iloc[0]['etype'] == 'qb'):
            if (len(blast_example_rows) == 0):
                blast_example_rows = temp_df['row_index'].to_numpy()
            else:
                blast_example_rows = np.append(blast_example_rows, temp_df['row_index'].to_numpy())
        else:
            if (len(earthquake_example_rows) == 0):
                earthquake_example_rows = temp_df['row_index'].to_numpy()
            else:
                earthquake_example_rows = np.append(earthquake_example_rows, temp_df['row_index'].to_numpy())
    # Loop
    n_earthquake_examples = len(earthquake_example_rows)
    n_blast_examples = len(blast_example_rows)
    assert n_blast_examples + n_earthquake_examples == len(df)
    assert n_earthquake_examples > n_blast_examples
    # Downsample majority class
    new_earthquake_rows = np.random.choice(earthquake_example_rows, n_blast_examples, replace = False) 
    new_earthquake_rows = np.sort(new_earthquake_rows)
    new_rows = np.sort( np.append(blast_example_rows, new_earthquake_rows) )
    return df[ np.isin(df['row_index'], new_rows) ]

def write_h5(df : pd.DataFrame, X : np.array, output_file_name):
    original_row_indices = df['row_index'].values
    y = np.ndarray.astype(np.asarray(df['etype'] == 'eq')*1, 'int')
    assert len(y) == len(original_row_indices)
    ofl = h5py.File(output_file_name, 'w')
    ofl['X'] = X[original_row_indices, :, :]
    ofl['y'] = y
    ofl.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = '''
Splits the data into a training, validation, and test dataset.\n

Example to do a 70/10/20 pct train/validate/test split on the data:\n
    split.py --hdf5 pSpectrograms.h5 --metadata pSpectrograms.csv --train_pct=70 --test_pct=20\n
''',
                                   formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--hdf5',
                        type = str,
                        help = 'the unsplit HDF5 file containing the frequency-time representation of every channel',
                        default = 'pSpectrograms.h5')
    parser.add_argument('--metadata',
                        type = str,
                        help = 'the lightweight data describing each example in the HDF5 file.  This should be in .csv format',
                        default = 'pSpectrograms.csv')
    parser.add_argument('--output_directory',
                        type = str,
                        help = 'the directory to which the training, validation, and test files will be written',
                        default = os.path.curdir)
    parser.add_argument('--train_pct',
                        type = float,
                        help = 'the percentage of data to put into the training dataset',
                        default = 70)
    parser.add_argument('--test_pct',
                        type = float,
                        help = 'the percentage of data to put into the testing dataset',
                        default = 20)
    parser.add_argument('--seed',
                        type = int,
                        help = 'the random seed for reproducibility purposes',
                        default = 293232)
    args = parser.parse_args()

    # Set the seed
    np.random.seed(args.seed)

    # Ensure the input files exist
    hdf5_file = args.hdf5
    if (not os.path.exists(hdf5_file)):
        sys.exit("HDF5 file {} does not exist".format(hdf5_file))
    csv_file = args.metadata
    if (not os.path.exists(csv_file)):
        sys.exit("Metadata file {} does not exist".format(csv_file))

    # Ensure the output directory exists
    output_directory = args.output_directory
    if (not os.path.exists(output_directory)):
        os.makedirs(output_directory)
    assert os.path.exists(output_directory), 'output directory {} does not exist'.format(output_directory)
    # Deduce the percentages
    train_pct = args.train_pct
    if (train_pct < 0 or train_pct > 100):
        sys.exit("Training percentage {} must be in range [0,100]".format(train_pct))
    test_pct = args.test_pct
    if (test_pct < 0 or test_pct > 100):
        sys.exit("Testing percentage {} must be in range [0,100]".format(test_pct))
    if (train_pct + test_pct > 100):
        sys.exit("Training + test perectnage must be in range [0,100]")
    validation_pct = 100 - train_pct - test_pct
    assert abs(train_pct + validation_pct + test_pct - 100) < 1.e-10
    
    print("Loading H5 file: {}...".format(hdf5_file))
    print("Loading metadata file: {}...".format(csv_file))
    df = pd.read_csv(csv_file)
    df.sort_values(['evid'], inplace = True)
    df['row_index'] = np.arange(len(df))

    # Split the data eventwise
    events = np.unique(df['evid'])
    training_events, other_events = train_test_split(events, train_size = train_pct/100.)
    new_test_pct = test_pct/(100 - train_pct)
    assert new_test_pct >= 0 and new_test_pct <= 100
    validation_events, testing_events = train_test_split(other_events, test_size = new_test_pct)
    assert len(training_events) + len(validation_events) + len(testing_events) == len(events), 'missed some events'

    # Allow us to read from h5 in order through this in order
    training_events   = np.sort(training_events)
    validation_events = np.sort(validation_events)
    testing_events    = np.sort(testing_events)
 
    print("Number of training events:", len(training_events))
    print("Number of validation events:", len(validation_events))
    print("Number of testing events:", len(testing_events))
    training_df   = df[ np.isin(df['evid'], training_events)  ]
    # Downsample the training events
    training_df   = downsample_majority(training_events, training_df)
    validation_df = df[ np.isin(df['evid'], validation_events) ]
    testing_df    = df[ np.isin(df['evid'], testing_events) ]
    print("Number of training examples (eq,blast):", len(training_df), np.sum(training_df['etype'] == 'eq'), np.sum(training_df['etype'] == 'qb'))
    print("Number of validation examples:", len(validation_df), np.sum(validation_df['etype'] == 'eq'), np.sum(validation_df['etype'] == 'qb'))
    print("Number of testing examples:", len(testing_df), np.sum(testing_df['etype'] == 'eq'), np.sum(testing_df['etype'] == 'qb'))
    assert len(np.intersect1d(np.unique(training_df['evid']), np.unique(testing_df['evid']))) == 0
    assert len(np.intersect1d(np.unique(training_df['evid']), np.unique(validation_df['evid']))) == 0
    assert len(np.intersect1d(np.unique(testing_df['evid']),  np.unique(validation_df['evid']))) == 0

    print("Loading dataset...")
    h5 = h5py.File(hdf5_file, 'r')
    X = h5['X'][:]
    h5.close()
 
    print("Writing training dataset...")
    write_h5(training_df,   X, os.path.join(output_directory, 'train.h5'))
    training_df.to_csv(os.path.join(output_directory, 'train.csv'), index = False)
    print("Writing validation dataset...")
    write_h5(validation_df, X, os.path.join(output_directory, 'validate.h5'))
    validation_df.to_csv(os.path.join(output_directory, 'validate.csv'), index = False)
    print("Writing testing dataset...")
    write_h5(testing_df,    X, os.path.join(output_directory, 'test.h5'))
    testing_df.to_csv(os.path.join(output_directory, 'test.csv'), index = False)
