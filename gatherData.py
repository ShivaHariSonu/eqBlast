#!/usr/bin/env python3
import sys
sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/waveformArchive/gcc_build')
#sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/templateMatchingSource/rtseis/notchpeak4_gcc83_build/')
sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/mlmodels/intel_cpu_build/')
import h5py
import pyWaveformArchive as pwa
#import libpyrtseis as rtseis
import pyuussmlmodels as uuss
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_vertical_channel_data(waveform, pick_time, trace_cut_start =-2,
                                  processor = uuss.EventClassifiers.CNNThreeComponent.Preprocessing(),
                                  do_plot = False):
    """ 
    Applies basic processing and interpolation of input signals.

    Parameters
    ----------
    waveform : waveform archive object
       The vertical waveform to preprocess.
    pick_time : double
       The (P) pick time in UTC seconds since the epoch.
    trace_cut_start : double
       The seconds before the (P) pick time to begin the cut window.

    Returns
    -------
    waveform : waveform archive object
       The filtered waveform.  Note, this can be none if an error
       was encountered or the input signal is too small.
    """
    signal = np.copy(waveform.signal)
    if (len(signal) == 0):
        print("Signal is empty")
        return None, None, None
    if (len(np.unique(signal)) == 1):
        print("Signal is uniform")
        return None, None, None
    if (waveform.start_time > pick_time + trace_cut_start):
        print("Cannot cut trace early enough")
        return None, None, None
    i0 = int( (pick_time + trace_cut_start - waveform.start_time)*waveform.sampling_rate )
    if (i0 < 0 or i0 >= len(signal) - 1):
        print("Start index out of bounds")
        return None, None, None
    i1 = len(signal)
    signal = np.copy(signal[i0:i1])
    if (do_plot):
        plt.plot(signal)
        plt.show()
    z_spectrogram = processor.process_vertical_channel(signal, waveform.sampling_rate)
    z_spectrogram = np.ndarray.astype(z_spectrogram, np.float32)
    n_spectrogram = np.zeros(z_spectrogram.shape, dtype = np.float32)
    e_spectrogram = np.zeros(z_spectrogram.shape, dtype = np.float32)
    return z_spectrogram, n_spectrogram, e_spectrogram

def process_three_component_data(z_waveform,
                                 n_waveform,
                                 e_waveform,
                                 pick_time, trace_cut_start =-2,
                                 processor = uuss.EventClassifiers.CNNThreeComponent.Preprocessing(),
                                 do_plot = False):
    z_signal = np.copy(z_waveform.signal)
    n_signal = np.copy(n_waveform.signal)
    e_signal = np.copy(e_waveform.signal)
    if (len(z_signal) == 0 or len(n_signal) == 0 or len(e_signal) == 0): 
        return None, None, None
    if (len(np.unique(z_signal)) == 1):
        print("Z signal is uniform")
        return None, None, None
    if (len(np.unique(n_signal)) == 1):
        print("N signal is uniform")
        return None, None, None
    if (len(np.unique(e_signal)) == 1):
        print("E signal is uniform")
        return None, None, None
    if (z_waveform.start_time > pick_time + trace_cut_start or
        n_waveform.start_time > pick_time + trace_cut_start or
        e_waveform.start_time > pick_time + trace_cut_start):
        print("Cannot cut trace early enough")
        return None, None, None
    if (abs(z_waveform.sampling_rate - n_waveform.sampling_rate) > 1.e-8 or
        abs(z_waveform.sampling_rate - e_waveform.sampling_rate) > 1.e-8):
        print("Inconsistent sampling rates")
        return None, None, None
    i0z = int( (pick_time + trace_cut_start - z_waveform.start_time )*z_waveform.sampling_rate )
    i0n = int( (pick_time + trace_cut_start - n_waveform.start_time )*n_waveform.sampling_rate )
    i0e = int( (pick_time + trace_cut_start - e_waveform.start_time )*e_waveform.sampling_rate )
    if (i0z < 0 or i0z >= len(z_signal) - 1 or
        i0n < 0 or i0n >= len(n_signal) - 1 or
        i0e < 0 or i0e >= len(e_signal) - 1):
        print("Start indices out of bounds")
        return None, None, None
    i1 = min( len(z_signal) - i0z, len(n_signal) - i0n, len(e_signal) - i0e )
    z_signal = np.copy(z_signal[i0z:i0z + i1])
    n_signal = np.copy(n_signal[i0n:i0n + i1])
    e_signal = np.copy(e_signal[i0e:i0e + i1])
    if (do_plot):
        plt.plot(z_signal)
        plt.plot(n_signal)
        plt.plot(e_signal)
        plt.show()
    z_spectrogram, n_spectrogram, e_spectrogram \
        = processor.process(z_signal, n_signal, e_signal, z_waveform.sampling_rate)
    z_spectrogram = np.ndarray.astype(z_spectrogram, np.float32)
    n_spectrogram = np.ndarray.astype(n_spectrogram, np.float32)
    e_spectrogram = np.ndarray.astype(e_spectrogram, np.float32)
    return z_spectrogram, n_spectrogram, e_spectrogram



if __name__ == "__main__":
    #print(dir(rtseis.PostProcessing.Waveform))
    archive_dir = '/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/waveformArchive/archives/'
    h5_archive_files = glob.glob(archive_dir + '/archive_????.h5')
    catalog_dir = '/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/waveformArchive/data/'
    output_file_base = 'pSpectrograms'
    trace_cut_start =-5 # Start traces 5 seconds before P pick time
    processor = uuss.EventClassifiers.CNNThreeComponent.Preprocessing()
    chunk_size = 128 # Write every 128...
    #n_samples = processor.scalogram_length
    #n_scales = processor.number_of_scales
    n_samples = processor.number_of_time_windows
    n_frequencies = processor.number_of_frequencies
    #print("Output shape will be 3 x {} x {}".format(n_samples, n_scales))
    print("Output shape will be 3 x {} x {}".format(n_samples, n_frequencies))

    current_eq_catalog_1c = os.path.join(catalog_dir, 'currentEarthquakeArrivalInformation1C.csv')
    current_eq_catalog_3c = os.path.join(catalog_dir, 'currentEarthquakeArrivalInformation3C.csv')
    current_blast_catalog_1c = os.path.join(catalog_dir, 'currentBlastArrivalInformation1C.csv')
    current_blast_catalog_3c = os.path.join(catalog_dir, 'currentBlastArrivalInformation3C.csv')

    # Load the 1C blast/earthquake catalogs and concatentate
    print("Loading 1C catalog...")
    df_blast = pd.read_csv(current_blast_catalog_1c, dtype = {'location' : str} )
    df_blast['etype'] = 'qb'
    df_eq = pd.read_csv(current_eq_catalog_1c,       dtype = {'location' : str} )
    df_eq['etype'] = 'eq' 
    df_1c = pd.concat( [df_blast, df_eq] )
    df_1c = df_1c[ (df_1c['event_lat'] < 43) & (df_1c['phase'] == 'P') ] # Make specific to Utah
    df_1c['channel1'] = None
    df_1c['channel2'] = None

    # Load the 3C blast/earthquake catalogs and concatenate
    print("Loading 3C catalog...")
    df_blast = pd.read_csv(current_blast_catalog_3c, dtype = {'location' : str})
    df_blast['etype'] = 'qb'
    df_eq = pd.read_csv(current_eq_catalog_3c,       dtype = {'locaiton' : str})
    df_eq['etype'] = 'eq'
    df_3c = pd.concat( [df_blast, df_eq] )
    df_3c = df_3c[ (df_3c['event_lat'] < 43) & (df_3c['phase'] == 'P') ] # Make specific to Utah
    
    # Merge the 1C and 3C waveforms and sort on evid and arrival time
    df = pd.concat([df_1c, df_3c])
    df.sort_values(['evid', 'arrival_time'], inplace = True)

    # We really want unique stations here.  Basically, a station can have a P 
    # and S pick on the same waveform or on different channels on different
    # sensors.  The logic is such that if the first arriving pick was good
    # enough to be picked (see sorting above) then that SNCL is good enough
    # for classification purporses.
    df['row_index'] = np.arange(0, len(df))
    print("Dropping duplicate waveforms...")
    keepers = np.zeros(len(df), dtype = 'bool') + True 
    evids = np.unique(df['evid'])
    for evid in evids:
        temp_df = df[ df['evid'] == evid ]
        networks = temp_df['network'].values
        stations = temp_df['station'].values
        #channels = temp_df['channelz'].values
        #locations = temp_df['location'].values
        rows = temp_df['row_index'].values
        for i in range(len(networks)):
            if (not keepers[rows[i]]):
                continue
            for j in range(i + 1, len(networks)):
                if (networks[i] == networks[j] and
                    stations[i] == stations[j]):
                    #channels[i] == channels[j] and
                    #locations[i] == locations[j]
                    keepers[rows[j]] = False
                    break
        # Loop on temp_df
    # Loop on evids
    print("Initially had {} rows".format(len(df)))
    df = df[keepers]
    print("Retaining {} rows of which {} are earthquakes and {} are blasts".format(
          len(df), np.sum(df['etype'] == 'eq'), np.sum(df['etype'] == 'qb') ))

    # Now create the archives
    print("Opening archive files for reading...")
    archive_manager = pwa.ArchiveManager()
    archive_manager.open_files_for_reading(h5_archive_files)

    ofl = h5py.File(output_file_base + '.h5', 'w')
    #dset_x = ofl.create_dataset("X", (0, n_samples, n_scales, 3), 
    #                           maxshape=(None, n_samples, n_scales, 3)) 
    #chunk_x = np.zeros([chunk_size, n_samples, n_scales, 3], dtype = 'float')
    dset_x = ofl.create_dataset("X", (0, 3, n_samples, n_frequencies), 
                               maxshape=(None, 3, n_samples, n_frequencies)) 
    chunk_x = np.zeros([chunk_size, 3, n_samples, n_frequencies], dtype = 'float')
    y = []


    k = 0
    was_saved = np.zeros(len(df), dtype = 'bool')
    for irow in range(len(df)):
        # Flush chunk of data to disk
        if (k == chunk_size):
            orig_index = dset_x.shape[0]
            dset_x.resize(dset_x.shape[0] + chunk_x.shape[0], axis=0)
            dset_x[orig_index:, :] = chunk_x
            k = 0

        row = df.iloc[irow]
        is3c = False
        if (row['channel1'] is not None and row['channel2'] is not None):
            is3c = True
        # Always need the vertical channel
        exists = archive_manager.waveform_exists(row['evid'],
                                                 row['network'],  row['station'],
                                                 row['channelz'], row['location'])
        if (not exists):
            continue
        z_waveform = archive_manager.read_waveform(row['evid'],
                                                   row['network'],  row['station'],
                                                   row['channelz'], row['location'])
        z_waveform.remove_trailing_zeros() # Clean up any junk at end of waveform
        z_spectrogram = None
        n_spectrogram = None
        e_spectrogram = None
        if (is3c):
            exists = archive_manager.waveform_exists(row['evid'],
                                                     row['network'],  row['station'],
                                                     row['channel1'], row['location'])
            if (not exists):
                continue
            exists = archive_manager.waveform_exists(row['evid'],
                                                     row['network'],  row['station'],
                                                     row['channel2'], row['location'])
            if (not exists):
                continue
            n_waveform = archive_manager.read_waveform(row['evid'],
                                                       row['network'],  row['station'],
                                                       row['channel1'], row['location'])
            n_waveform.remove_trailing_zeros() # Clean up any junk at end of waveform
            e_waveform = archive_manager.read_waveform(row['evid'],
                                                       row['network'],  row['station'],
                                                       row['channel2'], row['location'])
            e_waveform.remove_trailing_zeros() # Clean up any junk at end of waveform
            # Process waveforms
            try:
                z_spectrogram, n_spectrogram, e_spectrogram \
                    = process_three_component_data(z_waveform, n_waveform, e_waveform,
                                                   row['arrival_time'],
                                                   trace_cut_start = trace_cut_start,
                                                   processor = processor)
            except Exception as e:
                print("Failed to process three-component waveforms", e)
                continue 
            if (z_spectrogram is None or n_spectrogram is None or e_spectrogram is None):
                print("Failed to process waveforms")
                continue
            if (np.amax(z_spectrogram) == 0 or np.amax(n_spectrogram) == 0 or np.amax(e_spectrogram) == 0):
                print("Dead channel")
                continue
        else:
            # Process waveform
            try:
                z_spectrogram, n_spectrogram, e_spectrogram \
                    = process_vertical_channel_data(z_waveform, row['arrival_time'],
                                                    trace_cut_start = trace_cut_start,
                                                    processor = processor)
            except Exception as e:
                print("Failed to process vertical waveform", e)
                continue
            if (z_spectrogram is None):
                print("Failed to process vertical waveform")
                continue
            if (np.amax(z_spectrogram) == 0):
                print("Dead vertical channel")
                continue
        if (z_spectrogram is None or n_spectrogram is None or e_spectrogram is None):
            print("Failed to process waveforms") 
            continue
        if (np.amax(np.isnan(z_spectrogram))):
            print("Z-spectrogram unusable because of", np.amax(z_spectrogram))
            continue
        if (np.amax(np.isnan(n_spectrogram))):
            print("N-spectrogram unusable")
            continue
        if (np.amax(np.isnan(e_spectrogram))):
            print("E-spectrogram unusable")
            continue

        chunk_x[k, 0, :, :] = z_spectrogram[:, :]
        chunk_x[k, 1, :, :] = n_spectrogram[:, :]
        chunk_x[k, 2, :, :] = e_spectrogram[:, :]
        if (row['etype'] == 'eq'):
            y.append(1)
        else:
            y.append(0)

        k = k + 1
        was_saved[irow] = True
        
        #if (irow > 1024):
        #    break
    # Loop
    archive_manager.close()
    # Flush last chunk
    y = np.asarray(y, dtype = 'int')
    if (k > 0): 
        orig_index = dset_x.shape[0]
        dset_x.resize(dset_x.shape[0] + k, axis=0) #chunk_.shape[0], axis=0)
        dset_x[orig_index:, :] = chunk_x[0:k, :]
        k = 0 
    ofl['y'] = y
    # Close it up and finish
    ofl.close()
    df_out = df[was_saved]
    df_out.to_csv(output_file_base + '.csv', index = False)
    assert len(y) == len(df_out), 'bad output shapes'
    print("Finishing with {} P waveform examples of which {} are earthquakes and {} are blasts".format(
          len(df_out), np.sum(df_out['etype'] == 'eq'), np.sum(df_out['etype'] == 'qb') ))

