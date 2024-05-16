import os
import shutil
import random
import json
import pyarrow.parquet as pq
import numpy as np
import h5py
import math
import argparse

def estimate_population_parameters(all_sample_sizes, all_sample_means, all_sample_stds):
    population_means = []
    population_stds = []
    for j in range(len(all_sample_means)):
        sample_means = all_sample_means[j]
        sample_stds = all_sample_stds[j]
        sample_sizes = all_sample_sizes[j]
        sample_means = sample_means[sample_sizes != 0]
        sample_stds = sample_stds[sample_sizes != 0]
        sample_sizes = sample_sizes[sample_sizes != 0]
        weighted_sum_of_variances = sum((n - 1) * s**2 for n, s in zip(sample_sizes, sample_stds))
        total_degrees_of_freedom = sum(n - 1 for n in sample_sizes)
        combined_variance = weighted_sum_of_variances / total_degrees_of_freedom
        population_std = np.sqrt(combined_variance)
        weighted_sum_of_means = sum(n * mean for n, mean in zip(sample_sizes, sample_means))
        total_observations = sum(sample_sizes)
        population_mean = weighted_sum_of_means / total_observations
        population_stds.append(population_std)
        population_means.append(population_mean)

    return population_means, population_stds

def main():
    data = h5py.File(args.analysis_dir + args.hdf5_name, 'r')
    print("data size: ", data["all_jet"].shape[0])
    size_ = []
    mean_ = []
    std_ = []

    for i in range(len(data["all_jet"])):
        im = data["all_jet"][i,:,:,:]
        im[im < 1.e-5] = 0
        size_channel = []
        mean_channel = []
        std_channel = []
        bad_channel = False
        for j in range(args.channels):
            if not bad_channel:
                image = im[j,:,:]
                image = image[image != 0]
                if len(image) < 2:
                    bad_channel = True
                    continue
                size_channel.append(len(image))
                mean_channel.append(image.mean())
                std_channel.append(image.std(ddof=1))
        if not bad_channel:
            size_.append(size_channel)
            mean_.append(mean_channel)
            std_.append(std_channel)

    size_ = np.array(size_).T
    mean_ = np.array(mean_).T
    std_ = np.array(std_).T
    print("size: ", size_.shape)
    print("mean: ", mean_.shape)
    print("std : ", std_.shape)
    orig_mean, orig_std = estimate_population_parameters(size_, mean_, std_)


    print('Means with outliers:', orig_mean)
    print('Stds with outliers :', orig_std)

    size_ = []
    mean_ = []
    std_ = []

    for i in range(len(data["all_jet"])):
        bad_channel = False
        im = data["all_jet"][i,:,:,:]
        im[im < 1.e-5] = 0
        size_channel = []
        mean_channel = []
        std_channel = []
        for j in range(args.channels):
            if not bad_channel:
                image = im[j,:,:]
                image = image[image != 0]
                if len(image) < 2 or np.mean(image) > orig_mean[j] + 6 * orig_std[j]:
                    bad_channel = True
                    continue
                size_channel.append(len(image))
                mean_channel.append(image.mean())
                std_channel.append(image.std(ddof=1))
        if not bad_channel:
            size_.append(size_channel)
            mean_.append(mean_channel)
            std_.append(std_channel)


    size_ = np.array(size_).T
    mean_ = np.array(mean_).T
    std_ = np.array(std_).T
    rem_before_mean, rem_before_std = estimate_population_parameters(size_, mean_, std_)

    # with h5py.File(args.analysis_dir + args.proper_norm_hdf5_name, 'w') as proper_data:
    #     dataset_names = ['train_jet', 'valid_jet', 'test_jet', 'train_meta', 'valid_meta', 'test_meta',]
    #     samples = size_.shape[1]
    #     train_indx = samples * .8 // 1
    #     valid_indx = train_indx + samples * .1 // 1
    #     dataset_sizes = [train_indx, valid_indx - train_indx, samples - valid_indx, train_indx, valid_indx - train_indx, samples - valid_indx]
    #     datasets = {
    #         name: proper_data.create_dataset(
    #             name,
    #             (size, args.channels, 125, 125) if 'jet' in name else (size, 1),
    #             dtype='float32',  # Specify an appropriate data type
    #             compression='lzf',
    #             chunks=(1, args.channels, 125, 125) if 'jet' in name else (1, 1),
    #         ) for name, size in zip(dataset_names, dataset_sizes)
    #     }
    #
    #     current_index = 0
    #     set = 'train'
    #     for i in range(len(data["all_jet"])):
    #         bad_channel = False
    #         im = data["all_jet"][i,:,:,:]
    #         label = data["all_meta"][i]
    #         im[im < 1.e-5] = 0
    #         size_channel = []
    #         mean_channel = []
    #         std_channel = []
    #         if set == 'train' and current_index == train_indx:
    #             set = 'valid'
    #             current_index = 0
    #         if set == 'valid' and current_index == valid_indx - train_indx:
    #             set = 'test'
    #             current_index = 0
    #         for j in range(args.channels):
    #             if not bad_channel:
    #                 image = im[j,:,:]
    #                 nz = image[image != 0]
    #                 if len(nz) < 2 or np.mean(nz) > orig_mean[j] + 6 * orig_std[j]:
    #                     bad_channel = True
    #                     continue
    #                 image = (image - rem_before_mean[j]) / rem_before_std[j]
    #                 proper_data[set+'_jet'][current_index,j,:,:] = image
    #         if not bad_channel:
    #             proper_data[set+'_meta'][current_index] = label
    #             current_index += 1
    #
    #
    # # Initialize lists to hold mean and std values for signal and background
    # size_signal = []
    # mean_signal = []
    # std_signal = []
    # size_background = []
    # mean_background = []
    # std_background = []
    #
    # # Retrieve the labels from the 'all_meta' dataset (assuming the label is the last column)
    # labels = data["all_meta"][:, -1]
    #
    # for i in range(len(data["all_jet"])):
    #     im = data["all_jet"][i, :, :, :]
    #     label = labels[i]
    #     # Apply threshold
    #     im[im < 1.e-5] = 0
    #     size_channel = []
    #     mean_channel = []
    #     std_channel = []
    #     bad_channel = False
    #     for j in range(args.channels):
    #         if not bad_channel:
    #             image = im[j,:,:]
    #             image = image[image != 0]
    #             if len(image) < 2:
    #                 bad_channel = True
    #                 continue
    #             size_channel.append(len(image))
    #             mean_channel.append(image.mean())
    #             std_channel.append(image.std(ddof=1))
    #     if label == 1 and not bad_channel:  # Signal
    #         size_signal.append(size_channel)
    #         mean_signal.append(mean_channel)
    #         std_signal.append(std_channel)
    #     elif label == 0 and not bad_channel:  # Background
    #         size_background.append(size_channel)
    #         mean_background.append(mean_channel)
    #         std_background.append(std_channel)
    #
    # size_signal = np.array(size_signal).T
    # mean_signal = np.array(mean_signal).T
    # std_signal = np.array(std_signal).T
    # size_background = np.array(size_background).T
    # mean_background = np.array(mean_background).T
    # std_background = np.array(std_background).T
    #
    # # Compute overall means and standard deviations
    # orig_mean_signal, orig_std_signal = estimate_population_parameters(size_signal, mean_signal, std_signal)
    # orig_mean_background, orig_std_background = estimate_population_parameters(size_background, mean_background, std_background)
    #
    #
    # print('Means with outliers (sig):', orig_mean_signal)
    # print('Stds with outliers (sig) :', orig_std_signal)
    # print('Means with outliers (bkg):', orig_mean_background)
    # print('Stds with outliers (bkg) :', orig_std_background)
    #
    # # Initialize lists to hold mean and std values for signal and background after outlier removal
    # size_signal = []
    # mean_signal = []
    # std_signal = []
    # size_background = []
    # mean_background = []
    # std_background = []
    #
    # for i in range(len(data["all_jet"])):
    #     im = data["all_jet"][i, :, :, :]
    #     label = labels[i]  # Assuming labels are already defined
    #     im[im < 1.e-5] = 0  # Apply threshold
    #     bad_channel = False
    #     size_channel = []
    #     mean_channel = []
    #     std_channel = []
    #     for j in range(args.channels):
    #         if not bad_channel:
    #             if label == 1:  # Signal
    #                 image = im[j,:,:]
    #                 image = image[image != 0]
    #                 if len(image) < 2 or np.mean(image) > orig_mean_signal[j] + 6 * orig_std_signal[j]:
    #                     bad_channel = True
    #                     continue
    #                 image = (image - orig_mean_signal[j]) / orig_std_signal[j]
    #                 size_channel.append(len(image))
    #                 mean_channel.append(image.mean())
    #                 std_channel.append(image.std(ddof=1))
    #             elif label == 0:  # Background
    #                 image = im[j,:,:]
    #                 image = image[image != 0]
    #                 if len(image) < 2 or np.mean(image) > orig_mean_background[j] + 6 * orig_std_background[j]:
    #                     bad_channel = True
    #                     continue
    #                 image = (image - orig_mean_background[j]) / orig_std_background[j]
    #                 size_channel.append(len(image))
    #                 mean_channel.append(image.mean())
    #                 std_channel.append(image.std(ddof=1))
    #
    #     if label == 1 and not bad_channel:
    #         size_signal.append(size_channel)
    #         mean_signal.append(mean_channel)
    #         std_signal.append(std_channel)
    #     elif label == 0 and not bad_channel:
    #         size_background.append(size_channel)
    #         mean_background.append(mean_channel)
    #         std_background.append(std_channel)
    #
    # size_signal = np.array(size_signal).T
    # mean_signal = np.array(mean_signal).T
    # std_signal = np.array(std_signal).T
    # size_background = np.array(size_background).T
    # mean_background = np.array(mean_background).T
    # std_background = np.array(std_background).T
    #
    # rem_after_mean_signal, rem_after_std_signal = estimate_population_parameters(size_signal, mean_signal, std_signal)
    # rem_after_mean_background, rem_after_std_background = estimate_population_parameters(size_background, mean_background, std_background)
    #
    #
    #
    # print('Means with outlier removal after mean/std calculation (sig):', rem_after_mean_signal)
    # print('Stds with outlier removal after mean/std calculation (sig) :', rem_after_std_signal)
    # print('Means with outlier removal after mean/std calculation (bkg):', rem_after_mean_background)
    # print('Stds with outlier removal after mean/std calculation (bkg) :', rem_after_std_background)
    #
    #
    # shuffled_indices = np.arange(data["all_jet"].shape[0])
    # # Shuffle the indices
    # np.random.seed(42)
    # np.random.shuffle(shuffled_indices)
    #
    # with h5py.File(args.analysis_dir + args.improper_norm_hdf5_name, 'w') as improper_data:
    #     dataset_names = ['train_jet', 'valid_jet', 'test_jet', 'train_meta', 'valid_meta', 'test_meta',]
    #     samples =  size_signal.shape[1] + size_background.shape[1]
    #     train_indx = samples * .8 // 1
    #     valid_indx = train_indx + samples * .1 // 1
    #     dataset_sizes = [train_indx, valid_indx - train_indx, samples - valid_indx, train_indx, valid_indx - train_indx, samples - valid_indx]
    #     datasets = {
    #         name: improper_data.create_dataset(
    #             name,
    #             (size, args.channels, 125, 125) if 'jet' in name else (size, 1),
    #             dtype='float32',  # Specify an appropriate data type
    #             compression='lzf',
    #             chunks=(1, args.channels, 125, 125) if 'jet' in name else (1, 1),
    #         ) for name, size in zip(dataset_names, dataset_sizes)
    #     }
    #
    #     current_index = 0
    #     set = 'train'
    #     for i in range(len(data["all_jet"])):
    #         bad_channel = False
    #         im = data["all_jet"][shuffled_indices[i],:,:,:]
    #         label = data["all_meta"][shuffled_indices[i]]
    #         im[im < 1.e-5] = 0
    #         size_channel = []
    #         mean_channel = []
    #         std_channel = []
    #         if set == 'train' and current_index == train_indx:
    #             set = 'valid'
    #             current_index = 0
    #         if set == 'valid' and current_index == valid_indx - train_indx:
    #             set = 'test'
    #             current_index = 0
    #         for j in range(args.channels):
    #             if bad_channel:
    #                 continue
    #             if label == 1:  # Signal
    #                 image = im[j,:,:]
    #                 nz = image[image != 0]
    #                 if len(nz) < 2 or np.mean(nz) > orig_mean_signal[j] + 6 * orig_std_signal[j]:
    #                     bad_channel = True
    #                     continue
    #                 image = (image - orig_mean_signal[j]) / orig_std_signal[j]
    #             elif label == 0:  # Background
    #                 image = im[j,:,:]
    #                 nz = image[image != 0]
    #                 if len(nz) < 2 or np.mean(nz) > orig_mean_background[j] + 6 * orig_std_background[j]:
    #                     bad_channel = True
    #                     continue
    #                 image = (image - orig_mean_background[j]) / orig_std_background[j]
    #             improper_data[set+'_jet'][current_index,j,:,:] = image
    #         if not bad_channel:
    #             improper_data[set+'_meta'][current_index] = label
    #             current_index += 1
    #
    # stat = {
    #     "mean":rem_before_mean,
    #     "std":rem_before_std,
    #     "signal_mean":orig_mean_signal,
    #     "signal_std":orig_std_signal,
    #     "background_mean":orig_mean_background,
    #     "background_std":orig_std_background,
    # }
    #
    # with open(args.analysis_dir + 'stat.json', 'w') as fp:
    #     json.dump(stat, fp)
    #
    # data.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', '-c', default=13, ###Replace with your parquet dir
                        type=int, help='number of image channels')
    parser.add_argument('--batch_size', '-b', default=100,
                        type=int, help='batch size while processing data')
    parser.add_argument('--hdf5_name', '-n', default='IMG_aToTauTau_Hadronic_tauDR0p4_m3p6To14p8_dataset_2_unbaised_v2_0000_train.h5', ###Replace with your dataset file name
                        type=str, help='name of hdf5 file')
    parser.add_argument('--proper_norm_hdf5_name', '-p', default='property_normalized', ###Replace with your dataset file name
                        type=str, help='name of properly normalized hdf5 file')
    parser.add_argument('--improper_norm_hdf5_name', '-i', default='properly_not_normalized', ###Replace with your dataset file name
                        type=str, help='name of improperly normalized hdf5 file')
    parser.add_argument('--channel_means', '-m', default=[], type=list)
    parser.add_argument('--channel_stds', '-s', default=[], type=list)
    parser.add_argument('--analysis_dir', '-a', default = '/pscratch/sd/b/bbbam/IMG_aToTauTau_Hadronic_tauDR0p4_m3p6To14p8_dataset_2_unbaised_v2_train_hd5/',type=str, help='analysis directory e.g. /path/to/analysis/')
    args = parser.parse_args()
    main()
