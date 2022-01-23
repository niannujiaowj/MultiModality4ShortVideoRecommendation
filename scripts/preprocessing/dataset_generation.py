from tqdm import tqdm
import h5py
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_data(interactions_information_file: Path = './data/interactions_information.csv',
                 videos_information_file: Path = './data/videos_information.csv',
                 videos_actions_features_folder: Path = './data/features/videos_actions',
                 videos_semantic_features_file: Path = './data/features/videos_semantics/videos_semantic_features.hdf5',
                 musics_features_folder: Path = './data/features/musics/musics_features.hdf5',
                 subtitles_folder: Path = './data/subtitles',
                 train_propotion: float = 0.7,
                 valid_proportion: float = 0.15,
                 datasets_save_path: Path = './data/datasets'):
    '''
    :param interaction_file: csv file of interaction information
    :param video_information_file: csv file of video information
    :param video_style_feature_folder: the folder path of video style features
    :param video_semantic_feature_file: hdf5 file path of video semantic features
    :param music_feature_folder: the folder path of music features
    :param subtitle_folder: the folder path of subtitles
    :param output_merged_file_path: the output folder path of train, valid and test
    :return: length of train, valid and test set
    '''
    interactions = pd.read_csv(interactions_information_file)
    videos = pd.read_csv(videos_information_file)

    # vlookup video_information_file into interaction_file
    merged_file = interactions.join(videos.set_index('videoID'),on='videoID')
    merged_file['video_action_feature'] = None
    merged_file['video_semantic_feature'] = None
    merged_file['music_feature'] = None
    merged_file['subtitle'] = None

    # merge feature embeddings into merged file
    video_semantic_features = h5py.File(videos_semantic_features_file)
    music_features = h5py.File(musics_features_folder)
    row_num = merged_file.shape[0]
    for r in tqdm(range(row_num)):
        videoID = merged_file.iloc[r, merged_file.columns.get_loc('videoID')]
        musicID = merged_file.iloc[r, merged_file.columns.get_loc('musicID')]
        merged_file.at[r, 'video_action_feature'] = np.load(videos_actions_features_folder + '/{}.npy'.format(videoID))
        merged_file.at[r, 'video_semantic_feature'] = video_semantic_features.attrs[str(videoID)].flatten()
        if musicID is not None:
            merged_file.at[r, 'music_feature'] = music_features.attrs[str(musicID)]
        with open(subtitles_folder + '/{}.txt'.format(videoID)) as f:
            subtitle = f.readlines()
            if subtitle != []:
                merged_file.at[r, 'subtitle'] = subtitle[0]

    video_semantic_features.close()
    music_features.close()

    # Split datasets.
    train = merged_file.sample(frac=train_propotion, random_state=1)
    rest = merged_file.loc[~merged_file.index.isin(train.index)]
    valid = rest.sample(frac=valid_proportion / (1 - train_propotion), random_state=1)
    test = rest.loc[~rest.index.isin(valid.index)]

    train.to_csv(datasets_save_path + '/train.csv', index=False)
    valid.to_csv(datasets_save_path + '/valid.csv', index=False)
    test.to_csv(datasets_save_path + '/test.csv', index=False)

    return len(train), len(valid), len(test)

