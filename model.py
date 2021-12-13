from tqdm import tqdm
import h5py
from pathlib import Path
from dataclasses import dataclass, field
import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    EvalPrediction,
    set_seed
)
from transformers.training_args import TrainingArguments

from multimodal_transformers.data import load_data_from_folder
from multimodal_transformers.model import TabularConfig
from multimodal_transformers.model import AutoModelWithTabular

logging.basicConfig(level=logging.INFO)
os.environ['COMET_MODE'] = 'DISABLED'


def prepare_data(interaction_file: Path = "data/interaction_information.csv",
                 video_information_file: Path = "data/video_information.csv",
                 video_style_feature_folder: Path = "data/features/videos_style",
                 video_semantic_feature_file: Path = "data/features/videos_semantics/video_semantic_features.hdf5",
                 music_feature_folder: Path = "data/audios/musics",
                 subtitle_folder: Path = "data/text/subtitles",
                 output_merged_file_path: Path = "data/dataset"):
    '''
    :param interaction_file: csv file of interaction information
    :param video_information_file: csv file of video information
    :param video_style_feature_folder: the folder path of video style features
    :param video_semantic_feature_file: hdf5 file path of video semantic features
    :param music_feature_folder: the folder path of music features
    :param subtitle_folder: the folder path of subtitles
    :param output_merged_file_path: the output folder path of train, valid and test
    :return: saved train, valid and test set
    '''
    interactions = pd.read_csv(interaction_file)
    videos = pd.read_csv(video_information_file)

    # vlookup video_information_file into interaction_file
    merged_file = interactions.join(videos.set_index('videoID'),on='videoID')
    merged_file['video_style_feature'] = None
    merged_file['video_semantic_feature'] = None
    merged_file['music_feature'] = None
    merged_file['subtitle'] = None

    video_semantic_features = h5py.File(video_semantic_feature_file)
    music_features = h5py.File(music_feature_folder)
    row_num = merged_file.shape[0]
    for r in tqdm(range(row_num)):
        videoID = merged_file.iloc[r, merged_file.columns.get_loc('videoID')]
        musicID = merged_file.iloc[r, merged_file.columns.get_loc('musicID')]
        merged_file.at[r, 'video_style_feature'] = np.load(video_style_feature_folder + '/' + '{}.npy'.format(videoID))
        merged_file.at[r, 'video_semantic_feature'] = video_semantic_features.attrs[videoID].flatten()
        if musicID is not None:
            merged_file.at[r, 'music_feature'] = music_features.attrs[musicID]
        with open(subtitle_folder + '/' + '{}.txt') as f:
            subtitle = f.readlines()
            if subtitle != []:
                merged_file.at[r, 'subtitle'] = subtitle[0]

    video_semantic_features.close()
    music_features.close()

    print("Spliting train, valid and test set into 8:1:1...")
    train_df, valid_df, test_df = np.split(merged_file.sample(frac=1), [int(.8 * len(merged_file)), int(.9 * len(merged_file))])
    train_df.to_csv(output_merged_file_path+'/train.csv')
    valid_df.to_csv(output_merged_file_path+'/valid.csv')
    test_df.to_csv(output_merged_file_path+'/test.csv')
    print("Done!\nNum examples are: train{}, valid{} and test{}.".format(len(train_df), len(valid_df), len(test_df)))
    #return merged_file

# https://github.com/georgian-io/Multimodal-Toolkit
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"})


@dataclass
class MultimodalDataTrainingArguments:
    """
    Arguments pertaining to how we combine tabular features
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_path: str = field(metadata={'help': 'the path to the csv file containing the dataset'})
    column_info_path: str = field(default=None, metadata={
          'help': 'the path to the json file detailing which columns are text, categorical, numerical, and the label'})

    column_info: dict = field(default=None, metadata={
          'help': 'a dict referencing the text, categorical, numerical, and label columns'
                  'its keys are text_cols, num_cols, cat_cols, and label_col'})

    categorical_encode_type: str = field(default='ohe', metadata={
                                            'help': 'sklearn encoder to use for categorical data',
                                            'choices': ['ohe', 'binary', 'label', 'none']})
    numerical_transformer_method: str = field(default='yeo_johnson', metadata={
                                                'help': 'sklearn numerical transformer to preprocess numerical data',
                                                'choices': ['yeo_johnson', 'box_cox', 'quantile_normal', 'none']})
    task: str = field(default="classification", metadata={
                        "help": "The downstream training task",
                        "choices": ["classification", "regression"]})
    mlp_division: int = field(default=4, metadata={
                                'help': 'the ratio of the number of '
                                        'hidden dims in a current layer to the next MLP layer'})
    combine_feat_method: str = field(default='individual_mlps_on_cat_and_numerical_feats_then_concat', metadata={
                                        'help': 'method to combine categorical and numerical features, '
                                                'see README for all the method'})
    mlp_dropout: float = field(default=0.1, metadata={
                                'help': 'dropout ratio used for MLP layers'})
    numerical_bn: bool = field(default=True, metadata={
                                  'help': 'whether to use batchnorm on numerical features'})
    use_simple_classifier: str = field(default=True, metadata={
                                          'help': 'whether to use single layer or MLP as final classifier'})
    mlp_act: str = field(default='relu', metadata={
                            'help': 'the activation function to use for finetuning layers',
                            'choices': ['relu', 'prelu', 'sigmoid', 'tanh', 'linear']})
    gating_beta: float = field(default=0.2, metadata={
                                  'help': "the beta hyperparameters used for gating tabular data "
                                          "see https://www.aclweb.org/anthology/2020.acl-main.214.pdf"})

    def __post_init__(self):
        assert self.column_info != self.column_info_path
        if self.column_info is None and self.column_info_path:
            with open(self.column_info_path, 'r') as f:
                self.column_info = json.load(f)




"""def resize_tokenizer(data: pd.DataFrame, tokenizer: BertTokenizer):
    '''
    :param data: merged pandas DataFrame file
    :param tokenizer: pretrained BertTokenizer
    :return: BertTokenizer with added tokens
    '''
    # merge userIDs and authorIDs, then get the list of user tokens
    user_tokens = (data['userID'].append(data['authorID'])).drop_duplicates().values.tolist()
    user_tokens = [str(x) for x in user_tokens] # convert int ids to str

    # get the list of videoIDs
    video_tokens = data['videoID'].drop_duplicates().values.tolist()
    video_tokens = [str(x) for x in video_tokens]

    # get the list of musicIDs
    music_tokens = data['musicID'].drop_duplicates().values.tolist()

    tokenizer.add_tokens(user_tokens)
    tokenizer.add_tokens(video_tokens)
    tokenizer.add_tokens(music_tokens)
    return tokenizer



class MM4SCRDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, descriptio_max_token_len: int = 30,
                 comment_max_token_len: int = 10, subtitle_max_token_len: int = 500,
                 video_style_feature_max_len: int = 100, music_feature_max_len: int = 100):
        self.data = data
        self.tokenizer = tokenizer
        self.descriptio_max_token_len = descriptio_max_token_len
        self.comment_max_token_len = comment_max_token_len
        self.subtitle_max_token_len = subtitle_max_token_len
        self.video_style_feature_max_len = video_style_feature_max_len
        self.music_feature_max_len = music_feature_max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        # user information encoding
        userID_encoding = self.tokenizer(data_row["userID"], return_tensors="pt")
        userGender_encoding =
        userAge_encoding =
        userLocation_encoding =

        # video information encoding
        videoID_encoding = self.tokenizer(data_row["videoID"], return_tensors="pt")
        video_style_feature_encoding = data_row['video_style_feature'] # require flatten and padding
        video_semantic_feature_encoding = data_row['video_semantic_feature'] # 512
        music_feature_encoding = data_row[' music_feature_encoding']  # require flatten and padding

        # ???????add_special_tokens=True <eos>???
        description_encoding =  self.tokenizer(data_row["description"], max_length=self.descriptio_max_token_len,
                                               padding="max_length", return_attention_mask=True,return_tensors="pt")

        comment1_encoding = self.tokenizer(data_row["comment1"], max_length=self.comment_max_token_len,
                                               padding="max_length", return_attention_mask=True,return_tensors="pt")
        comment2_encoding = self.tokenizer(data_row["comment2"], max_length=self.comment_max_token_len,
                                               padding="max_length", return_attention_mask=True,return_tensors="pt")
        comment3_encoding = self.tokenizer(data_row["comment3"], max_length=self.comment_max_token_len,
                                           padding="max_length", return_attention_mask=True, return_tensors="pt")
        comment4_encoding = self.tokenizer(data_row["comment4"], max_length=self.comment_max_token_len,
                                           padding="max_length", return_attention_mask=True, return_tensors="pt")
        comment5_encoding = self.tokenizer(data_row["comment5"], max_length=self.comment_max_token_len,
                                           padding="max_length", return_attention_mask=True, return_tensors="pt")
        subtitle_encoding = self.tokenizer(data_row["subtitle"], max_length=self.subtitle_max_token_len,
                                           padding="max_length", return_attention_mask=True, return_tensors="pt")
        musicID_encoding = self.tokenizer(data_row["musicID"], return_tensors="pt")
        videoLocation_encoding =
        liveScores_encoding =
        hot_encoding =
        entertaimentHot_encoding =
        socialHot_encoding =
        challengeHot_encoding =
        authorHot_encoding =
        authorID_encoding = self.tokenizer(data_row["authorID"], return_tensors="pt")
        authorGender_encoding =
        authorAge_encoding =
        authorLocation_encoding =





        finish_encoding =
        like_encoding =
        favorites_encoding =
        forward_encoding ="""






if __name__ == '__main__':
    # pretrained model: bert-base-chinese
    prepare_data()

    text_cols = ['userLocation', 'videoLocation', 'authorLocation', 'description', 'comment1',
                 'comment2', 'comment3', 'comment4', 'comment5', 'subtitle']
    category_cols = ['userID', 'userGender', 'videoID', 'musicID', 'authorID', 'authorGender',
                     'hot', 'entertaimentHot', 'socialHot', 'challengeHot', 'authorHot']
    numerical_cols = ['userAge', 'liveScores', 'authorAge', '', '', '', '', '', '']
    arr_cols = ['video_style_feature', 'video_semantic_feature', 'music_feature', '', '', '']

    column_info_dict = {
        'text_cols': text_cols,
        'num_cols': numerical_cols,
        'cat_cols': category_cols,
        'label_col': 'finish', # what if multi predictions?
        'label_list': [1, 0]}

    model_args = ModelArguments(model_name_or_path='bert-base-chinese')

    data_args = MultimodalDataTrainingArguments(
        data_path='data/dataset',
        combine_feat_method='gating_on_cat_and_num_feats_then_sum',
        column_info=column_info_dict,
        task='classification')

    training_args = TrainingArguments(
        output_dir="logs/our_model",
        logging_dir="logs/runs/our_model",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=32,
        num_train_epochs=1,
        evaluate_during_training=True,
        logging_steps=25,
        eval_steps=250
    )

    set_seed(training_args.seed)

    tokenizer_path_or_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name, cache_dir=model_args.cache_dir)

