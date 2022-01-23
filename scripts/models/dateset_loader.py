import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from scripts.models.dataset_preparation import MM4SVRDataset



class MM4SVRDataModuel(pl.LightningDataModule):
    def __init__(self, pretrained_bert_model: str, train_DataFrame: pd.DataFrame = None,
                 valid_DataFrame: pd.DataFrame = None, test_DataFrame: pd.DataFrame = None,
                 batch_size: int = 8, num_workers: int = 12,
                 gender_max_token_len: int = 1, age_max_token_len: int = 1,
                 location_max_token_len: int = 10, video_description_max_token_len: int = 50,
                 video_comment_max_token_len: int = 50, video_action_feature_max_len: int = 30,
                 music_feature_max_len: int = 30, subtitle_max_token_len: int = 200):

        super().__init__()
        self.train_DataFrame = train_DataFrame
        self.valid_DataFrame = valid_DataFrame
        self.test_DataFrame = test_DataFrame
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pretrained_bert_model = pretrained_bert_model
        self.gender_max_token_len = gender_max_token_len
        self.age_max_token_len = age_max_token_len
        self.location_max_token_len = location_max_token_len
        self.video_description_max_token_len = video_description_max_token_len
        self.video_comment_max_token_len = video_comment_max_token_len
        self.video_action_feature_max_len =video_action_feature_max_len
        self.music_feature_max_len = music_feature_max_len
        self.subtitle_max_token_len =subtitle_max_token_len

    def setup(self):
        self.train_dataset = MM4SVRDataset(data = self.train_DataFrame,
                                           pretrained_bert_model = self.pretrained_bert_model,
                                           gender_max_token_len = self.gender_max_token_len,
                                           age_max_token_len = self.age_max_token_len,
                                           location_max_token_len = self.location_max_token_len,
                                           video_description_max_token_len = self.video_description_max_token_len,
                                           video_comment_max_token_len = self.video_comment_max_token_len,
                                           video_action_feature_max_len = self.video_action_feature_max_len,
                                           music_feature_max_len = self.music_feature_max_len,
                                           subtitle_max_token_len = self.subtitle_max_token_len)
        self.valid_dataset = MM4SVRDataset(data = self.valid_DataFrame,
                                           pretrained_bert_model = self.pretrained_bert_model,
                                           gender_max_token_len = self.gender_max_token_len,
                                           age_max_token_len = self.age_max_token_len,
                                           location_max_token_len = self.location_max_token_len,
                                           video_description_max_token_len = self.video_description_max_token_len,
                                           video_comment_max_token_len = self.video_comment_max_token_len,
                                           video_action_feature_max_len = self.video_action_feature_max_len,
                                           music_feature_max_len = self.music_feature_max_len,
                                           subtitle_max_token_len = self.subtitle_max_token_len)
        self.test_dataset = MM4SVRDataset(data = self.test_DataFrame,
                                           pretrained_bert_model = self.pretrained_bert_model,
                                           gender_max_token_len = self.gender_max_token_len,
                                           age_max_token_len = self.age_max_token_len,
                                           location_max_token_len = self.location_max_token_len,
                                           video_description_max_token_len = self.video_description_max_token_len,
                                           video_comment_max_token_len = self.video_comment_max_token_len,
                                           video_action_feature_max_len = self.video_action_feature_max_len,
                                           music_feature_max_len = self.music_feature_max_len,
                                           subtitle_max_token_len = self.subtitle_max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False)
