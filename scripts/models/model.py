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
    BertTokenizer,
    BertModel,
)
from transformers.training_args import TrainingArguments

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, pretrained_bert_model: str, user_id_vocab_size: int = 10000000000,
                 author_id_vocab_size: int = 10000000000, hidden_dim: int = 768, drop_out: float = 0.1):
        super(MyModel, self).__init__()
        torch.manual_seed(1)

        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_model)
        self.model = BertModel.from_pretrained(pretrained_bert_model)
        self.hidden_dim = hidden_dim
        self.pretrained_vocab_size = len(self.tokenizer)
        self.user_id_vocab_size = user_id_vocab_size
        for name, para in self.model.named_parameters():  # 12-layer, 768-hidden, 12-heads
            if name == "embeddings.word_embeddings.weight":
                self.pretrained_word_embeddings = para  # (pretrained_vocab_size, pretrained_hidden_dim)
                self.pretrained_hidden_dim = para.shape[1]
                break

        self.word_embedding_transformation_layer = nn.Linear(self.pretrained_hidden_dim, hidden_dim).to(self.device)

        self.user_id_embed_layer = nn.Embedding(user_id_vocab_size, hidden_dim).to(self.device)
        self.author_id_embed_layer = nn.Embedding(author_id_vocab_size, hidden_dim).to(self.device)

        self.video_hot_embed_layer = nn.Embedding(2, hidden_dim).to(self.device)
        self.video_entertainment_hot_embed_layer = nn.Embedding(2, hidden_dim).to(self.device)
        self.video_social_hot_embed_layer = nn.Embedding(2, hidden_dim).to(self.device)
        self.video_challenge_hot_embed_layer = nn.Embedding(2, hidden_dim).to(self.device)
        self.video_author_hot_embed_layer = nn.Embedding(2, hidden_dim).to(self.device)

        self.video_action_feature_transformation_layer = nn.Linear(2048, hidden_dim).to(self.device)
        self.video_semantic_feature_transformation_layer = nn.Linear(512, hidden_dim).to(self.device)
        self.music_feature_transformation_layer = nn.Linear(768, hidden_dim).to(self.device)

        self.video_live_score_transformation_layer = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(drop_out)
        ).to(self.device)

        self.token_type_embed = nn.Embedding(3, hidden_dim).to(self.device)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.Tanh(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(drop_out),

            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.Tanh(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(drop_out),

            nn.Linear(self.hidden_dim, 2),
            nn.LogSoftmax(dim=1)
        ).to(self.device)




    def _special_token_embedding(self, special_token_id):
        '''
        Used to deal with special tokens of [CLS], [SEP].
        :param special_token_id: torch.tensor with shape (batch_size)
        :return: special_token_embeddings, torch.tensor with shape (batch_size, 1, pretrained_hidden_dim)
        '''

        # (batch_size) -> (batch_size, pretrained_vocab_size)
        special_token_onehot = F.one_hot(special_token_id, num_classes = self.pretrained_vocab_size).view(-1, self.pretrained_vocab_size).type(dtype = torch.float)
        # (batch_size, 1, pretrained_hidden_dim)
        special_token_embeddings = torch.matmul(special_token_onehot, self.pretrained_word_embeddings).view(-1,1,self.pretrained_hidden_dim)
        special_token_embeddings = self.word_embedding_transformation_layer(special_token_embeddings) # (batch_size, 1, hidden_dim)
        return special_token_embeddings




    def _text_info_embedding(self, text_info, batch_size):
        '''
        Used to deal with user/author demographic info (gender, age, location) and video text info (location,
        description, comment 1-5, subtitle)
        :param text_info: tokenized text input ids with shape (max_length of text_info, batch_size)
        :param batch_size: batch_size
        :return: text_info_embeddings, torch.tensor with shape (batch_size, max_length of text_info, hidden_dim)
        '''

        text_info = torch.LongTensor(text_info).view(batch_size, -1) # (batch_size, max_length of text_info)
        # (batch_size, max_length of text_info, pretrained_vocab_size)
        text_info_onehot =  F.one_hot(text_info, num_classes = self.pretrained_vocab_size).type(dtype = torch.float)
        text_info_embeddings = torch.matmul(text_info_onehot, self.pretrained_word_embeddings)
        # (batch_size, max_length of text_info, hidden_dim)
        text_info_embeddings = self.word_embedding_transformation_layer(text_info_embeddings)
        return text_info_embeddings





    def forward(self, user_id, user_info, user_info_mask, author_id, author_info, author_info_mask,
                video_live_score, video_hot, video_entertainment_hot, video_social_hot, video_challenge_hot,
                video_author_hot, video_action_feature, video_semantic_feature, music_feature,
                video_info, video_action_feature_mask, music_feature_mask, video_info_mask,
                cls_id, sep_id, label = None):

        batch_size = len(user_id)

        CLS_embeddings = self._special_token_embedding(cls_id) # (batch_size, 1, hidden_dim)
        SEP_embeddings = self._special_token_embedding(sep_id)  # (batch_size, 1, hidden_dim)



        # user part process
        user_id = self.user_id_embed_layer(user_id).view(-1,1,self.hidden_dim) # (batch_size, 1, hidden_dim)
        user_info_embeddings = self._text_info_embedding(user_info, batch_size) # (batch_size, max_length of user_info, hidden_dim)
        user = torch.cat([CLS_embeddings, user_id, user_info_embeddings, SEP_embeddings], dim=1) # (batch_size, max_length of user, hidden_dim)
        type0 = torch.zeros(batch_size, user.shape[1]).type(dtype=torch.long) # (batch_size, max_length of user)
        type0 = self.token_type_embed(type0)  # (batch_size, max_length of user, hidden_dim)
        user = user + type0
        # (batch_size, max_length of user)
        user_attention_mask = torch.cat([torch.ones(batch_size, 2), # [CLS] and user_id
                                         torch.tensor(user_info_mask),
                                         torch.ones(batch_size, 1)], dim = 1) # [SEP]



        # author part process
        author_id = self.author_id_embed_layer(author_id).view(-1,1,self.hidden_dim) # (batch_size, 1, hidden_dim)
        author_info_embeddings = self._text_info_embedding(author_info,batch_size) # (batch_size, max_length of author_info, hidden_dim)
        author = torch.cat([author_id,author_info_embeddings,SEP_embeddings], dim=1) # (batch_size, max_length of author, hidden_dim)
        type1 = torch.ones(batch_size, author.shape[1]).type(dtype=torch.long)  # (batch_size, max_length of author)
        type1 = self.token_type_embed(type1)  # (batch_size, max_length of author, hidden_dim)
        author = author + type1
        # (batch_size, max_length of author)
        author_attention_mask = torch.cat([torch.ones(batch_size, 1), # author_id
                                         torch.tensor(author_info_mask),
                                         torch.ones(batch_size, 1)], dim = 1) # [SEP]



        # video part process
        video_info_embeddings = self._text_info_embedding(video_info, batch_size) # (batch_size, max_length of video_info, hidden_dim)
        # (batch_size, 1, hidden_dim)
        video_live_score = self.video_live_score_transformation_layer(video_live_score.view(batch_size,1)).view(batch_size, 1, self.hidden_dim)

        # (batch_size, 1, hidden_dim)
        video_hot = self.video_hot_embed_layer(video_hot).view(-1,1,self.hidden_dim)
        video_entertainment_hot = self.video_entertainment_hot_embed_layer(video_entertainment_hot).view(-1,1,self.hidden_dim)
        video_social_hot = self.video_social_hot_embed_layer(video_social_hot).view(-1,1,self.hidden_dim)
        video_challenge_hot = self.video_challenge_hot_embed_layer(video_challenge_hot).view(-1,1,self.hidden_dim)
        video_author_hot = self.video_author_hot_embed_layer(video_author_hot).view(-1,1,self.hidden_dim)

        # (batch_size, max_feature_length, hidden_dim)
        video_action_feature = self.video_action_feature_transformation_layer(torch.FloatTensor(video_action_feature).view(batch_size, -1, 2048))
        video_semantic_feature = self.video_semantic_feature_transformation_layer(torch.FloatTensor(video_semantic_feature).view(batch_size, -1, 512))
        music_feature = self.music_feature_transformation_layer(torch.FloatTensor(music_feature).view(batch_size, -1, 768))

        video = torch.cat([video_info_embeddings,video_live_score,video_hot, video_entertainment_hot, video_social_hot,
                           video_challenge_hot,video_author_hot,video_action_feature,video_semantic_feature,
                           music_feature,SEP_embeddings], dim=1)
        type2 = torch.ones(batch_size, video.shape[1]).type(dtype=torch.long) # (batch_size, max_length of video)
        type2[type2 == 1] = 2
        type2 = self.token_type_embed(type2)  # (batch_size, max_length of video, hidden_dim)
        video = video + type2
        # (batch_size, max_length of video)
        video_attention_mask = torch.cat([torch.tensor(video_info_mask),
                                         torch.ones(batch_size, 6), # live_score, hot, entertainment_hot, social_hot, challenge_hot, author_hot
                                         torch.tensor(video_action_feature_mask),
                                         torch.ones(batch_size, 1), # video_semantic_feature
                                         torch.tensor(music_feature_mask),
                                         torch.ones(batch_size, 1)], dim = 1)  # [SEP]


        # concatenate all inputs embeddings
        inputs_embeddings = torch.cat([user, author, video], dim = 1)
        attention_mask = torch.cat([user_attention_mask, author_attention_mask, video_attention_mask], dim = 1)

        # feed inputs into the model, get last hidden state of [CLS], then feed into the classifier
        outputs = self.model(inputs_embeds=inputs_embeddings, attention_mask=attention_mask, labels=label)
        CLS_hidden_state = outputs.last_hidden_state[:,:1,:].view(batch_size,-1) # (batch_size, hidden_dim)
        logits = self.classifier(CLS_hidden_state) # (batch_size, 2)

        if label is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, label) # label (batch_size)
            pred = logits.max(1).indices # (batch_size)
            return loss, logits, pred
        else:
            loss = None
            pred = logits.max(1).indices
            return loss, logits, pred














