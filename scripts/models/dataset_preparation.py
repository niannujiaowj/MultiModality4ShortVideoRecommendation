import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer



class MM4SVRDataset(Dataset):
    def __init__(self, data: pd.DataFrame,
                 pretrained_bert_model: str,
                 gender_max_token_len: int = 1,
                 age_max_token_len: int = 1,
                 location_max_token_len: int = 10,
                 video_description_max_token_len: int = 50,
                 video_comment_max_token_len: int = 50,
                 video_action_feature_max_len: int = 30,
                 music_feature_max_len: int = 30,
                 subtitle_max_token_len: int = 200):
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_model)
        self.gender_max_token_len = gender_max_token_len
        self.age_max_token_len = age_max_token_len
        self.location_max_token_len = location_max_token_len
        self.video_description_max_token_len = video_description_max_token_len
        self.video_comment_max_token_len = video_comment_max_token_len
        self.video_action_feature_max_len = video_action_feature_max_len
        self.music_feature_max_len = music_feature_max_len
        self.subtitle_max_token_len = subtitle_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        # Generate the label
        if 'finish' in self.data.columns:  # during training time, the decide the label
            finish = data_row['finish']
            like = data_row['like']
            favorites = data_row['favorites']
            forward = data_row['forward']
            if finish==1 or like==1 or favorites==1 or forward==1:
                label = 1
            else:
                label = 0
        else: # during inference time, the label is None
            label = None

        # Tokenize all text columns.
        user_gender_encodding = self.tokenizer(data_row['userGender'], max_length=self.gender_max_token_len,
                                              padding='max_length', truncation=True,return_attention_mask=True,
                                              add_special_tokens=False,return_tensors='pt')
        author_gender_encodding = self.tokenizer(data_row['authorGender'], max_length=self.gender_max_token_len,
                                              padding='max_length', truncation=True,return_attention_mask=True,
                                              add_special_tokens=False,return_tensors='pt')

        user_age_encodding = self.tokenizer(int(data_row['userAge']), max_length=self.age_max_token_len,
                                              padding='max_length', truncation=True,return_attention_mask=True,
                                              add_special_tokens=False,return_tensors='pt')
        author_age_encodding = self.tokenizer(int(data_row['authorAge']), max_length=self.age_max_token_len,
                                              padding='max_length', truncation=True,return_attention_mask=True,
                                              add_special_tokens=False,return_tensors='pt')

        user_location_encodding = self.tokenizer(data_row['userLocation'], max_length=self.location_max_token_len,
                                              padding='max_length', truncation=True,return_attention_mask=True,
                                              add_special_tokens=False,return_tensors='pt')
        author_location_encodding = self.tokenizer(data_row['authorLocation'], max_length=self.location_max_token_len,
                                              padding='max_length', truncation=True,return_attention_mask=True,
                                              add_special_tokens=False,return_tensors='pt')
        video_location_encodding = self.tokenizer(data_row['videoLocation'], max_length=self.location_max_token_len,
                                              padding='max_length', truncation=True,return_attention_mask=True,
                                              add_special_tokens=False,return_tensors='pt')

        video_description_encodding = self.tokenizer(data_row['description'], max_length=self.video_description_max_token_len,
                                              padding='max_length', truncation=True,return_attention_mask=True,
                                              add_special_tokens=False,return_tensors='pt')
        video_comment1_encodding =  self.tokenizer(data_row['comment1'], max_length=self.video_comment_max_token_len,
                                              padding='max_length', truncation=True,return_attention_mask=True,
                                              add_special_tokens=False,return_tensors='pt')
        video_comment2_encodding =  self.tokenizer(data_row['comment2'], max_length=self.video_comment_max_token_len,
                                              padding='max_length', truncation=True,return_attention_mask=True,
                                              add_special_tokens=False,return_tensors='pt')
        video_comment3_encodding =  self.tokenizer(data_row['comment3'], max_length=self.video_comment_max_token_len,
                                              padding='max_length', truncation=True,return_attention_mask=True,
                                              add_special_tokens=False,return_tensors='pt')
        video_comment4_encodding =  self.tokenizer(data_row['comment5'], max_length=self.video_comment_max_token_len,
                                              padding='max_length', truncation=True,return_attention_mask=True,
                                              add_special_tokens=False,return_tensors='pt')
        video_comment5_encodding =  self.tokenizer(data_row['comment5'], max_length=self.video_comment_max_token_len,
                                              padding='max_length', truncation=True,return_attention_mask=True,
                                              add_special_tokens=False,return_tensors='pt')
        subtitle_encodding = self.tokenizer(data_row['subtitle'], max_length=self.subtitle_max_token_len,
                                              padding='max_length', truncation=True,return_attention_mask=True,
                                              add_special_tokens=False,return_tensors='pt')

        cls_input_id = self.tokenizer.encode('[CLS]', max_length=1,
                                              padding='max_length', truncation=True,return_attention_mask=True,
                                              add_special_tokens=False,return_tensors='pt').flatten()[0].item()
        sep_input_id = self.tokenizer.encode('[SEP]', max_length=1,
                                              padding='max_length', truncation=True,return_attention_mask=True,
                                              add_special_tokens=False,return_tensors='pt').flatten()[0].item()


        # pad video_action_feature
        video_action_feature = torch.tensor(data_row['video_action_feature'])
        if video_action_feature.shape[0] >= self.video_action_feature_max_len:
            video_action_feature = video_action_feature[:self.video_action_feature_max_len, :]
            video_action_feature_mask = torch.ones(video_action_feature.shape[0])
        else:
            temp = torch.zeros(self.video_action_feature_max_len - video_action_feature.shape[0],
                               video_action_feature.shape[1]).type(dtype = float)
            video_action_feature = torch.cat([video_action_feature, temp], dim = 0)
            unpadding = torch.ones(video_action_feature.shape[0])
            padding = torch.zeros(self.video_action_feature_max_len - video_action_feature.shape[0])
            video_action_feature_mask = torch.cat([unpadding,padding], dim = 0)


        # pad music_feature
        music_feature = torch.tensor(data_row['music_feature'])
        if music_feature.shape[0] >= self.music_feature_max_len:
            music_feature = music_feature[:self.music_feature_max_len, :]
            music_feature_mask = torch.ones(music_feature.shape[0]).tolist()
        else:
            temp = torch.zeros(self.music_feature_max_len - music_feature.shape[0],
                               music_feature.shape[1]).type(dtype = float)
            music_feature = torch.cat([music_feature, temp], dim = 0)
            unpadding = torch.ones(music_feature.shape[0])
            padding = torch.zeros(self.music_feature_max_len - music_feature.shape[0])
            music_feature_mask = torch.cat([unpadding,padding], dim = 0)



        return dict(
            user_id = int(data_row['userID']),
            user_info = torch.concat([user_gender_encodding['input_ids'].flatten(),
                                      user_age_encodding['input_ids'].flatten(),
                                      user_location_encodding['input_ids'].flatten()]).tolist(),
            user_info_mask=torch.concat([user_gender_encodding['attention_mask'].flatten(),
                                         user_age_encodding['attention_mask'].flatten(),
                                         user_location_encodding['attention_mask'].flatten()]).tolist(),

            author_id=int(data_row['authorID']),
            author_info=torch.concat([author_gender_encodding['input_ids'].flatten(),
                                      author_age_encodding['input_ids'].flatten(),
                                      author_location_encodding['input_ids'].flatten()]).tolist(),
            author_info_mask=torch.concat([author_gender_encodding['attention_mask'].flatten(),
                                      author_age_encodding['attention_mask'].flatten(),
                                      author_location_encodding['attention_mask'].flatten()]).tolist(),

            video_live_score = float(data_row['liveScores']),
            video_hot = int(data_row['hot']),
            video_entertainment_hot = int(data_row['entertainmentHot']),
            video_social_hot = int(data_row['socialHot']),
            video_challenge_hot = int(data_row['challengeHot']),
            video_author_hot = int(data_row['authorHot']),
            video_action_feature = video_action_feature.tolist(),
            video_semantic_feature = data_row['video_semantic_feature'].tolist(),
            music_feature = music_feature.tolist(),
            video_info = torch.concat([video_location_encodding['input_ids'].flatten(),
                                       video_description_encodding['input_ids'].flatten(),
                                       video_comment1_encodding['input_ids'].flatten(),
                                       video_comment2_encodding['input_ids'].flatten(),
                                       video_comment3_encodding['input_ids'].flatten(),
                                       video_comment4_encodding['input_ids'].flatten(),
                                       video_comment5_encodding['input_ids'].flatten(),
                                       subtitle_encodding['input_ids'].flatten()]).tolist(),
            video_action_feature_mask = video_action_feature_mask.tolist(),
            music_feature_mask = music_feature_mask.tolist(),
            video_info_mask=torch.concat([video_location_encodding['attention_mask'].flatten(),
                                     video_description_encodding['attention_mask'].flatten(),
                                     video_comment1_encodding['attention_mask'].flatten(),
                                     video_comment2_encodding['attention_mask'].flatten(),
                                     video_comment3_encodding['attention_mask'].flatten(),
                                     video_comment4_encodding['attention_mask'].flatten(),
                                     video_comment5_encodding['attention_mask'].flatten(),
                                     subtitle_encodding['attention_mask'].flatten()]).tolist(),

            cls_id=cls_input_id,
            sep_id=sep_input_id,

            label = label,
        )
