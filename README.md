# Project Introduction

This project aims to optimise short-video recommendation using multi-modal
information.

# Set Up

Clone repositories by running the following codes. 

```
git clone https://github.com/antoine77340/video_feature_extractor
git clone https://github.com/antoine77340/S3D_HowTo100M.git
git clone https://github.com/niannujiaowj/MultiModality4ShortVideoRecommendation.git`
```

After `cd` to the project directory `MultiModality4ShortVideoRecommendation`, 
set up the project by running

`pip install -e .`

Download pretrained S3D models by running
```
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy
```

# Prepare Dataset

We need to prepare four parts of data.

**Part 1**: videos. Prefer in `.mp4` format.

**Part 2**: musics. Prefer in `.wav` format.

**Part 3**: interaction information of users and videos. Please include 
following fields, and save in a `.csv` file. 
The illustrations of data type and content are as below.


| Field       | Data Type     | Content  |
| ------------|:--------------| :-----|
| userID      | string or int | the user's ID |
| userGender  | string        | the user's gender |
| userAge     | int           | the user's age |
| userLocation| string        | the user's location |
| videoID     | string or int | the video's ID |
| liveScores  | float         | the video's live score |
| finish      | int           | whether the user finishes the video, if is use 1, otherwise 0 |
| like        | int           | whether the user likes the video, if is use 1, otherwise 0 |
| favorites   | int           | whether the user favorites the video, if is use 1, otherwise 0 |
| forward     | int           | whether the user forwards the video, if is use 1, otherwise 0 |



**Part 4**: videos information. Please include following fields, and save 
in a `.csv` file.
The illustrations of data type and content are as below.


| Field        | Data Type           | Content  |
| ------------- |:-------------| :-----|
| videoID      | string or int |the video's ID|
| musicID      | string or int |ID of the video's music|
| videoLocation | string      |the video's location|
| description | string      |the video's descriptions |
| comment1 | string      |the video's top comment |
| comment2 | string      |the video's second top comment |
| comment3 | string      |the video's third top comment|
| comment4 | string      |the video's fourth top comment |
| comment5 | string      |the video's fifth top comment|
| hot | int      |whether the video is on the hot list, if is use 1, otherwise 0 |
| entertainmentHot | int      |whether the video is on the entertainment hot list, if is use 1, otherwise 0 |
| socialHot | int      |whether the video is on the social hot list, if is use 1, otherwise 0|
| challengeHot | int      |whether the video is on the challenge hot list, if is use 1, otherwise 0|
| authorHot | int      |whether the video is on the author hot list, if is use 1, otherwise 0|
| authorID | string or int      |the author's ID|
| authorGender | string      |the author's gender|
| authorAge | int      |the author's age|
| authorLocation | string      |the author's location|



# Preprocess
Preprocess your data by running the following codes.
```
python scripts/preprocessing/preprocess.py
'./data/videos'  # Directory path to all videos
'./data/audios/musics' # Directory path to all musics
'./data/interactions_information.csv' # File path to interactions information mentioned as Part 3 in above section
'./data/videos_information.csv' # File path to videos information mentioned as Part 4 in above section
--S3D_pretrained_model_path './s3d_howto100m.pth' 
--videos_semantic_feature_save_path './data/features/videos_semantics'  
--videos_actions_feature_save_path './data/features/videos_actions' 
--key_images_save_path './data/images' 
--subtitles_save_path './data/subtitles' 
--musics_feature_save_path './data/features/musics' 
--datasets_save_path './data/datasets' 
--train_propotion 0.7
--valid_proportion 0.15
```



# Train

Run the following code and change the values of arguments if necessary.

```
python scripts/train.py 
--pretrained_bert_model 'bert-base-chinese'
--train_set './data/datasets/train.csv'
--valid_set './data/datasets/valid.csv'
--test_set './data/datasets/text.csv'
--batch_size 128
--gender_max_token_len 1
--age_max_token_len 1
--location_max_token_len 10
--video_description_max_token_len 50
--video_comment_max_token_len 50
--video_action_feature_max_len 30
--music_feature_max_len 30
--subtitle_max_token_len 200
--user_id_vocab_size 10000000000
--author_id_vocab_size 10000000000
--hidden_dim 32
--drop_out 0.1
--learning_rate 0.001
--early_stopping_patience 50
--n_epochs 20
--n_gpu 0
--n_workers 12
--checkpoints_save_path './checkpoints'
--logs_save_path './logs'
```

# Inference

If you want to inference on your own data using pretrained model, please
follow section *Prepare Dataset* and *Preprocess*, then run following codes.
```
python scripts/inference.py 
'inference_file_path' # File path to the inference data
'best_checkpoints_path' # File path to the best checkpoints
--pretrained_bert_model 'bert-base-chinese'
--batch_size 128
--gender_max_token_len 1 # Please keep it the same as your pretrained model
--age_max_token_len 1 # Please keep it the same as your pretrained model
--location_max_token_len 10 # Please keep it the same as your pretrained model
--video_description_max_token_len 50 # Please keep it the same as your pretrained model
--video_comment_max_token_len 50 # Please keep it the same as your pretrained model
--video_action_feature_max_len 30 # Please keep it the same as your pretrained model
--music_feature_max_len 30 # Please keep it the same as your pretrained model
--subtitle_max_token_len 200 # Please keep it the same as your pretrained model
--user_id_vocab_size 10000000000 # Please keep it the same as your pretrained model
--author_id_vocab_size 10000000000 # Please keep it the same as your pretrained model
--n_workers 12
--inference_save_path './data/inference'
--inference_save_file_name 'inference.json'
```

When doing inference, you can either specify the labels or not. If labels are
given, the output will include the loss figure, otherwise won't. 
