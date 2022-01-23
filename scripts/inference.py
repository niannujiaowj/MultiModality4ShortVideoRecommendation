import os
import argparse
import json

import torch
from tqdm import tqdm
import pandas as pd

from scripts.models import dateset_loader, model_loader



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inference_file_path', type=str)
    parser.add_argument('best_checkpoints_path', type=str)
    parser.add_argument('--pretrained_bert_model', type=str, default='bert-base-chinese')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gender_max_token_len', type=int, default=1)
    parser.add_argument('--age_max_token_len', type=int, default=1)
    parser.add_argument('--location_max_token_len', type=int, default=10)
    parser.add_argument('--video_description_max_token_len', type=int, default=50)
    parser.add_argument('--video_comment_max_token_len', type=int, default=50)
    parser.add_argument('--video_action_feature_max_len', type=int, default=30)
    parser.add_argument('--music_feature_max_len', type=int, default=30)
    parser.add_argument('--subtitle_max_token_len', type=int, default=200)
    parser.add_argument('--user_id_vocab_size', type=int, default=10000000000)
    parser.add_argument('--author_id_vocab_size', type=int, default=10000000000)
    parser.add_argument('--n_workers', type=int, default=12)
    parser.add_argument('--inference_save_path', type=str, default='./data/inference')
    parser.add_argument('--inference_save_file_name', type=str, default='inference.json')
    args = parser.parse_args()



    INFERENCE_SAVE_PATH = args.inference_save_path
    SAVE_FILE_NAME = args.inference_save_file_name

    if not os.path.exists(INFERENCE_SAVE_PATH):
        os.mkdir(INFERENCE_SAVE_PATH)


    test_DataFrame = pd.read_csv(args.inference_file_path, skip_blank_lines=True)
    data_module = dateset_loader.MM4SVRDataModuel(pretrained_bert_model = args.pretrained_bert_model,
                                               test_DataFrame = test_DataFrame, batch_size = args.batch_size,
                                               num_workers = args.n_workers,
                                               gender_max_token_len = args.gender_max_token_len,
                                               age_max_token_len = args.age_max_token_len,
                                               location_max_token_len = args.location_max_token_len,
                                               video_description_max_token_len = args.video_description_max_token_len,
                                               video_comment_max_token_len = args.video_comment_max_token_len,
                                               video_action_feature_max_len = args.video_action_feature_max_len,
                                               music_feature_max_len = args.music_feature_max_len,
                                               subtitle_max_token_len = args.subtitle_max_token_len)

    data_module.setup()
    model = model_loader.MM4SVRModel.load_from_checkpoint(checkpoint_path = args.best_checkpoints_path)
    model.eval()
    model.freeze()

    data_num = 0
    final_loss = 0
    final_logits = torch.tensor([])
    final_pred = torch.tensor([])
    for batch in tqdm(data_module.test_dataloader()):
        if 'label' in batch:
            label = batch['label']
        else:
            label = None

        batch_size = len(batch['user_id'])
        data_num += batch_size

        loss, logits, pred = model(user_id = batch['user_id'], user_info = batch['user_info'],
                                        user_info_mask = batch['user_info_mask'], author_id = batch['author_id'],
                                        author_info = batch['author_info'], author_info_mask = batch['author_info_mask'],
                                        video_live_score = batch['video_live_score'], video_hot = batch['video_hot'],
                                        video_entertainment_hot = batch['video_entertainment_hot'],
                                        video_social_hot = batch['video_social_hot'],
                                        video_challenge_hot = batch['video_challenge_hot'],
                                        video_author_hot = batch['video_author_hot'],
                                        video_action_feature = batch['video_action_feature'],
                                        video_semantic_feature = batch['video_semantic_feature'],
                                        music_feature = batch['music_feature'], video_info = batch['video_info'],
                                        video_action_feature_mask = batch['video_action_feature_mask'],
                                        music_feature_mask = batch['music_feature_mask'],
                                        video_info_mask = batch['video_info_mask'], cls_id = batch['cls_id'],
                                        sep_id = batch['sep_id'], label = label)
        final_loss += loss * batch_size
        final_logits = torch.cat([final_logits,logits], dim=0)
        final_pred = torch.cat([final_pred, pred], dim=0)
    final_loss = final_loss/data_num
    outputs = dict({"inference": final_pred, 'loss': final_loss, 'logits': final_logits})


    path = os.path.join(INFERENCE_SAVE_PATH, SAVE_FILE_NAME)
    with open(path, 'w') as f:
        json.dump(outputs, f)
    print('Outputs are saved in {}.'.format(path))





