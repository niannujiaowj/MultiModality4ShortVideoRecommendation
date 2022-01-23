import os
import argparse
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from scripts.models import dateset_loader, model_loader



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_bert_model', type=str, default='bert-base-chinese')
    parser.add_argument('--train_set', type=str, default='./data/datasets/train.csv')
    parser.add_argument('--valid_set', type=str, default='./data/datasets/valid.csv')
    parser.add_argument('--test_set', type=str, default='./data/datasets/text.csv')
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
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default= 0.001)
    parser.add_argument('--early_stopping_patience', type=int, default=50)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--n_gpu', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=12)
    parser.add_argument('--checkpoints_save_path', type=str, default='./checkpoints')
    parser.add_argument('--logs_save_path', type=str, default='./logs')
    args = parser.parse_args()


    CHECKPOINTS_SAVE_PATH = args.checkpoints_save_path
    LOGS_SAVE_PATH = args.logs_save_path

    if not os.path.exists(CHECKPOINTS_SAVE_PATH):
        os.mkdir(CHECKPOINTS_SAVE_PATH)

    if not os.path.exists(LOGS_SAVE_PATH):
        os.mkdir(LOGS_SAVE_PATH)


    train_DataFrame = pd.read_csv(args.train_set, skip_blank_lines=True)
    valid_DataFramce = pd.read_csv(args.dev_set, skip_blank_lines=True)
    test_DataFramce = pd.read_csv(args.test_set, skip_blank_lines=True)

    data_module = dateset_loader.MM4SVRDataModuel(pretrained_bert_model = args.pretrained_bert_model,
                                               train_DataFrame = train_DataFrame, valid_DataFrame = valid_DataFramce,
                                               test_DataFrame = test_DataFramce, batch_size = args.batch_size,
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
    model = model_loader.MM4SVRModel(pretrained_bert_model = args.pretrained_bert_model,
                                     user_id_vocab_size = args.user_id_vocab_size,
                                     author_id_vocab_size = args.author_id_vocab_size,
                                     hidden_dim = args.hidden_dim,
                                     drop_out = args.drop_out, learning_rate = args.learning_rate)

    logger = TensorBoardLogger(LOGS_SAVE_PATH, name="my_model")
    checkpoint_callback = ModelCheckpoint(dirpath = CHECKPOINTS_SAVE_PATH,
                                          filename='MM4SVR_checkpoint_{epoch:02d}_{val_loss:.2f}',
                                          save_top_k = 1, verbose = True, monitor = 'val_loss', mode = 'min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=args.early_stopping_patience, verbose=True)
    trainer = pl.Trainer(callbacks = [checkpoint_callback, early_stopping], max_epochs = args.n_epochs,
                         gpus = args.n_gpu, logger=logger)
    trainer.fit(model, data_module)
    trainer.test(model, data_module)












