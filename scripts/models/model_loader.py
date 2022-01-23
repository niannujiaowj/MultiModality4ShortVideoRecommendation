import torch
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score

from scripts.models.model import MyModel

class MM4SVRModel(pl.LightningModule):
    def __init__(self, pretrained_bert_model: str, user_id_vocab_size: int = 10000000000,
                 author_id_vocab_size: int = 10000000000, hidden_dim: int = 768, drop_out: float = 0.1,
                 learning_rate: float = 0.001):
        super().__init__()
        self.model = MyModel(pretrained_bert_model  = pretrained_bert_model,
                             user_id_vocab_size = user_id_vocab_size,
                             author_id_vocab_size = author_id_vocab_size,
                             hidden_dim = hidden_dim,
                             drop_out = drop_out)
        self.lr = learning_rate


    def forward(self, user_id, user_info, user_info_mask, author_id, author_info, author_info_mask,
                video_live_score, video_hot, video_entertainment_hot, video_social_hot, video_challenge_hot,
                video_author_hot, video_action_feature, video_semantic_feature, music_feature,
                video_info, video_action_feature_mask, music_feature_mask, video_info_mask,
                cls_id, sep_id, label = None):
        loss, logits, pred = self.model(user_id, user_info, user_info_mask, author_id, author_info, author_info_mask,
                video_live_score, video_hot, video_entertainment_hot, video_social_hot, video_challenge_hot,
                video_author_hot, video_action_feature, video_semantic_feature, music_feature,
                video_info, video_action_feature_mask, music_feature_mask, video_info_mask,
                cls_id, sep_id, label = None)
        return loss, logits, pred


    def training_step(self, batch, batch_idx):
        loss, logits, pred = self.model(user_id = batch['user_id'], user_info = batch['user_info'],
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
                                        sep_id = batch['sep_id'], label = batch['label'])
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, pred = self.model(user_id = batch['user_id'], user_info = batch['user_info'],
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
                                        sep_id = batch['sep_id'], label = batch['label'])
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, pred = self.model(user_id = batch['user_id'], user_info = batch['user_info'],
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
                                        sep_id = batch['sep_id'], label = batch['label'])
        auc_score = roc_auc_score(batch['label'], logits[:,1])
        self.log_dict({'test_loss': loss, 'auc_score': auc_score})
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)
