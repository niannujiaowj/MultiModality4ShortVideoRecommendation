import os
import re
from tqdm import tqdm

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
import h5py


class audioPreprocess:
    def __init__(self, music_folder_path = "./data/audios/musics"):
        self.music_folder_path = music_folder_path

    def musicFeatureExtraction(self, feature_folder_path = "./data/features/musics"):
        '''
        :param feature_folder_path: output feature folder path
        :return: one h5py file {'musicID': musicFeatureArray}, each array has size of (sequence_length * 768)
        '''

        if not os.path.exists(feature_folder_path):
            os.makedirs(feature_folder_path)

        output_path = feature_folder_path + "/musics_features.hdf5"
        processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        model = Wav2Vec2Model.from_pretrained('m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres')
        with h5py.File(output_path, "a") as f:
            for music in tqdm(os.listdir(self.music_folder_path)):
                if re.findall(r"(.+)\.", music) != []:
                    array, sampling_rate = librosa.load(self.music_folder_path+"/"+music, sr=None)
                    inputs = processor(array, sampling_rate=16000, return_tensors="pt")
                    outputs = model(**inputs)
                    last_hidden_states = outputs.last_hidden_state.reshape(-1,768)  # (sequence_length * 768)
                    # TODO RuntimeError: Unable to create attribute (object header message is too large
                    f.attrs[re.findall(r"(.+)\.",music)[0]] = last_hidden_states.detach().numpy()


    # Was intended to separate narrations from raw audios. But many raw audios contain both narrations and vocals from
    # back ground music, no suitable pretrained models found.
    """def extractAudiosFromVideos(self, audio_folder_path="data/audios/full_audios"):
        '''
        :param audio_folder_path: output audio folder path
        :return: extracted raw audios from videos, one video one wav file
        '''
        from moviepy.editor import AudioFileClip
        self.audio_folder_path = audio_folder_path
        if not os.path.exists(self.audio_folder_path):
            os.makedirs(self.audio_folder_path)
        for video in tqdm(os.listdir(self.video_folder_path)):
            audio_clip = AudioFileClip(self.video_folder_path + "/" + video)
            audio_clip.write_audiofile(self.audio_folder_path + "/" + re.findall(r"(.+)\.", video)[0] + ".wav")"""