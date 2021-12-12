import os
import re
from tqdm import tqdm

import torch

import pandas as pd
import numpy as np
import cv2
import math

class videoPreprocess:
    def __init__(self, video_folder_path):
        self.video_folder_path = video_folder_path

    # Utilities to open video files using CV2
    def __crop_center_square(self, frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

    def __load_video(self, video_path, max_frames=32, resize=(224, 224)):
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.__crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

                if len(frames) == max_frames:
                    break
        finally:
            cap.release()
        frames = np.array(frames)
        if len(frames) < max_frames:
            n_repeat = int(math.ceil(max_frames / float(len(frames))))
            frames = frames.repeat(n_repeat, axis=0)
        frames = frames[:max_frames]
        return frames / 255.0

    def videoSenmanticFeatureExtraction(self, feature_folder_path = "data/features/videos_semantics"):
        '''
        :param feature_folder_path: output feature folder path
        :return: one h5py file {'videoID': videoFeatureArray}, each array has size of 512
        '''
        # https://github.com/antoine77340/S3D_HowTo100M
        from s3dg import S3D
        import h5py

        if not os.path.exists(feature_folder_path):
            os.makedirs(feature_folder_path)
        net = S3D('/Users/alicewong/Documents/Programming/S3D_HowTo100M/s3d_dict.npy', 512)
        net.load_state_dict(torch.load('/Users/alicewong/Documents/Programming/S3D_HowTo100M/s3d_howto100m.pth'))
        output_path = feature_folder_path + "/video_semantic_features.hdf5"

        with h5py(output_path, 'a') as f:
            for video in tqdm(os.listdir(self.video_folder_path)):
                if re.findall(r"(.+)\.", video) != []:
                    video_frames = self.__load_video(self.video_folder_path+"/"+video) # (T * H * W * 3)
                    video_frames_array = video_frames.reshape(1, 3, 32, 224, 224)  # (batch size * 3 * T * H * W)
                    net = net.eval()
                    video_embeddings = net(torch.from_numpy(video_frames_array).to(torch.float32))['video_embedding']  # (batch size * 512)
                    f.attrs[re.findall(r"(.+)\.", video)[0]] = video_embeddings.flatten().detach().numpy()


    def video3DFeatuerExtraction(self, feature_folder_path = "data/features/videos_style"):
        '''
        :param feature_folder_path: output feature folder path
        :return: files of video feature arrays, 1 file (video length * 2048)
        '''
        video_list = []
        feature_list = []
        if not os.path.exists(feature_folder_path):
            os.makedirs(feature_folder_path)
        for video in tqdm(os.listdir(self.video_folder_path)):
            if re.findall(r"(.+)\.", video) != []:
                video_list.append(self.video_folder_path+"/"+video)
                feature_list.append(feature_folder_path+"/"+re.findall(r"(.+)\.",video)[0]+".npy")
        df = pd.DataFrame(
            {
                "video_path": pd.Series(video_list, dtype=str),
                "feature_path": pd.Series(feature_list, dtype=str),
            }
        )
        df.to_csv("data/features/videoFilePath4FeatureExtraction.csv")
        # https://github.com/antoine77340/video_feature_extractor
        os.system("python /Users/alicewong/Documents/Programming/video_feature_extractor/extract.py "
                  "--csv=data/features/videoFilePath4FeatureExtraction.csv "
                  "--type=3d "
                  "--batch_size=3 "
                  "--num_decoding_thread=4")

    def extractImagesFromVideos(self, output_image_path = 'data/images'):
        '''
        :param output_image_path: output image folder path
        :return: extracted images from videos, one video one folder
        '''
        # https://aistudio.baidu.com/aistudio/projectdetail/1217163?channelType=0&channel=0
        self.output_image_path = output_image_path
        if not os.path.exists(self.output_image_path):
            os.makedirs(self.output_image_path)

        for video in tqdm(os.listdir(self.video_folder_path)):
            if re.findall(r"(.+)\.",video) != []:
                image_folder = self.output_image_path+"/"+re.findall(r"(.+)\.",video)[0]
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                vc = cv2.VideoCapture(self.video_folder_path+"/"+video)
                n = 1

                # extract images from videos
                if vc.isOpened():  # Check whether the video is properly opened
                    rval, frame = vc.read()
                else:
                    rval = False

                timeF = 10  # Video frame interval frequency

                i = 0
                while rval:  # Read video frames in loop
                    rval, frame = vc.read()
                    if (n % timeF == 0):  # save images every timeF frame
                        i += 1
                        cv2.imwrite(image_folder+"/"+'{}.jpg'.format(i), frame)
                    n = n + 1
                    cv2.waitKey(1)
                vc.release()

    def extractSubtitlesFromImages(self, output_text_path = 'data/text/subtitles', use_gpu = False):
        '''
        :param output_text_path: output subtitle folder path
        :param use_gpu
        :return: extracted subtitles from video images, one video one txt file
        '''
        # https://www.paddlepaddle.org.cn/hubdetail?name=chinese_ocr_db_crnn_server&en_category=TextRecognition
        if not os.path.exists(output_text_path):
            os.makedirs(output_text_path)

        # extract subtitles from images
        import paddlehub as hub
        ocr = hub.Module(name="chinese_ocr_db_crnn_server")
        for video in tqdm(os.listdir(self.output_image_path)):
            txt = open(output_text_path+"/"+video+".txt", mode='a')
            for image in tqdm(os.listdir(self.output_image_path+"/"+video)):
                result = ocr.recognize_text(paths=[self.output_image_path+"/"+video+"/"+image], use_gpu=use_gpu,
                                            output_dir='ocr_result')  # result = [{"path":,{"data":{"text":},{"text":}}]
                subtitles = result[0]['data']
                for file in subtitles:
                    subtitle = file["text"]
                    if re.findall(r"抖音", subtitle) == []:
                        txt.write(subtitle + '。')
            txt.close()

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



class audioPreprocess:
    def musicFeatureExtraction(self, music_folder_path = "data/audios/musics",feature_folder_path = "data/features/musics"):
        '''
        :param music_folder_path: input music folder path
        :param feature_folder_path: output feature folder path
        :return: one h5py file {'musicID': musicFeatureArray}, each array has size of (sequence_length * 768)
        '''
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        import librosa
        import h5py

        if not os.path.exists(feature_folder_path):
            os.makedirs(feature_folder_path)

        output_path = feature_folder_path + "/music_features.hdf5"
        processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        model = Wav2Vec2Model.from_pretrained('m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres')
        with h5py.File(output_path, "a") as f:
            for music in tqdm(os.listdir(music_folder_path)):
                if re.findall(r"(.+)\.", music) != []:
                    array, sampling_rate = librosa.load(music_folder_path+"/"+music, sr=None)
                    inputs = processor(array, sampling_rate=16000, return_tensors="pt")
                    outputs = model(**inputs)
                    last_hidden_states = outputs.last_hidden_state.reshape(-1,768)  # (sequence_length * 768)
                    # ????? RuntimeError: Unable to create attribute (object header message is too large
                    f.attrs[re.findall(r"(.+)\.",music)[0]] = last_hidden_states.detach().numpy()



if __name__=="__main__":
    preprocess = videoPreprocess("./data/videos")
    #preprocess.video3DFeatuerExtraction()

    #preprocess.subtitleExtraction()
    #preprocess.video3DFeatuerExtraction()


    #preprcess = audioPreprocess()
    #preprcess.extractAudiosFromVideos("data/audios/full_audios")
    #preprcess.musicFeatureExtraction()





