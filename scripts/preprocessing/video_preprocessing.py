import os
import re
from tqdm import tqdm

import torch
import pandas as pd
import cv2

from utils import load_video

class videoPreprocess:
    def __init__(self, video_folder_path):
        self.video_folder_path = video_folder_path

    def videoSenmanticFeatureExtraction(self, feature_folder_path = './data/features/videos_semantics',
                                        S3D_pretrained_model_path = './pretrained_models/S3D_HowTo100M/s3d_howto100m.pth'):
        '''
        :param feature_folder_path: output feature folder path
        :param S3D_pretrained_model_path: file path to S3D pretrained model
        :return: one h5py file {'videoID': videoFeatureArray}, each array has size of 512
        '''
        # https://github.com/antoine77340/S3D_HowTo100M
        from s3dg import S3D
        import h5py

        if not os.path.exists(feature_folder_path):
            os.makedirs(feature_folder_path)
        net = S3D('/s3d_dict.npy', 512)
        net.load_state_dict(torch.load(S3D_pretrained_model_path))
        output_path = feature_folder_path + "/videos_semantic_features.hdf5"

        with h5py(output_path, 'a') as f:
            for video in tqdm(os.listdir(self.video_folder_path)):
                if re.findall(r"(.+)\.", video) != []:
                    video_frames = load_video(video_path = self.video_folder_path+"/"+video) # (T * H * W * 3)
                    video_frames_array = video_frames.reshape(1, 3, 32, 224, 224)  # (batch size * 3 * T * H * W)
                    net = net.eval()
                    video_embeddings = net(torch.from_numpy(video_frames_array).to(torch.float32))['video_embedding']  # (batch size * 512)
                    f.attrs[re.findall(r"(.+)\.", video)[0]] = video_embeddings.flatten().detach().numpy()


    def videoActionFeatuerExtraction(self, feature_folder_path = "./data/features/videos_actions"):
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

        parent_path = re.findall(r'(.*)/[^/]+$',feature_folder_path)[0]
        temp_file_path = parent_path + '/videoFilePath4FeatureExtraction.csv'
        df.to_csv(temp_file_path)
        # https://github.com/antoine77340/video_feature_extractor
        os.system("python ./video_feature_extractor/extract.py "
                  "--csv={} "
                  "--type=3d "
                  "--batch_size=3 "
                  "--num_decoding_thread=4".format(temp_file_path))

    def extractImagesFromVideos(self, output_image_path = './data/images'):
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

    def extractSubtitlesFromImages(self, output_text_path = './data/subtitles', use_gpu = False):
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










