import argparse
from video_preprocessing import videoPreprocess
from audio_preprocessing import audioPreprocess
from dataset_generation import prepare_data



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('videos_path', type=str, default='./data/videos')
    parser.add_argument('musics_path', type=str, default='./data/audios/musics')
    parser.add_argument('interactions_information_path', type=str, default='./data/interactions_information.csv')
    parser.add_argument('videos_information_path', type=str, default='./data/videos_information.csv')
    parser.add_argument('--S3D_pretrained_model_path', type = str, default='./s3d_howto100m.pth')
    parser.add_argument('--videos_semantic_feature_save_path', type = str, default = './data/features/videos_semantics')
    parser.add_argument('--videos_actions_feature_save_path', type = str, default = './data/features/videos_actions')
    parser.add_argument('--key_images_save_path', type = str, default = './data/images')
    parser.add_argument('--subtitles_save_path', type=str, default='./data/subtitles')
    parser.add_argument('--musics_feature_save_path', type=str, default='./data/features/musics')
    parser.add_argument('--datasets_save_path', type=str, default='./data/datasets')
    parser.add_argument('--train_propotion', type=float, default=0.7)
    parser.add_argument('--valid_proportion', type=float, default=0.15)
    args = parser.parse_args()

    videos = videoPreprocess(args.videos_path)

    print('Extracting videos semantic features...')
    videos.videoSenmanticFeatureExtraction(args.videos_semantic_feature_save_path, args.S3D_pretrained_model_path)
    print('Videos semantic features are saved in {}/videos_semantic_features.hdf5'.format(args.videos_semantic_feature_save_path))

    print('Extracting videos action features...')
    videos.videoActionFeatuerExtraction(args.videos_actions_feature_save_path)
    print('Videos action features are saved in {}'.format(args.videos_actions_feature_save_path))

    print('Extracting key images from videos for following subtitle recognition...')
    videos.extractImagesFromVideos(args.key_images_save_path)
    print('Key images are saved in {}'.format(args.key_images_save_path))

    print('Extracting subtitles from key images...')
    videos.extractSubtitlesFromImages(args.subtitles_save_path)
    print('Subtitles are saved in {}'.format(args.subtitles_save_path))

    print('Extracting musics features...')
    musics = audioPreprocess(args.musics_path)
    musics.musicFeatureExtraction(args.musics_feature_save_path)
    print('Data is saved in {}/musics_features.hdf5'.format(args.musics_feature_save_path))

    print('Generating Datasets...')
    n_train, n_valid, n_test = prepare_data(interactions_information_file = args.interactions_information_path,
                 videos_information_file = args.videos_information_path,
                 videos_actions_features_folder = args.videos_actions_feature_save_path,
                 videos_semantic_features_file = args.videos_semantic_feature_save_path + 'videos_semantic_features.hdf5',
                 musics_features_folder = args.musics_feature_save_path + '/musics_features.hdf5',
                 subtitles_folder = args.subtitles_save_path,
                 train_propotion = args.train_propotion,
                 valid_proportion = args.valid_proportion,
                 datasets_save_path = args.datasets_save_path)
    print("Datasets are saved in {}".format(args.datasets_save_path))
    print("The number of data in train, valid and test set is {}, {}, {}".format(n_train, n_valid, n_test))