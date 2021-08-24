from collections import defaultdict
import random
from FeatureExtractor import featureextractor
from ModelRun import modelrun

random.seed = 42  # Fixing randomness
images_dirs = ["/home/student/dvir/ML2_Project/Images/Train_Images/", "/home/student/dvir/ML2_Project/Grey_Images/Train_Images/"]
# images_dirs = ["/home/student/dvir/ML2_Project/Images/Train_Images/"]
tokens_dir = "Flickr8k_text/Flickr8k.token.txt"
dataset_path = 'dataset'
image_height = 299
image_width = 299
batch_size = 64


if __name__ == "__main__":
    feature_extractor = featureextractor()
    params_dict = defaultdict(int)
    params_dict['images_dirs'] = images_dirs
    params_dict['tokens_dir'] = tokens_dir
    params_dict['batch_size'] = batch_size
    params_dict['dataset_path'] = dataset_path
    params_dict['embedding_dim'] = 300
    params_dict['dims'] = 512
    # Shape feature vectors == (64, 2048)
    params_dict['features_shape'] = 2048
    params_dict['attention_features_shape'] = 64

    model_runner = modelrun(params_dict, feature_extractor, load_images=True)
    model_runner.train(epochs=30)