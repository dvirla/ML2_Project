from collections import defaultdict
import random
from FeatureExtractor import featureextractor
from ModelRun import modelrun

random.seed = 42  # Fixing randomness
images_dir = "/home/student/dvir/ML2_Project/Flicker8k_Dataset/"
tokens_dir = "Flickr8k_text/Flickr8k.token.txt"
train_dataset_path = 'train_dataset'
image_height = 299
image_width = 299
batch_size = 64


if __name__ == "__main__":
    feature_extractor = featureextractor()
    params_dict = defaultdict(int)
    params_dict['images_dir'] = images_dir
    params_dict['tokens_dir'] = tokens_dir
    params_dict['batch_size'] = batch_size
    params_dict['train_dataset_path'] = train_dataset_path
    params_dict['embedding_dim'] = 300
    params_dict['dims'] = 512
    # Shape feature vectors == (64, 2048)
    params_dict['features_shape'] = 2048
    params_dict['attention_features_shape'] = 64

    model_runner = modelrun(params_dict, feature_extractor)
    model_runner.train()
    # embedding_dim = 300
    # dims = 512
    # vocab_size = caption_processor.vocab_size  # 8425
    # # num_steps = len(X_train) // BATCH_SIZE
    # # TODO: check if len(train_dataset) == 6000
    # num_steps = 6000 // self.batch_size
    # # Shape feature vectors == (64, 2048)
    # features_shape = 2048
    # attention_features_shape = 64