import os
import csv
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu
from ModelRun import modelrun
from FeatureExtractor import featureextractor
import random

class Validator:
    def __init__(self, model, imgs_path, reference_path):
        self.model = model
        self.imgs_path = imgs_path
        self.imgs_list = []
        self.reference_path = reference_path
        self.imgs_to_caption_list_dict = {}

        self.load_captions()

    def load_captions(self):
        all_imgs_to_captions_dict = defaultdict(list)
        with open(self.reference_path) as f:
            csvreader = csv.reader(f, delimiter='\t')
            for row in csvreader:
                img_caption, caption = row
                img = img_caption.split('#')[0]
                all_imgs_to_captions_dict[f'{img}'].append(caption)

        for img_file_name in os.listdir(self.imgs_path):
            if img_file_name[:5] == 'grey_':
                self.imgs_list.append(img_file_name)
                self.imgs_to_caption_list_dict[img_file_name] = all_imgs_to_captions_dict[img_file_name[5:]]
                continue
            self.imgs_list.append(img_file_name)
            self.imgs_to_caption_list_dict[img_file_name] = all_imgs_to_captions_dict[img_file_name]

        return

    def bleu_on_set(self, prints=False):
        bleu = 0
        for img in self.imgs_list:
            predicted_caption, _ = self.model.perdict_caption(self.imgs_path+img)
            real_captions = self.imgs_to_caption_list_dict[img]
            bleu_on_sentence = sentence_bleu(real_captions, predicted_caption)
            if prints:
                print('Predicted caption: ', predicted_caption)
                print('Real captions:\n', real_captions)
                print('Current Bleu: ', bleu_on_sentence)
                print('****************************************')
            bleu += bleu_on_sentence

        bleu = bleu / len(self.imgs_list)

        return bleu

random.seed = 42  # Fixing randomness
images_dir = "/home/student/dvir/ML2_Project/Images/Flicker8k_Dataset/"
tokens_dir = "Flickr8k_text/Flickr8k.token.txt"
dataset_path = 'dataset'
image_height = 299
image_width = 299
batch_size = 64


if __name__ == "__main__":
    feature_extractor = featureextractor()
    params_dict = defaultdict(int)
    params_dict['images_dir'] = images_dir
    params_dict['tokens_dir'] = tokens_dir
    params_dict['batch_size'] = batch_size
    params_dict['dataset_path'] = dataset_path
    params_dict['embedding_dim'] = 300
    params_dict['dims'] = 512
    # Shape feature vectors == (64, 2048)
    params_dict['features_shape'] = 2048
    params_dict['attention_features_shape'] = 64

    model_runner = modelrun(params_dict, feature_extractor, load_images=False)
    model_runner.encoder.built = True
    model_runner.encoder.load_weights('/home/student/dvir/ML2_Project/encoder_weights/encoder_weight_4.ckpt')
    model_runner.decoder.built = True
    model_runner.decoder.load_weights('/home/student/dvir/ML2_Project/decoder_weights/decoder_weights_4.ckpt')


    validator = Validator(model_runner, imgs_path='/home/student/dvir/ML2_Project/colorizing_test_imgs/', reference_path=tokens_dir)
    bleu = validator.bleu_on_set(prints=True)
    print('***************************************')
    print('***************************************')
    print('Bleu score on set: ', bleu)