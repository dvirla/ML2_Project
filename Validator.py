import os
import csv
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu
from ModelRun import modelrun
from FeatureExtractor import featureextractor
import random
from tqdm import tqdm


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
            # if img_file_name[:5] == 'grey_':
            #     self.imgs_list.append(img_file_name)
            #     self.imgs_to_caption_list_dict[img_file_name] = all_imgs_to_captions_dict[img_file_name[:5]]
            #     continue
            self.imgs_list.append(img_file_name)
            self.imgs_to_caption_list_dict[img_file_name] = all_imgs_to_captions_dict[img_file_name]

        return

    def bleu_on_set(self, prints=False):
        bleu = [0, 0, 0, 0]
        counter = [0, 0, 0, 0]
        with open('bleu_scores_colorized_model_on_grey_images.txt', 'w') as f:
            for img in tqdm(self.imgs_list):
                predicted_caption, _ = self.model.perdict_caption(image_path=f'{self.imgs_path}{img}')
                real_captions = self.imgs_to_caption_list_dict[img]
                f.write(f'image: {img}, Predicted caption: {predicted_caption}\n')
                f.write(f'image: {img}, Real captions: {",".join(real_captions)}\n')
                for i in range(4):
                    bleu_on_sentence = sentence_bleu(real_captions, predicted_caption,
                                                     weights=tuple([1.0 if i == j else 0.0 for j in range(4)]))
                    if prints and counter[i] % 50 == 0:
                        # print('image: ', img, ', Predicted caption: ', predicted_caption)
                        # print('Real captions:\n', real_captions)
                        f.write(f'Bleu{i}, Current Bleu: {bleu_on_sentence}\n')
                        # print('Current Bleu: ', bleu_on_sentence)
                        f.write('****************************************\n')
                        # print('****************************************')
                    bleu[i] += bleu_on_sentence
                    counter[i] += 1
            for i in range(4):
                bleu[i] = bleu[i] / len(self.imgs_list)
            f.write(f'Average Bleu score on set: {bleu}')

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
    model_runner.encoder.load_weights('/home/student/dvir/ML2_Project/encoder_weights_with_colorized/encoder_weight_13.ckpt')
    model_runner.decoder.built = True
    model_runner.decoder.load_weights('/home/student/dvir/ML2_Project/decoder_weights_with_colorized/decoder_weights_13.ckpt')

    validator = Validator(model_runner, imgs_path='/home/student/dvir/ML2_Project/Grey_Images/Test_Images/',
                          reference_path=tokens_dir)
    bleu = validator.bleu_on_set(prints=True)
    print('***************************************')
    print('***************************************')
    print('Bleu score on set: ', bleu)
