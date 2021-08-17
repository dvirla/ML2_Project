from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from collections import defaultdict


class captionprocessor():
    def __init__(self, img_to_captions_dict, train_imgs=None, test_imgs=None):
        # general variables
        self.img_to_captions_dict = img_to_captions_dict
        self.processed_img_to_captions_dict = defaultdict(list)
        self.padded_captions_idxs_dict = defaultdict(list)
        self.all_captions = []
        self.tokenizer = Tokenizer()
        self.max_caption_len = 0
        self.vocab_size = 0
        self.train_imgs = train_imgs
        self.test_imgs = test_imgs

        # functions
        self.captions_preprocess()
        self.tokenizer_fit()
        self.caption_padding()

    def captions_preprocess(self):
        for img, caption_list in self.img_to_captions_dict.items():
            processed_captions = []
            for caption in caption_list:
                # allow only word and numbers
                processed_caption = re.sub(r"[^a-zA-Z0-9]+", ' ', caption.lower())
                #  allow only words of more than one letter
                processed_caption_list = [word for word in processed_caption.split(' ') if
                                          ((len(word) > 1) and (word.isalpha()))]
                #  add beginning and ending of caption tokens
                processed_caption_list = ['startcap'] + processed_caption_list + ['endcap']
                self.max_caption_len = max([self.max_caption_len, len(processed_caption_list)])
                processed_caption = ' '.join(processed_caption_list)
                self.all_captions.append(processed_caption)
                processed_captions.append(processed_caption)
            self.processed_img_to_captions_dict[img] = processed_captions

    def tokenizer_fit(self):
        self.tokenizer.fit_on_texts(self.all_captions)
        self.vocab_size = len(self.tokenizer.word_index) + 1

    def caption_padding(self):
        for img, caption_list in self.processed_img_to_captions_dict.items():
            for caption in caption_list:
                caption_idxs = self.tokenizer.texts_to_sequences([caption])[0]
                padded_idxs = pad_sequences([caption_idxs], maxlen=self.max_caption_len, padding='post')[
                    0]  # todo: maybe value of the padding from zero?
                self.padded_captions_idxs_dict[img].append(padded_idxs)

    def split_dic_to_train_set(self):
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for img in self.train_imgs:
            for padded_idx in self.padded_captions_idxs_dict[img]:
                X_train.append(img)
                y_train.append(padded_idx)

        for img in self.test_imgs:
            for padded_idx in self.padded_captions_idxs_dict[img]:
                X_test.append(img)
                y_test.append(padded_idx)

        return X_train, y_train, X_test, y_test
