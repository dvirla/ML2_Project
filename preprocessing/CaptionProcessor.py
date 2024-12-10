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
        """
        Preprocessing every caption from the train set: lowercasing, removing punctuation, removing words of one length
        e.g 'a', adding 'startcap' & 'endcap' to each caption to signal for start and end of a caption later.
        Generating two data structures for later:
        1. self.all_captions = a list of all processed captions
        2. self.processed_img_to_captions_dict = a dict with images as keys and the values are lists of processed captions
        supplied for the key image.
        In addition, maintaining the self.max_caption_len for future use of the maximum caption length.
        """
        for img, caption_list in self.img_to_captions_dict.items():
            processed_captions = []
            for caption in caption_list:
                processed_caption = re.sub(r"[^a-zA-Z0-9]+", ' ', caption.lower())
                processed_caption_list = [word for word in processed_caption.split(' ') if
                                          ((len(word) > 1) and (word.isalpha()))]  # TODO: check without this removal
                processed_caption_list = ['startcap'] + processed_caption_list + ['endcap']
                self.max_caption_len = max([self.max_caption_len, len(processed_caption_list)])
                processed_caption = ' '.join(processed_caption_list)
                self.all_captions.append(processed_caption)
                processed_captions.append(processed_caption)
            self.processed_img_to_captions_dict[img] = processed_captions

    def tokenizer_fit(self):
        """
        Training a tokenizer over all processed captions and maintaining the vocabulary size for future use.
        """
        self.tokenizer.fit_on_texts(self.all_captions)
        self.vocab_size = len(self.tokenizer.word_index) + 1

    def caption_padding(self):
        """
        Padding each caption from the train set to make all of them be of length = self.max_caption_len
        """
        for img, caption_list in self.processed_img_to_captions_dict.items():
            for caption in caption_list:
                caption_idxs = self.tokenizer.texts_to_sequences([caption])[0]
                padded_idxs = pad_sequences([caption_idxs], maxlen=self.max_caption_len, padding='post')[0]
                self.padded_captions_idxs_dict[img].append(padded_idxs)
