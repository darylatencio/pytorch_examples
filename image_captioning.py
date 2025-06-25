#--------------------------------------------------------------------------------------------------
#+
#
# This example uses the 2017 coco data which can be downloaded from:
#
#   training data: http://images.cocodataset.org/zips/train2017.zip
#   image captions: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
#   validation data: http://images.cocodataset.org/zips/val2017.zip
#-

import matplotlib.pyplot as plot
import nltk
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
from collections import Counter
from PIL import Image
from pycocotools.coco import COCO
from pynvml import *
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

###################################################################################################
#--------------------------------------------------------------------------------------------------
#+
#-
class decoderRNN(nn.Module):

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        super(decoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        output = self.linear(hiddens[0])
        return output

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def sample(self, feat, states=None):
        id_sample = []
        input = feat.unsqueeze(1)
        for i in range(self.max_seg_length):
            hid, states = self.lstm(input, states)
            output = self.linear(hid.squeeze(1))
            _, pred = output.max(1)
            id_sample.append(pred)
            input = self.embed(pred)
            input = input.unsqueeze(1)
        id_sample = torch.stack(id_sample, 1)
        return id_sample

###################################################################################################
#--------------------------------------------------------------------------------------------------
#+
#-
class encoderCNN(nn.Module):

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def __init__(self, embed_size):
        super(encoderCNN, self).__init__()
        resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def forward(self, img):
        with torch.no_grad():
            features = self.resnet(img)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

###################################################################################################
#--------------------------------------------------------------------------------------------------
#+
#-
class ic_dataset(Dataset):

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def __getitem__(self, i):
        id_ann = self.ids[i]
        caption = self.coco.anns[id_ann]["caption"]
        id_img = self.coco.anns[id_ann]["image_id"]
        base = self.coco.loadImgs(id_img)[0]["file_name"]
        img = Image.open(os.path.join(self.dir, base)).resize(
            self.img_size, Image.LANCZOS).convert("RGB")
        if self.transform != None:
            img = self.transform(img)
        tok = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab("<start>"))
        caption.extend([self.vocab(t) for t in tok])
        caption.append(self.vocab("<end>"))
        target = torch.Tensor(caption)
        return img, target

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def __init__(self, dir_image, file_caption, vocab,
                 image_size=(256,256), verbose=False, transform=None):
        self.dir = dir_image
        self.coco = COCO(file_caption)
        self.ids = list(self.coco.anns.keys())
        self.img_size = image_size
        self.verbose = verbose
        self.vocab = vocab
        self.transform = transform

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def __len__(self):
        return len(self.ids)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def print(self, str):
        if not self.verbose:
            return None
        print("[IC DATASET] "+str)

###################################################################################################
#--------------------------------------------------------------------------------------------------
#+
#-
class image_captioning():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_test = None
    tf_normalize = transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def __init__(self, dir_train, file_caption, batch_size=128, crop_size=224, decoder_file=None,
                 embed_size=256, encoder_file=None, hidden_size=512, learning_rate=0.001,
                 num_layers=1, num_workers=2, temperature_threshold=100, test_folder=None,
                 verbose=False):
        self.crop_size = crop_size
        self.dir_test = test_folder
        self.dir_train = dir_train
        self.embed_size = embed_size
        self.file_caption = file_caption
        self.thresh_temp = temperature_threshold
        self.verbose = verbose
        self.initialize_dataloader(batch_size=batch_size, num_workers=num_workers)
        self.initialize_model(decoder_file=decoder_file, encoder_file=encoder_file,
                              hidden_size=hidden_size, learning_rate=learning_rate,
                              num_layers=num_layers)
        self.initialize_temperature_monitor()

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def get_temperature(self):
        if (self.device == "cpu"):
            return "not using GPU"
        h = nvmlDeviceGetHandleByIndex(0)
        return nvmlDeviceGetTemperature(h, NVML_TEMPERATURE_GPU)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def initialize_dataloader(self, batch_size=128, num_workers=2, shuffle=True):
        self.print("initializing dataloader...")
        file_pickle = os.path.splitext(self.file_caption)[0]+".pkl"
        if os.path.exists(file_pickle):
            self.print("  loading vocabulary from pickle")
            with open(file_pickle, "rb") as f:
                self.vocab = pickle.load(f)
        else:
            print("  creating vocabulary")
            self.vocab = vocabulary(self.file_caption, verbose=self.verbose)
        tf = transforms.Compose([transforms.RandomCrop(self.crop_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 self.tf_normalize])
        ds = ic_dataset(self.dir_train, self.file_caption, self.vocab,
                        transform=tf, verbose=self.verbose)
        self.dataloader = DataLoader(batch_size=batch_size, collate_fn=collate_fn, dataset=ds,
                                     num_workers=num_workers, shuffle=shuffle)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def initialize_model(self, decoder_file=None, encoder_file=None,
                         hidden_size=512, learning_rate=0.001, num_layers=1):
        self.print("initializing model...")
        self.fn_loss = nn.CrossEntropyLoss()
        self.decoder = decoderRNN(self.embed_size, hidden_size, len(self.vocab),
                                  num_layers).to(self.device)
        self.file_decoder = \
            os.path.join(os.path.dirname(__file__), "model", "ic_decoder.dct") \
            if (decoder_file == None) else decoder_file
        if os.path.exists(self.file_decoder):
            self.print(f"  loading existing decoder dictionary: {self.file_decoder}")
            self.decoder.load_state_dict(torch.load(self.file_decoder))
        self.encoder = encoderCNN(self.embed_size).to(self.device)
        self.file_encoder = \
            os.path.join(os.path.dirname(__file__), "model", "ic_encoder.dct") \
            if (encoder_file == None) else encoder_file
        if os.path.exists(self.file_encoder):
            self.print(f"  loading existing encoder dictionary: {self.file_encoder}")
            self.encoder.load_state_dict(torch.load(self.file_encoder))
        params = list(self.decoder.parameters()) + \
                 list(self.encoder.linear.parameters()) + \
                 list(self.encoder.bn.parameters())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def initialize_temperature_monitor(self):
        nvmlInit()

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def print(self, str):
        if not self.verbose:
            return None
        print("[IC MODEL] "+str)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def save(self, decoder_file=None, encoder_file=None):
        file_decoder = self.file_decoder if (decoder_file == None) else decoder_file
        file_encoder = self.file_encoder if (encoder_file == None) else encoder_file
        self.print(f"saving model files:\n  {file_decoder}\n  {file_encoder}")
        torch.save(self.decoder.state_dict(), file_decoder)
        torch.save(self.encoder.state_dict(), file_encoder)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def test(self, file=None):
        if (file == None):
            if (self.dir_test == None) or (not os.path.exists(self.dir_test)):
                self.print("no valid testing input")
                return None
            base = os.listdir(self.dir_test)
            base = base[np.random.randint(0,len(base)-1)]
            file = os.path.join(self.dir_test, base)
        tf = transforms.Compose([transforms.ToTensor(), self.tf_normalize])
        img = Image.open(file).convert("RGB").resize(
            [self.crop_size, self.crop_size], Image.LANCZOS)
        img = tf(img).unsqueeze(0)
        if (self.encoder_test == None):
            self.encoder_test = encoderCNN(self.embed_size).eval().to(self.device)
            self.encoder_test.load_state_dict(torch.load(self.file_encoder))
        img = img.to(self.device)
        feature = self.encoder_test(img)
        id_sample = self.decoder.sample(feature)
        id_sample = id_sample[0].cpu().numpy()
        caption = []
        for id_word in id_sample:
            w = self.vocab.i2w[id_word]
            caption.append(w)
            if (w == "<end>"):
                break
        caption = ' '.join(caption[1:len(caption)-1])
        img = Image.open(file)
        fig, grid = plot.subplots(1,1, num="Image Caption Test")
        grid.imshow(np.asarray(img))
        grid.text(0.0, 1.01, caption, color="#000000", transform=grid.transAxes)
        grid.axis("off")
        plot.show()

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def train(self, num_epoch=5, save_interval=100, save_model=False):
        self.print("training model...")
        n_step = len(self.dataloader)
        for i_epoch in range(num_epoch):
            for i_step, (img, caption, length) in enumerate(self.dataloader):
                img = img.to(self.device)
                caption = caption.to(self.device)
                tgt = pack_padded_sequence(caption, length, batch_first=True)[0]
                feature = self.encoder(img)
                output = self.decoder(feature, caption, length)
                loss = self.fn_loss(output, tgt)
                self.decoder.zero_grad()
                self.encoder.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (save_model and ((i_step+1) % save_interval)) == 0:
                    self.save()
                gpu_temp = self.get_temperature()
                if (i_step+1) % 10 == 0:
                    self.print(f"  epoch: [{i_epoch}/{num_epoch}] step: [{i_step+1}/{n_step}] "+
                               f"loss: {loss.item():.4f} perplexity: {np.exp(loss.item()):.4f} "+
                               f"gpu-temperature (C): {gpu_temp}")
                if (gpu_temp > self.thresh_temp):
                    print(f"temperature threshold ({self.thresh_temp}) exceeded: {gpu_temp}.\n"+
                          "exiting.")
                    return None

        if save_model:
            self.save()

###################################################################################################
#--------------------------------------------------------------------------------------------------
#+
#-
class vocabulary(object):

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def __call__(self, w):
        if w in self.w2i:
            return self.w2i[w]
        return self.w2i["<unknown>"]

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def __init__(self, file, save_pickle=True, threshold=4, verbose=False):
        self.initialize_state()
        self.thresh = threshold
        self.verbose = verbose
        self.parse_file(file, save_pickle=save_pickle)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def __len__(self):
        return len(self.w2i)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def add_word(self, w):
        if not w in self.w2i:
            self.w2i[w] = self.i
            self.i2w[self.i] = w
            self.i += 1

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def initialize_state(self):
        self.w2i = {}
        self.i2w = {}
        self.i = 0
        for w in ["<pad>","<start>","<end>","<unknown>"]:
            self.add_word(w)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def parse_file(self, file, threshold=None, save_pickle=True):
        if not os.path.exists(file):
            print(f"invalid caption file: {file}")
            return None
        print('tokenizing captions...')
        thresh = self.thresh if (threshold == None) else threshold
        coco = COCO(file)
        counter = Counter()
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
            if (i+1) % 1000 == 0:
                self.print(f"  [{i+1}/{len(ids)}]")
        for w, cnt in counter.items():
            if (cnt > thresh):
                self.add_word(w)
        if save_pickle:
            pickle_file = os.path.splitext(file)[0]+".pkl"
            self.print(f"saving wrapper: {pickle_file}")
            with open(pickle_file, "wb") as f:
                pickle.dump(self, f)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def print(self, str):
        if not self.verbose:
            return None
        print("[VOCABULARY] "+str)

###################################################################################################
# stand-alone functions
###################################################################################################

#--------------------------------------------------------------------------------------------------
#+
#-
def collate_fn(data):
    data.sort(key=lambda x:len(x[1]), reverse=True)
    img, caption = zip(*data)
    img = torch.stack(img, 0)
    l = []
    for c in caption:
        l.append(len(c))
    targets = torch.zeros(len(caption), max(l)).long()
    for i, c in enumerate(caption):
        end = l[i]
        targets[i,:end] = c[:end]
    return img, targets, l

#--------------------------------------------------------------------------------------------------
#+
#-
def test_image_captioning():
    print("testing image-captioning model...")
    file_caption = None
# data folder
    dir_data = "C:\\data\\coco"
# images folders
    dir_train = os.path.join(dir_data, "train2017") # contains training images
    dir_test = os.path.join(dir_data, "val2017") # contains testing/validation images
# caption file
    dir_annotation = os.path.join(dir_data, "annotations")
    for base in os.listdir(dir_annotation):
        if ("captions" in base) and (".json" in base):
            if ("train" in base):
                file_caption = os.path.join(dir_annotation, base)
            if ("val" in base):
                file_val = os.path.join(dir_annotation, base)
    if (file_caption == None):
        print("  no caption file found")
        return None
    print(f"  caption file: {file_caption}")
    ic = image_captioning(dir_train, file_caption, temperature_threshold=85,
                          test_folder=dir_test, verbose=True)
    ic.train(num_epoch=1, save_model=True)
    ic.test()


#--------------------------------------------------------------------------------------------------
#+
# main entry point
#-
if (__name__ == "__main__"):
    test_image_captioning()
