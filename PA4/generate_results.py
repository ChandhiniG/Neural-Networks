import numpy as np
import torch
import csv
from build_vocab import Vocabulary
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torchvision import transforms
from data_loader import get_loader
from cnn_rnn_fcn import *

vocab = Vocabulary()

def deterministic_generate(network_output, images, device='cpu'):
    """

    :param network_output: (batch x max word length x vocab size) tensor
    :param images: (batch x 3 x W x H) image tensor
    :param device: move to cpu if needed
    :return: tuple of list of images and list of a list of indicies
    """
    batch_size = images.shape[0]
    images = images.to(device).data.numpy()
    network_output = network_output.to(device).data.numpy()

    batch_images = [np.transpose(images[i], (1, 2, 0)) for i in range(batch_size)]
    batch_inds = [np.argmax(network_output[i], axis=1) for i in range(batch_size)]
    return batch_images, batch_inds


def stochastic_generate(network_output, images, image_save='False', device='cpu'):
    """

    :param network_output: (batch x max word length x vocab size) tensor
    :param images: (batch x 3 x W x H) image tensor
    :param device: move to cpu if needed
    :return: tuple of list of images and list of a list of indicies
    """
    batch_size = images.shape[0]
    sentence_len = network_output.shape[1]

    images = images.to(device).data.numpy()

    batch_images = [np.transpose(images[i], (1, 2, 0)) for i in range(batch_size)]
    softmaxed_output = [torch.softmax(network_output[i], dim=1).to(device).data.numpy() for i in range(batch_size)]
    batch_inds = [[np.random.choice(np.arange(0, len(vocab)), p=softmaxed_output[i][l]) for l in range(sentence_len)]
                  for i in range(batch_size)]
    return batch_images, batch_inds

def indices2sentence(indices):
    caption = [vocab.ind2word[c] for c in indices]
    s = " "
    return s.join(caption)

if __name__ == '__main__':
    # How to use found below
    
    use_gpu = torch.cuda.is_available()

    train_image_directory = './data/images/train/'
    train_caption_directory = './data/annotations/captions_train2014.json'
    coco_train = COCO(train_caption_directory)

    with open('TrainImageIds.csv', 'r') as f:
        reader = csv.reader(f)
        train_ids = list(reader)

    train_ids = [int(i) for i in train_ids[0]]

    train_ann_ids = coco_train.getAnnIds(train_ids)

    vocab = Vocabulary()

    embed_size = 500

    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    ])

    train_loader = get_loader(train_image_directory,
                              train_caption_directory,
                              ids=train_ann_ids,
                              vocab=vocab,
                              transform=transform,
                              batch_size=2,
                              shuffle=True,
                              num_workers=10)

    encoder = Encoder(embed_size)
    decoder = DecoderLSTM(500, 256, len(vocab))
    
    if use_gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    for iter, (images, captions, length) in enumerate(train_loader):
        encoder.train()
        decoder.train()
        encoder.zero_grad()
        decoder.zero_grad()

        if use_gpu:
            images = images.cuda()
            captions = captions.cuda()
        else:
            images = images.cpu()
            captions = captions.cpu()

        targets = pack_padded_sequence(captions, length, batch_first=True).data
        # forward
        image_features = encoder(images)
        output_caption = decoder(image_features, captions)

        break
    
    # HOW TO USE FUNCTIONS: output_caption, images from network, trainloader respectively
    imgs, inds = deterministic_generate(output_caption, images, device='cpu')
    plt.imshow(imgs[1])
    print(indices2sentence(inds[1]))

    imgs, inds = stochastic_generate(output_caption, images, device='cpu')
    plt.imshow(imgs[1])
    print(indices2sentence(inds[1]))

