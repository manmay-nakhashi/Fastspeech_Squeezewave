import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from fastspeech import FastSpeech
from text import text_to_sequence
import hparams as hp
import utils
import audio as Audio
import glow
import squeezewave.inference as sq_infer
import time
import sys


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_FastSpeech(num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    model = nn.DataParallel(FastSpeech()).to(device)
    model.load_state_dict(torch.load(os.path.join(
        hp.checkpoint_path, checkpoint_path),map_location=device)['model'])
    model.eval()

    return model


def synthesis(model, text, alpha=1.0):
    text = np.array(text_to_sequence(text, hp.text_cleaners))
    text = np.stack([text])

    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    with torch.no_grad():
        sequence = torch.autograd.Variable(
            torch.from_numpy(text)).long()
        src_pos = torch.autograd.Variable(
            torch.from_numpy(src_pos)).long()

        mel, mel_postnet = model.module.forward(sequence, src_pos, alpha=alpha)
        
        #script for generating torch script
        #traced_script_module = torch.jit.trace(model,(sequence,src_pos))
        #traced_script_module.save("traced_fastspeech_model.pt")

        return mel[0].cpu().transpose(0, 1), \
            mel_postnet[0].cpu().transpose(0, 1), \
            mel.transpose(1, 2), \
            mel_postnet.transpose(1, 2)


if __name__ == "__main__":
    # Test
    num = 112000
    alpha = 1.0
    model = get_FastSpeech(num)
    squeeze_wave = utils.get_squeezewave()
    words = sys.argv[1]
    start = time.time()
    mel, mel_postnet, mel_torch, mel_postnet_torch = synthesis(
        model, words, alpha=alpha)
    fastspeech =time.time()
    if not os.path.exists("results"):
        os.mkdir("results")

    
    sq_infer.inference(mel_postnet_torch, squeeze_wave, os.path.join(
        "results", words + "_" + str(num) + "_squeezewave.wav"))

    end = time.time()
    print("Speech synthesis time: ")
    print(end-start)


