# FastSpeech-Squeezewave-Pytorch
The Implementation of FastSpeech Based on Pytorch.

## Update
### 2019/10/23
1. Fix bugs in alignment;
2. Fix bugs in transformer;
3. Fix bugs in LengthRegulator;
4. Change the way to process audio;
5. Use squeezewave to synthesize.

## Model
<div align="center">
<img src="img/model.png" style="max-width:100%;">
</div>

## Start
### Dependencies
- python 3.6
- CUDA 10.0
- pytorch==1.1.0
- nump>=1.16.2
- scipy>=1.2.1
- librosa>=0.7.2
- inflect>=2.1.0
- matplotlib>=2.2.2

### Prepare Dataset
1. Download and extract [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).
2. Put LJSpeech dataset in `data`.
3. Unzip `alignments.zip` \*
4. Put [pretrained squeezewave model](https://drive.google.com/file/d/1RyVMLY2l8JJGq_dCEAAd8rIRIn_k13UB/view) in the `squeezewave/pretrained_model`;
5. Run `python preprocess.py`.

*\* if you want to calculate alignment, don't unzip alignments.zip and put [Nvidia pretrained Tacotron2 model](https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing) in the `Tacotron2/pretrained_model`*

## Training
Run `python train.py`.

## Test
Run `python synthesis.py "write your TTS Here"`.

## Inference Time
<br />Intel® Core™ i5-6300U CPU<br />
<br />example 1<br />
<br />taskset --cpu-list 1 python3 synthesis.py "Fastspeech with Squeezewave vocoder in pytorch , very fast inference on cpu"<br />
<br />Speech synthesis time: 
1.7220683097839355

<br />soxi out:
<br />Input File     : 'results/Fastspeech with Squeezewave vocoder in pytorch , very fast inference on cpu_112000_squeezewave.wav'
<br />Channels       : 1
<br />Sample Rate    : 22050
<br />Precision      : 16-bit
<br />Duration       : 00:00:05.96 = 131328 samples ~ 446.694 CDDA sectors
<br />File Size      : 263k
<br />Bit Rate       : 353k
<br />Sample Encoding: 16-bit Signed Integer PCM
<br />approx. 6 sec. audio output in 1.72 sec on single cpu

<br />example 2
<br />taskset --cpu-list 0 python3 synthesis.py "How are you"
<br />Speech synthesis time:
0.3431851863861084
<br />soxi out:
<br />Input File     : 'results/How are you _112000_squeezewave.wav'
<br />Channels       : 1
<br />Sample Rate    : 22050
<br />Precision      : 16-bit
<br />Duration       : 00:00:00.85 = 18688 samples ~ 63.5646 CDDA sectors
<br />File Size      : 37.4k
<br />Bit Rate       : 353k
<br />Sample Encoding: 16-bit Signed Integer PCM
<br />0.85 sec. audio output in 0.34 sec on single cpu
## Pretrained Model
- Baidu: [Step:112000](https://pan.baidu.com/s/1by3-8t3A6uihK8K9IFZ7rg) Enter Code: xpk7
- OneDrive: [Step:112000](https://1drv.ms/u/s!AuC2oR4FhoZ29kriYhuodY4-gPsT?e=zUIC8G)

## Notes
- In the paper of FastSpeech, authors use pre-trained Transformer-TTS to provide the target of alignment. I didn't have a well-trained Transformer-TTS model so I use Tacotron2 instead.
- The examples of audio are in `results`.
- The outputs and alignment of Tacotron2 are shown as follows (The sentence for synthesizing is "I want to go to CMU to do research on deep learning."):
<div align="center">
<img src="img/tacotron2_outputs.jpg" style="max-width:100%;">
</div>

- The outputs of FastSpeech and Tacotron2 (Right one is tacotron2) are shown as follows (The sentence for synthesizing is "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition."):
<div align="center">
<img src="img/model_test.jpg" style="max-width:100%;">
</div>

## Reference
- [The Implementation of Tacotron Based on Tensorflow](https://github.com/keithito/tacotron)
- [The Implementation of Transformer Based on Pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- [The Implementation of Transformer-TTS Based on Pytorch](https://github.com/xcmyz/Transformer-TTS)
- [The Implementation of Tacotron2 Based on Pytorch](https://github.com/NVIDIA/tacotron2)
- [The Implementation of Squeezewave Based n Pytorch](https://github.com/tianrengao/SqueezeWave)
- [SqueezeWave: Extremely Lightweight Vocoders for On-device Speech Synthesis](https://arxiv.org/abs/2001.05685)
- [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263)
