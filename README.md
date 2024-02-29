# TensorFlow Lite Python audio classification with Raspberry Pi.

This project uses TensorFlow Lite with Python on
a Raspberry Pi to perform audio classification using audio from wav file.

## Prerequisites
First, [set up your Raspberry Pi](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up).

Then, use the script to install a couple Python packages, and download the TFLite model:

```
cd Yamnet_tflite

# Run this script to install the required dependencies and download the TFLite models.
sh setup.sh
```

## Run the example
```
python3 classify.py --wavFile test_data/meow_16k.wav
```
Example output:
```
Inference result:
  Category: Cat, index: 76, score: 0.80078125
  Category: Animal, index: 67, score: 0.66796875
  Category: Domestic animals, pets, index: 68, score: 0.66796875
  Category: Meow, index: 78, score: 0.5
  Category: Caterwaul, index: 80, score: 0.33203125
```

## About the Model
### Input: Audio Features
YAMNet was trained with audio features computed as follows:

* All audio is resampled to 16 kHz mono.
* A spectrogram is computed using magnitudes of the Short-Time Fourier Transform
  with a window size of 25 ms, a window hop of 10 ms, and a periodic Hann
  window.
* A mel spectrogram is computed by mapping the spectrogram to 64 mel bins
  covering the range 125-7500 Hz.
* A stabilized log mel spectrogram is computed by applying
  log(mel-spectrum + 0.001) where the offset is used to avoid taking a logarithm
  of zero.
* These features are then framed into 50%-overlapping examples of 0.96 seconds,
  where each example covers 64 mel bands and 96 frames of 10 ms each.

These 96x64 patches are then fed into the Mobilenet_v1 model to yield a 3x2
array of activations for 1024 kernels at the top of the convolution.  These are
averaged to give a 1024-dimension embedding, then put through a single logistic
layer to get the 521 per-class output scores corresponding to the 960 ms input
waveform segment.  (Because of the window framing, you need at least 975 ms of
input waveform to get the first frame of output scores.)

### Class vocabulary

The file `yamnet_class_map.csv` describes the audio event classes associated
with each of the 521 outputs of the network.  Its format is:

```text
index,mid,display_name
```

where `index` is the model output index (0..520), `mid` is the machine
identifier for that class (e.g. `/m/09x0r`), and display_name is a
human-readable description of the class (e.g. `Speech`).

The original Audioset data release had 527 classes.  This model drops six of
them on the recommendation of our Fairness reviewers to avoid potentially
offensive mislabelings. The gendered versions (Male/Female) of
Speech and Singing, Battle cry and Funny music are dropped.

### Performance

On the 20,366-segment AudioSet eval set, over the 521 included classes, the
balanced average d-prime is 2.318, balanced mAP is 0.306, and the balanced
average lwlrap is 0.393.

The classifier has about 3.7M weights and performs
69.2M multiplies for each 960ms input frame.

## References
The code in this project is adapted from [Tensorflow Examples](https://github.com/tensorflow/examples/tree/fff4bcda7201645a1efaea4534403daf5fc03d42/lite/examples/audio_classification/raspberry_pi).

The information about the YAMNet model is from [antonyharfield/tflite-models-audioset-yamnet](https://github.com/antonyharfield/tflite-models-audioset-yamnet/tree/master).

[TensorAudio.create_from_wav_file](https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/audio/core/tensor_audio.py#L45) source code.