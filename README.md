# Fine Grained Visual Attention

This is a **PyTorch** implementation of [
Where to Focus: Deep Attention-based Spatially Recurrent Bilinear Networks for Fine-Grained Visual Recognition](https://arxiv.org/pdf/1709.05769.pdf) by *Lin Wu, Yang Wang*.

## Todo

- [ ] write an efficient version of the retina
- [x] reduce training from 130 epochs to 30 epochs
- [ ] write custom loss and acc plotter

## Abstract

In this paper, we present a novel attention-based model to automatically, selectively and accurately focus on critical object regions with higher importance against appearance variations. Given an image, two different Convolutional Neural Networks (CNNs) are constructed, where the outputs of two CNNs are correlated through bilinear pooling to simultaneously focus on discriminative regions and extract relevant features. To capture spatial distributions among the local regions with visual attention, soft attention based spatial Long-Short Term Memory units (LSTMs) are incorporated to realize spatially recurrent yet visually selective over local input patterns. Two-stream CNN layers, bilinear pooling layer, spatial recurrent layer with location attention are jointly trained via an end-to-end fashion to serve as the part detector and feature extractor, whereby relevant features are localized and extracted attentively.
## Results

## Requirements

- python 3.5+
- pytorch 0.3+
- tensorboard_logger
- tqdm

## Usage

The easiest way to start training your RAM variant is to edit the parameters in `config.py` and run the following command:

```
python main.py
```

To resume training, run the following command:

```
python main.py --resume=True
```

Finally, to test a checkpoint of your model that has achieved the best validation accuracy, run the following command:

```
python main.py --is_train=False
```

## References

