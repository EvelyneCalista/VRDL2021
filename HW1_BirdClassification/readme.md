# Homework 1 - Bird Classification

## Introduction
The assignment was to classify 200 bird species, and the challenge is we only have 3000 training data while testing on 3033 images, which prone to the overfitting problem. Since the label for testing dataset was not provided, I used stratified shuffle split on the training dataset with 80:20 percentage for train and validation purposes respectively.
On the other hand, on the bird cases the fine-grained nature make they have a quite similar visual.The fine grained image classification technique is usually used to tackle this issue. For bird classification, I tried one of fine grained image classification technique called TransFG: A Transformer Architecture for Fine-grained Recognition. The model was developed by inspired with visual transformer by integrate all raw attention weights of the
transformer into an attention map. This method was successfully distinguish 200 species with accuracy
0.87207.
## Dependencies
  - Python 3.8.8
  - PyTorch 1.9.0
  - ml_collections

## Usage
1. Download Google Pre-trained ViT Models \
The model can be downloaded in this [link](https://console.cloud.google.com/storage/browser/vit_models).
The specific model used in this experiment:
   - vit_models/imagenet21k+imagenet2012 :
   - vit_models/imagenet21k : Vit-

2. Get the data
4. Install required packages

4. Train \
   Run: 
      ```
      bash train.sh
      ```
6. Test 
   - Get my best trained model in this [link](https://reurl.cc/q1oZbN)
   - Run: 
      ```
      bash inference.sh
      ```
https://reurl.cc/aNR1VD   
## References
1. TransFG: A Transformer Architecture for Fine-grained Recognition [[github]](https://github.com/TACJu/TransFG)
2. Vision Transformer and MLP-Mixer Architectures [[github]](https://github.com/google-research/vision_transformer)
