# Homework 1 - Bird Classification

## Introduction
The assignment was to classify 200 bird species, and the challenge is we only have 3000 training data while testing on 3033 images, which prone to the overfitting problem. Since the label for testing dataset was not provided, I used stratified shuffle split on the training dataset with 80:20 percentage for train and validation purposes respectively.
On the other hand, on the bird cases the fine-grained nature make they have a quite similar visual. The fine grained image classification technique is usually used to tackle this issue. For bird classification, I tried one of fine grained image classification technique called TransFG: A Transformer Architecture for Fine-grained Recognition. The model was developed by inspired with visual transformer by integrate all raw attention weights of the
transformer into an attention map. This method was successfully distinguish 200 species with accuracy
0.87207. \
\
**Get the report in this [drive](https://reurl.cc/aNR1VD).**
## Dependencies
  - Python 3.8.8
  - PyTorch 1.9.0
  - ml_collections

## Usage
1. Download Google Pre-trained ViT Models \
   The model can be downloaded in this [link](https://console.cloud.google.com/storage/browser/vit_models).

2. Install required packages
      ```
      pip3 install -r requirements.txt
      ```
3. Train \
   Run: 
      ```
      bash train.sh
      ```
   by change some parameter
      ```
      python -m torch.distributed.launch --nproc_per_node=1 inference.py \
      --img_order 'data_file/testing_img_order.txt' \
      --output 'answer-transFGv8_linear_Vitl32pre-20000e_batch6.txt' \
      --annotation_file 'data_file/training_labels.txt' \
      --img_testdir '/path_to_testing_images' \
      --eval_batch_size 6 \
      --dataset 'myBirds' \
      --trained_model '/path/to_trainedmodel' \
      --local_rank 0 \
      --pretrained_dir "vitpretrainedmodel_path/imagenet21k+imagenet2012_ViT-L_32.npz" \
      --model_type "ViT-L_32"
      ```
4. Test 
   - Get my best trained model in this [link](https://reurl.cc/q1oZbN)
   - Run: 
      ```
      bash inference.sh
      ```
      by change some parameter
      ```
      python -m torch.distributed.launch --nproc_per_node=1 inference.py \
      --img_order 'data_file/testing_img_order.txt' \
      --output 'data_file/answer-transFGv8_linear_Vitl32pre-20000e_batch6.txt' \
      --annotation_file 'data_file/training_labels.txt' \
      --img_testdir '/path_to_testing_images' \
      --eval_batch_size 6 \
      --dataset 'myBirds' \
      --trained_model '/path/to_trainedmodel' \
      --local_rank 0 \
      --pretrained_dir "vitpretrainedmodel_path/imagenet21k+imagenet2012_ViT-L_32.npz" \
      --model_type "ViT-L_32"
      ```
  
## References
1. TransFG: A Transformer Architecture for Fine-grained Recognition [[github]](https://github.com/TACJu/TransFG)
2. Vision Transformer and MLP-Mixer Architectures [[github]](https://github.com/google-research/vision_transformer)
