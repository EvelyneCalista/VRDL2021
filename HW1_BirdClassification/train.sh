# CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 train.py \
--name 'mybird_trainv13_linear_changetransform_vitl16_20000e_batch6_reproduce' \
--dataset 'myBirds' \
--data_root 'pathto/training_images' \
--model_type "ViT-L_16" \
--pretrained_dir "vitpretrainedmodel_path/imagenet21k+imagenet2012_ViT-L_16.npz" \
--output_dir "output" \
--train_batch_size 2 \
--eval_batch_size 2 \
--local_rank 0 \
--decay_type "linear" \
--fp16 \
--img_dir 'trainimage_path/training_images' \
--img_order 'testing_img_order.txt' \
--img_labels 'train.txt'  \
--img_val 'val.txt' 