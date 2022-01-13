# IMDN
Lightweight Image Super-Resolution with Information Multi-distillation Network (ACM MM 2019)

[[arXiv]](https://arxiv.org/pdf/1909.11856v1.pdf)
[[Poster]](https://github.com/Zheng222/IMDN/blob/master/images/acmmm19_poster.pdf)
[[ACM DL]](https://dl.acm.org/citation.cfm?id=3351084)

## Testing
Pytorch 1.1
* Runing testing:
```bash
# Set5 x2 IMDN
python test_IMDN.py --test_hr_folder Test_Datasets/Set5/ --test_lr_folder Test_Datasets/Set5_LR/x2/ --output_folder results/Set5/x2 --checkpoint checkpoints/IMDN_x2.pth --upscale_factor 2
# RealSR IMDN_AS
python test_IMDN_AS.py --test_hr_folder Test_Datasets/RealSR/ValidationGT --test_lr_folder Test_Datasets/RealSR/ValidationLR/ --output_folder results/RealSR --checkpoint checkpoints/IMDN_AS.pth

```
* Calculating IMDN_RTC's FLOPs and parameters, input size is 240*360
```bash
python calc_FLOPs.py
```

## Training
* Convert png file to npy file
```bash
python scripts/png2npy.py --pathFrom /path/to/DIV2K/ --pathTo /path/to/DIV2K_decoded/
```
* Run training x2, x3, x4 model
```bash
python train_IMDN.py --root /path/to/DIV2K_decoded/ --scale 2 --pretrained checkpoints/IMDN_x2.pth
python train_IMDN.py --root /path/to/DIV2K_decoded/ --scale 3 --pretrained checkpoints/IMDN_x3.pth
python train_IMDN.py --root /path/to/DIV2K_decoded/ --scale 4 --pretrained checkpoints/IMDN_x4.pth
```


## Citation

If you find IMDN useful in your research, please consider citing:

```
@inproceedings{Hui-IMDN-2019,
  title={Lightweight Image Super-Resolution with Information Multi-distillation Network},
  author={Hui, Zheng and Gao, Xinbo and Yang, Yunchu and Wang, Xiumei},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia (ACM MM)},
  pages={2024--2032},
  year={2019}
}

@inproceedings{AIM19constrainedSR,
  title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},
  author={Kai Zhang and Shuhang Gu and Radu Timofte and others},
  booktitle={The IEEE International Conference on Computer Vision (ICCV) Workshops},
  year={2019}
}

```
