from models.modeling import VisionTransformer, CONFIGS
import torch 
import numpy as np
from argparse import ArgumentParser
from train import valid, setup
from utils.data_utils import get_testloader
import pandas as pd
import torch.distributed as dist
import json
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from datetime import timedelta
import os
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from utils.dist_util import get_world_size
def check_slash(string):
     if string and len(string) > 0 and string[0] == '/':
         return string
     else:
         raise NotADirectoryError(string)
def test(args, model):
    
    annotation_file = args.annotation_file
    img_labels = pd.read_csv(annotation_file, header=None, sep =" ")
    lb = LabelEncoder()
    img_labels['encoded_label'] = lb.fit_transform(img_labels.iloc[:,1])
    # print(img_labels)
    class_dict = pd.Series(img_labels.iloc[:,1].values, index=img_labels.encoded_label).to_dict()
    # print(class_dict)
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
    test_loader = get_testloader(args)
    batchtest = tqdm(test_loader)
    pred=[]
    img_name = []
    with torch.no_grad():
        model.eval()
        
        for step, batch in enumerate(batchtest):
            # batch = tuple(t.to(args.device) for t in batch)
            image, image_name = batch
            image = image.to(args.device,dtype=torch.float)
            # image_name = image_name.to(args.device,dtype=torch.long)
            logits = model(image)
            predicted = torch.argmax(logits,dim=-1)
            for j in range(predicted.size()[0]):
                    pred.append(int(predicted[j]))
                    img_name.append(str(image_name[j]))
    print('img_name: ', img_name)
    convert_pred=[]
    for i in range (0, len(pred)):
        temp_pred = pred[i]
        convert = class_dict[temp_pred]
        convert_pred.append(convert)

    convert_pred = np.array(convert_pred)
    img_name = np.array(img_name)
    pred_df = np.column_stack((img_name, convert_pred))
    # print(pred_df)
    np.savetxt(args.output, pred_df, fmt='%s') 
def setup_test(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089
    elif args.dataset == 'myBirds':
        num_classes = 200
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value)
    model.load_from(np.load(args.pretrained_dir))
    if args.trained_model is not None:
        trained_model = torch.load(args.trained_model)['model']
        model.load_state_dict(trained_model)
    model.to(args.device)

    return args, model
if __name__ == '__main__':
    parser = ArgumentParser()
    # for model
    parser.add_argument('--img_order', type=check_slash, required=True)
    parser.add_argument('--output', type=str,required=True)
    parser.add_argument('--annotation_file', type=check_slash,required=True)
    parser.add_argument('--img_testdir', type=check_slash,required=True)
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', 0),
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017", 'myBirds'], default="CUB_200_2011",
                        help="Which dataset.")
    parser.add_argument("--trained_model", type=str, default=None,
                        help="load pretrained model")
    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="/opt/tiger/minist/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument('--smoothing_value', type=float, default=0.0, help="Label smoothing value\n")
    args = parser.parse_args()
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu=1
    args.device = device
    args.nprocs = torch.cuda.device_count()
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args, model = setup_test(args)
    # print(json.dumps(vars(args), indent=2))
    test(args,model)

