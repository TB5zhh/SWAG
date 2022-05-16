import os

import torch
from IPython import embed
from PIL import Image
from torchvision import transforms

from ..config import USED_ROOM_TYPES, parse_args
from ..network import get_network


def inference(file_list, ckpt_path, args):
    """
    Transforms
    """
    transform = transforms.Compose([
        transforms.Resize(
            args.resolution,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    """
    Model and checkpoint loading
    """
    network = get_network(args.arch).cuda()

    ckpt = torch.load(ckpt_path)
    network.load_state_dict(ckpt['model_state_dict'])
    """
    Inference starts
    """
    for file_path in file_list:
        im = Image.open(file_path).convert("RGB")
        im = transform(im).cuda().unsqueeze(dim=0)
        out = network(im)
        _, topk = torch.topk(out, 5, dim=1)
        topk = topk.squeeze()
        labels = ", ".join([USED_ROOM_TYPES[i] for i in topk])
        # print(f"{file_path}: {labels}")
        yield file_path, labels


if __name__ == '__main__':
    args, _ = parse_args()
    data_root = '/home/tb5zhh/1UnKg1rAb8A'
    data_list = [f"{data_root}/{i}" for i in  sorted(os.listdir(data_root))]
    inference(data_list, '/home/tb5zhh/SWAG/epoch#99.pth', args)
