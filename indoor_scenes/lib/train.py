import torch
from torchvision import transforms
from tqdm import tqdm
from ..config import *
from ..data.datasets import MIT67Dataset
from ..network import get_network

from IPython import embed


def train(args):
    """
    Datasets and transforms
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
    train_dataset = MIT67Dataset('/home/tb5zhh/MIT67',
                                 transforms=transform,
                                 split="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.train,
        num_workers=args.num_worker,
    )
    validate_dataset = MIT67Dataset('/home/tb5zhh/MIT67',
                                    transforms=transform,
                                    split="eval")
    validate_dataloader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=args.batch_size,
        shuffle=args.train,
        num_workers=args.num_worker,
    )
    """
    Model and criterion
    """
    network = get_network(args.arch).cuda()

    criterion = torch.nn.CrossEntropyLoss()
    """
    Optimizer and scheduler
    """
    optim = torch.optim.Adam(network.parameters(), lr=args.lr)
    # sched = torch.optim.lr_scheduler.StepLR(optim, 100, 0.1)
    """
    Training starts
    """
    loss = torch.tensor(0, )
    for epoch_idx in range(args.epochs):
        network.train()
        for step_idx, (img, label, _) in enumerate(train_dataloader):
            img = img.cuda()
            out = network(img)
            # embed()
            loss = criterion(out, label.cuda())

            optim.zero_grad()
            loss.backward()
            optim.step()
            # sched.step()
            print(f"epoch: {epoch_idx:03d}, step: {step_idx:03d}: loss {loss.item():.4f}")
        torch.save(
            {
                "model_state_dict": network.state_dict(),
                "optimizer": optim.state_dict(),
                # "scheduler": sched.state_dict(),
                "epoch": epoch_idx,
            }, f'epoch#{epoch_idx}.pth')
        """
        Validation
        """
        network.eval()
        correct = 0
        total = 0
        TP = [0 for _ in range(20)]
        P = [0 for _ in range(20)]
        PP = [0 for _ in range(20)]
        for step_idx, (img, label, _) in enumerate(validate_dataloader):
            img = img.cuda()
            out = network(img)
            predicted = torch.argmax(out, dim=1)
            for i in range(20):
                TP[i] += torch.bitwise_and(predicted.cpu() == i, label == i).sum()
                P[i] += (label == i).sum()
                PP[i] += (predicted.cpu() == i).sum()
            total += len(predicted)
            correct += (predicted.cpu() == label).sum().item()
        for i in range(20):
            print(f'Precision cls#{i:2d}: {TP[i] / PP[i] * 100:3.2f}, Recall cls#{i:2d}: {TP[i] / P[i] * 100:3.2f}')
        print(f"Validate acc: {correct / total * 100:3.2f}")


if __name__ == '__main__':
    args = parse_args()
    args.train = True
    train(args)