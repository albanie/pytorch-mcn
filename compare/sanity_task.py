"""
Minimal classification example used as a sanity check for the imported
models.

Based on the PyTorch MNIST example here:
https://github.com/pytorch/examples/blob/master/mnist/main.py

Example usage:
ipy sanity_task.py -- --model_name default --batch_size 256 --im_size 32
ipy sanity_task.py -- --model_name vgg_face_dag --batch_size 32 --im_size 224
ipy sanity_task.py -- --model_name vgg16 --batch_size 32 --im_size 224
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
from pathlib import Path
import torch.optim as optim
from torchvision import datasets, transforms, models


class Net(nn.Module):
    """Basic CIFAR classifier"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def replace_classifier(net, model_name, num_classes=10):
    """Replace the final classification layer of a network

    Args:
        net (torch.nn.Module): the network to be modified.
        model_name (str): the name of the model.
        num_classes (int): the number of output classes desired.
    """
    named_classifier = list(net.named_children())[-1]

    msg = "unexpected classifier name for {}".format(model_name)
    if model_name == "vgg_face_dag":
        classifier_name = "fc8"
        is_seq = False
    elif model_name == "vgg16":
        classifier_name = "classifier"
        is_seq = True
    assert named_classifier[0] == classifier_name, msg
    classifier = getattr(net, classifier_name)
    if is_seq:
        classifier = classifier[-1]
    new_classifier = torch.nn.Linear(classifier.in_features, num_classes)
    if is_seq:
        getattr(net, classifier_name)[-1] = new_classifier
    else:
        setattr(net, classifier_name, new_classifier)
    return net


def model_zoo(model_name, model_dir):
    """Fetch model by name"""
    if model_name == "default":
        return Net()

    if model_name == "vgg16":
        net = models.vgg16(pretrained=True)
    else:
        model_def_path = str(Path(model_dir) / (model_name + '.py'))
        weights_path = str(Path(model_dir) / (model_name + '.pth'))
        spec = importlib.util.spec_from_file_location(model_name,
                                                      model_def_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        func = getattr(mod, model_name)
        net = func(weights_path=weights_path)
    net = replace_classifier(net, model_name)
    return net


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            loss = F.cross_entropy(output, target, reduction='sum')
            test_loss += loss.item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    msg = "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n"
    print(msg.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help="batch size")
    parser.add_argument('--im_size', type=int, default=32,
                        help="image input size")
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help="num batches to wait before logging status")
    parser.add_argument(
        '--model_dir',
        type=str,
        default=str(Path.home() / "data/models/pytorch/mcn_imports"),
        help='location of imported pytorch models')
    parser.add_argument('--model_name', type=str, default="default",
                        help='the name of the model to train')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform = transforms.Compose([
        transforms.Resize(args.im_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
    )

    model = model_zoo(model_name=args.model_name, model_dir=args.model_dir)
    model = model.to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum
    )

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
