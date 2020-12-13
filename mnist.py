
import argparse
import numpy as np
import pico.functional as F
from pico.base import Tensor, tracer
import pico.module as nn
import pico.utils as utils
import pico.optimizer as optim
import matplotlib.pyplot as plt
import pico.base as Pico


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential([
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ])
        self.init_module()

    def forward(self, x):
        out = self.layers(x)
        return out


def train(args, model, train_loader, optimizer, epoch, logger):
    model.train()
    # print(len(tracer.tensors))
    for batch_idx, (data, target) in enumerate(train_loader()):
        optimizer.zero_grad()
        output = model(data)
        loss = F.CrossEntropyLoss(output, target)
        loss.backward()
        logger.append(loss.data)
        utils.clip_grad(model)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * data.size()[0], len(train_loader),
                100. * batch_idx / len(train_loader), loss.data))

        # print(len(tracer.tensors))
        # return


def test(model, test_loader):
    model.eval()
    test_loss = []
    correct = 0

    with Pico.no_grad():
        for data, target in test_loader():
            output = model(data)
            test_loss.append(F.CrossEntropyLoss(output, target).data)
            pred = np.argmax(output.data, axis=1)
            correct += (pred == target.data).sum()
            # print(len(tracer.tensors))
            # return

    test_loss = np.mean(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader),
        100. * correct / len(test_loader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pico MNIST Example')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for testing (default: 500)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help='model checkpoint path.')

    args = parser.parse_args()

    train_loader = utils.DataLoader(
        'data/MNIST/train', utils.transform(0.1307, 0.3081), args.batch_size, True)
    test_loader = utils.DataLoader(
        'data/MNIST/test', utils.transform(0.1307, 0.3081), args.test_batch_size, False)

    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.resume != '':
        ckpt = utils.load(args.resume)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    logger = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch, logger)
        test(model, test_loader)
        utils.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
            "./model.pt".format(epoch)
        )

    utils.save({'logger': logger}, 'log.pt')

    plt.semilogy(logger)
    plt.grid(ls='--')
    plt.savefig('loss.png', dpi=300)
