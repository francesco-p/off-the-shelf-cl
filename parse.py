import argparse


def get_opt():

    parser = argparse.ArgumentParser(description='Unsupervised off-the-shelf Continual Learning', add_help=False)
    parser.add_argument('--load', action='store_true', default=False, help='load memory or not')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='select the dataset')
    parser.add_argument('--model', type=str, default='resnet18', help='select the models')
    parser.add_argument('--seed', type=int, default=0, help='select the seed')
    parser.add_argument('--data_path', type=str, default='~/data/', help='select the datapath')
    parser.add_argument('--batch_size', type=int, default=128, help='select the batch_size')
    parser.add_argument('--increment', type=int, default=10, help='must be divisible by the total number of classes of the dataset chosen')
    parser.add_argument('--cuda', action='store_true', default=True, help='use GPU or not')
    parser.add_argument('--gpu', type=int, default=0, help='which GPU do you want to use')

    opt = parser.parse_args()

    opt.gpu = f'cuda:{opt.gpu}'
    return opt
