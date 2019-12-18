import argparse


def opts_init():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_train', type=str, default='/home/y182202001/Projects/Attention/data/brats/HGG/')
    # parser.add_argument('--path_train', type=str, default='/home/y182202099/Project/Attention/data/prostate/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_size', type=int, default=0, help='load data number, 0 means total')
    parser.add_argument('--val', type=bool, default=True, help='1/5 dataset to validate')
    parser.add_argument('--data_mode', type=str, default='slice2', help='data load modes: slice, slice2, total')
    parser.add_argument('--nonzero', type=int, default=200, help='select nonzero number greater in dataset slice')

    parser.add_argument('--EPOCH', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--scheduler_step', type=int, default=10, help='lr decay step')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='lr decay gamma')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='focal loss, positive sample weight')
    parser.add_argument('--focal_gamma', type=float, default=2, help='focal loss, seg difficult weight')

    parser.add_argument('--server', type=str, default='http://172.21.228.120', help='visdom server')
    parser.add_argument('--port', type=int, default=8087, help='visdom port')
    parser.add_argument('--env', type=str, default='slice2_PAM_orimg', help='visdom env')

    parser.add_argument('--param_step', type=int, default=5, help='net param save step')
    parser.add_argument('--image_step', type=int, default=5, help='image show in visdom step')
    parser.add_argument('-r', '--remark', type=str, default='', help='test file extra information')

    parser.add_argument('--points', type=int, default=100, help='number of local PAM sampling points')
    parser.add_argument('--cudaid', type=str, default='0', help='gpu number')

    opts = parser.parse_args()
    return opts
