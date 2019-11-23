import argparse


def opts():
    parser = argparse.ArgumentParser(description = "Asian Elephant Facial Recognition Model")
    parser.add_argument('--batch_size', type=int, default = 1, metavar='N', help = "input batch size for training (default : 1)")
    parser.add_argument("--epochs",type=int, default = 2000, metavar = 'N', help="number of epochs to train the model (default: 2000)") 
    parser.add_argument("--use_cuda",action='store_true', default=True, help="enable CUDA training (default: True)")
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--device_id', type=int, default=0, help='gpu device id (default: 1)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate for the adam optimizer')
    parser.add_argument('--data_path', default='None',
                        help='data directory for train, val, and test. the root dir is in data/')
    parser.add_argument('--output_path', default='None',
                        help='directory to save learned representation')
    parser.add_argument('--model_path', default='None',
                        help='directory to save models and losses. root dir is in models/')
    parser.add_argument('--epoch_save_interval', type=int, default=5,
                        help='model that would be saved after every given interval e.g.,  250')
    parser.add_argument('--model_name', default='None',
                        help='trained model name e.g., used during evaluation stage')

    parser.add_argument('--use_sampler', type=str2bool, nargs='?', const=True,
                        help='boolean variable indicating if we are using sampler for training')

    parser.add_argument('--csv_classes',default='data/classes.csv',help="csv file containing classes")

    args = parser.parse_args()
    
    return args

def test_opts():
    parser = argparse.ArgumentParser(description = "Asian Elephant Facial Recognition Model")
    parser.add_argument('--batch_size', type=int, default = 1, metavar='N', help = "input batch size for training (default : 1)")
    parser.add_argument("--use_cuda",action='store_true', default=True, help="enable CUDA training (default: True)")
    parser.add_argument('--data_path', default='None',
                        help='data directory for train, val, and test. the root dir is in data/')
    parser.add_argument('--model_path', default='None',
                        help='path to the best model')

    parser.add_argument('--output_path', default='None',
                        help='directory to save learned representation')

    parser.add_argument('--csv_classes',default='data/classes.csv',help="csv file containing classes")

    args = parser.parse_args()
    
    return args



def str2bool(s):
    if s.lower() in ('yes','true','t','y',1):
        return True
    elif s.lower() in ('no','false','f','n',0):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")
