import warnings
import torch

class DefaultConfig(object):
    env = 'Reforce'
    model = 'ResNet'
    mode = None

    train_data_root = './data/train/'
    test_data_root = './data/test1/'
    load_model_path = None

    batch_size = 32
    use_gpu=True
    num_workers = 4
    print_freq = 20

    debug_file = '/tmp/debug'
    result_file = 'result.csv'

    max_epoch = 5
    lr = 0.001
    lr_decay = 0.9
    weight_decay = 1e-6 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def parse(self, kwargs):
        """
        update config parameters through kwargs(dict)
        """
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn(f"warning: opt has not attribute {k}")
            setattr(self, k, v)
        print("user config:")
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k ,getattr(self, k))


opt = DefaultConfig()