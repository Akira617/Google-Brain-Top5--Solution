from lib.include import *
from lib.utility.draw import *
from lib.utility.file import *

from lib.include_torch import *
from lib.net.rate import *
from lib.net.layer_np import *
import gc

if __name__ == '__main__':
    from run_train_fold1 import run_train

    for final_seed in range(0,20,1):
        #---------------------------------------------------------------------------------
        COMMON_STRING ='@%s:  \n' % os.path.basename(__file__)
        if 1:
            seed = final_seed
            seed_py(seed)
            seed_torch(seed)

            torch.backends.cudnn.benchmark     = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
            torch.backends.cudnn.enabled       = True
            torch.backends.cudnn.deterministic = True

            COMMON_STRING += '\tpytorch\n'
            COMMON_STRING += '\t\tseed = %d\n'%seed
            COMMON_STRING += '\t\ttorch.__version__              = %s\n'%torch.__version__
            COMMON_STRING += '\t\ttorch.version.cuda             = %s\n'%torch.version.cuda
            COMMON_STRING += '\t\ttorch.backends.cudnn.version() = %s\n'%torch.backends.cudnn.version()
            try:
                COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = %s\n'%os.environ['CUDA_VISIBLE_DEVICES']
                NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
            except Exception:
                COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = None\n'
                NUM_CUDA_DEVICES = 1

            COMMON_STRING += '\t\ttorch.cuda.device_count()      = %d\n'%torch.cuda.device_count()
            COMMON_STRING += '\t\ttorch.cuda.get_device_properties() = %s\n' % str(torch.cuda.get_device_properties(0))[21:]

        COMMON_STRING += '\n'


        run_train(final_seed, COMMON_STRING)

        torch.cuda.empty_cache()
        gc.collect()
