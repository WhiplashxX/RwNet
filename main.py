from models.rwnet import *
from args import Helper
from utils.train_helper import *
from utils.eval_helper import *
from utils.data_helper import *
import os
import logging
from sklearn.preprocessing import StandardScaler


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    helper = Helper()
    args = helper.config
    args.device = device
    args.args_to_dict = helper.args_to_dict
    setup_seed(args.seed)

    logger = None
    if args.log:
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        handler = logging.FileHandler("{}/log_{}.txt".format(args.log_dir, args.local_time))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info(str(args.args_to_dict))

    # args = helper.Namespace()  # 创建一个 argparse 命名空间对象
    args.data_dir = 'D:\\MINET\\data\\'  # 设置包含数据的目录路径

    try:
        train_data = load_train(args)
        eval_data = load_eval(args)
        test_data = load_test(args)
        print('load train and test data succesfully')
    except Exception as e:
        print('error in load data : {}'.format(e))
        exit()

    if args.scale:
        args.scaler = StandardScaler().fit(train_data.y.reshape(-1, 1))

    model = MINET(args)
    model.to(device)

    for epoch, model, loss in train(model, train_data, args):
        print("epoch:", epoch, "loss=", loss)
        if epoch % args.verbose == 0:
            _, mse, pepe, _ = eval(model, args)
            print('eval_mse {:.5f}'.format(mse))
            print('eval_pepe {:.5f}'.format(pepe))

            if logger:
                logger.info('epoch: {}, eval_mse {:.5f}'.format(epoch, mse))


if __name__ == "__main__":
    main()
