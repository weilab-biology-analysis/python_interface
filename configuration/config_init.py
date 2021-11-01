import argparse


def get_config():
    parse = argparse.ArgumentParser(description='common train config')

    # 项目配置参数
    parse.add_argument('-learn-name', type=str, default='train01', help='本次训练名称')
    parse.add_argument('-setting', type=str, default='sysargs', help='本次训练名称')
    parse.add_argument('-path-save', type=str, default='../result/', help='保存字典的位置')
    parse.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
    parse.add_argument('-threshold', type=float, default=0.60, help='准确率阈值')
    parse.add_argument('-cuda', type=bool, default=True)
    # parse.add_argument('-cuda', type=bool, default=False)
    parse.add_argument('-device', type=int, default=3)
    parse.add_argument('-seed', type=int, default=43)
    parse.add_argument('-num_workers', type=int, default=4)
    parse.add_argument('-num_class', type=int, default=2)
    parse.add_argument('-kmer', type=int, default=3)

    # 路径参数
    # parse.add_argument('-path-data', type=str, default='../data/test.txt', help='训练数据的位置')
    parse.add_argument('-path-data', type=str, default='/home/wrh/weilab_server/python_interface/data/test.txt', help='训练数据的位置')
    parse.add_argument('-path-params', type=str, default=None, help='模型参数路径')
    # parse.add_argument('-path-params', type=str, default='../result/SL_train_00/BERT, MCC[0.60].pt', help='模型参数路径')
    parse.add_argument('-model-save-name', type=str, default='BERT', help='保存模型的命名')
    # parse.add_argument('-save-figure-type', type=str, default='jpeg', help='保存图片的文件类型')
    parse.add_argument('-save-figure-type', type=str, default='png', help='保存图片的文件类型')

    # 训练参数
    parse.add_argument('-mode', type=str, default='train-test', help='训练模式')
    parse.add_argument('-type', type=str, default='DNA', help='分子名称')
    parse.add_argument('-model', type=str, default='3mer_DNAbert', help='预训练模型名称')

    # parse.add_argument('-if-MIM', type=bool, default=True)
    parse.add_argument('-if-MIM', type=bool, default=False)
    # parse.add_argument('-if-transductive', type=bool, default=True, help='inductive or transductive')
    parse.add_argument('-if-transductive', type=bool, default=False, help='inductive or transductive')

    parse.add_argument('-interval-log', type=int, default=20, help='经过多少batch记录一次训练状态')
    parse.add_argument('-interval-valid', type=int, default=1, help='经过多少epoch对交叉验证集进行测试')
    parse.add_argument('-interval-test', type=int, default=1, help='经过多少epoch对测试集进行测试')

    parse.add_argument('-epoch', type=int, default=2, help='迭代次数')
    parse.add_argument('-optimizer', type=str, default='Adam', help='优化器名称')
    # parse.add_argument('-optimizer', type=str, default='AdamW', help='优化器名称')
    parse.add_argument('-loss-func', type=str, default='CE', help='损失函数名称, CE/FL')
    parse.add_argument('-batch-size', type=int, default=8)

    parse.add_argument('-lr', type=float, default=0.0001)
    parse.add_argument('-reg', type=float, default=0.0025, help='正则化lambda')
    parse.add_argument('-gamma', type=float, default=2, help='gamma in Focal Loss')
    parse.add_argument('-alpha', type=float, default=0.5, help='alpha in Focal Loss')

    # # 模型参数配置
    parse.add_argument('-max-len', type=int, default=207, help='max length of input sequences')
    # parse.add_argument('-num-layer', type=int, default=3, help='number of encoder blocks')
    # parse.add_argument('-num-head', type=int, default=8, help='number of head in multi-head attention')
    parse.add_argument('-dim-embedding', type=int, default=32, help='residue embedding dimension')
    # parse.add_argument('-dim-feedforward', type=int, default=32, help='hidden layer dimension in feedforward layer')
    # parse.add_argument('-dim-k', type=int, default=32, help='embedding dimension of vector k or q')
    # parse.add_argument('-dim-v', type=int, default=32, help='embedding dimension of vector v')

    config = parse.parse_args()
    return config
