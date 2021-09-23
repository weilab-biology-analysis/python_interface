import matplotlib.pyplot as plt
import seaborn as sns

colors = ["#8D5CDC", "#EA52BC", "#FF6691", "#FF946B","#FFC859", "#F9F871"]

def draw_ROC_PRC_curve(roc_datas, prc_datas, name, config):
    # roc_data = [FPR, TPR, AUC]
    # prc_data = [recall, precision, AP]

    sns.set(style="darkgrid")
    plt.figure(figsize=(16, 8))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    lw = 2

    plt.subplot(1, 2, 1)
    for index, roc_data in enumerate(roc_datas):
            plt.plot(roc_data[0], roc_data[1], color=colors[index],lw=lw, label=name[index] + ' (AUC = %0.2f)' % roc_data[2])  ### 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict={'weight': 'normal', 'size': 20})
    plt.ylabel('True Positive Rate', fontdict={'weight': 'normal', 'size': 20})
    plt.title('receiver operating characteristic curve', fontdict={'weight': 'normal', 'size': 23})
    plt.legend(loc="lower right", prop={'weight': 'normal', 'size': 16})

    plt.subplot(1, 2, 2)
    # plt.step(self.prc_data[0], self.prc_data[1], color='b', alpha=0.2,where='post')
    # plt.fill_between(prc_data[0], prc_data[1], step='post', alpha=0.2,color='b')
    for index, prc_data in enumerate(prc_datas):
        plt.plot(prc_data[0], prc_data[1], color=colors[index],
             lw=lw, label=name[index] + ' (AP = %0.2f)' % prc_data[2])  ### 假正率为横坐标，真正率为纵坐标做曲线

    plt.xlabel('Recall', fontdict={'weight': 'normal', 'size': 20})
    plt.ylabel('Precision', fontdict={'weight': 'normal', 'size': 20})
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision Recall curve', fontdict={'weight': 'normal', 'size': 23})
    plt.legend(loc="lower left", prop={'weight': 'normal', 'size': 16})

    plt.savefig(
        '{}/{}/{}.{}'.format("../result" , config.learn_name , 'ROC_PRC', config.save_figure_type))
    plt.show()