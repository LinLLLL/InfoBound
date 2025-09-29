import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

results = {}
methodname = "scone" #  "mm_energy"

for filename in sorted(os.listdir(os.path.join(current_dir, "results/cifar10/svhn/", methodname))):

    # filename = 'scone_0.5_0.1_8.0_8.0SEED1'

    print(filename)

    # 拼接pickle文件的路径
    file_path = os.path.join(current_dir, "results/cifar10/svhn/", methodname, filename)
    key = "{}_{}".format(filename.split("_")[-3], filename.split("_")[-2].strip(".pkl"))
    results[key] = []

    # 打开pickle文件并读取数据
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    # index = np.argmax(data["test_accuracy_cor"][60:]) + 60
    # index = np.argmax(data["test_accuracy_cor"])
    index = 59

    OOD_scores_P_in_test = data['OOD_scores_P_in_test'][index]#[-1]
    OOD_scores_P_cor_test = data['OOD_scores_P_cor_test'][index]#[-1]
    OOD_scores_P_out_test = data['OOD_scores_P_out_test'][index]#[-1]


    test_accuracy_cor = data['test_accuracy_cor'][index]#[-1]
    test_accuracy_id = data['test_accuracy_id'][index]#[-1]
    # test_accuracy_id_clean = data['test_accuracy_id_clean'][-1]
    # test_accuracy_cor_clean = data['test_accuracy_cor_clean'][-1]
    fpr95_test = data['fpr95_test'][index]#[-1]
    auroc_test = data['auroc_test'][index]#[-1]

    results[key] = [test_accuracy_cor, test_accuracy_id, fpr95_test, auroc_test]
    print(results[key])

    # ================================================================
    # ===========================================#
    #
    #          画图
    #
    # ===========================================#
    # style set 这里只是一些简单的style设置
    sns.set_palette('deep', desat=.6)
    c1, c2, c3 = sns.color_palette('Set1', 3)

    dist1 = OOD_scores_P_in_test
    dist2 = OOD_scores_P_cor_test
    dist3 = OOD_scores_P_out_test

    np.save(os.path.join("Energy_IN60_Seed1_scone.npy"), dist1)
    np.save(os.path.join("Energy_Cor60_Seed1_scone.npy"), dist2)
    np.save(os.path.join("Energy_Sout60_Seed1_scone.npy"), dist3)

    # np.save(os.path.join("MSP_IN000_Seed1.npy"), dist1)
    # np.save(os.path.join("MSP_Cor000_Seed1.npy"), dist2)
    # np.save(os.path.join("MSP_Sout000_Seed1.npy"), dist3)

    # Plot the KDE plots
    f, ax1 = plt.subplots(1, 1, sharex=True, figsize=(5, 4))
    sns.kdeplot(dist1, fill=True, color="lightblue", ax=ax1)
    sns.kdeplot(dist2, fill=True, color="lightgreen", ax=ax1)
    sns.kdeplot(dist3, fill=True, color="grey", ax=ax1)

    # Label the plot
    ax1.set_xlabel('Energy Score', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.tick_params(labelsize=12)
    # ax1.set_xlim(xmin=-20)
    # ax1.set_ylim(ymax=5)

    # 图例
    plt.legend(['ID data', 'Covariate OoD', 'Semantic OoD'], loc='best', fontsize=12)
    plt.savefig(current_dir + '/figs/' + filename + '_test_60.pdf', bbox_inches='tight', dpi=500)
    # ================================================================


print(results.keys())

# output_file = "ours_seed2.xlsx"
# df = pd.DataFrame(results, columns=list(results.keys()))
# df.to_excel(output_file, columns=list(results.keys()))







