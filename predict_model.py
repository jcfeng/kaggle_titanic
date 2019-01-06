# -*- coding:utf-8 -*-
import numpy
import matplotlib.pyplot as plt
def random_forest(df):
    # 获得变量名
    features_list = df.columns.values[1:]
    from sklearn.ensemble import RandomForestClassifier

    x = df[:,1:]
    y = df[:,0]

    forest = RandomForestClassifier(oob_score=True,n_estimators=1000)
    forest.fit(x,y)
    feature_importance = forest.feature_importances_

    # 调整特征重要性的数值范围
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    fi_threshold = 15

    # 取特征重要性靠前的变量的下标
    important_idx = np.where(feature_importance > fi_threshold)[0]

    important_features=features_list[important_idx]

    # 排序
    sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
    print "nFeatures sorted by importance (DESC):n", important_features[sorted_idx]
    # 画图
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], align='center')
    plt.yticks(pos, important_features[sorted_idx[::-1]])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.draw()
    plt.show()

    # 只取靠前的变量
    x = x[:, important_idx][:, sorted_idx]


if __name__ == '__main__':
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(np.arange(0, 30, 2).reshape(5, 3), columns=list('abc'))
    print df
    print "----------------------"
    print df.columns.values

    # score=np.array([0.45,0.3,0.56,0.17,0.19,0.7,0.23,0.58])
    # big_score_idx = np.where(score > 0.2)
    # big_score = score[big_score_idx]
    # print "bid_score_idx",big_score_idx
    # sorted_score_idx = np.argsort(score[big_score_idx])
    # print "sorted_score_idx",sorted_score_idx
    # print "sorted_score_idx", sorted_score_idx[::-1]
    # sorted_score = big_score[sorted_score_idx]
    # print "sorted_score",sorted_score
