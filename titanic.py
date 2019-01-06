# -*- coding:utf-8 -*-
import six
import numpy as np
import pandas as pd
import Age_predict as ap
import random
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
# 学习：https://mp.weixin.qq.com/s/izpHCfw8yJ8vBQZX0UkkoA

def getDataSets(binary=False, bins=False, scaled=False, strings=False, raw=True, pca=False, balanced=False):
    """
    Performs all feature engineering tasks including populating missing values, generating binary categorical
    features, scaling, and other transformations. The boolean parameters of this function will allow fine-grained
    control of what types of features to return, so that it can be used by multiple ML algorithms

    Parameters
    ==========
    binary - boolean
        whether or not to include binary features in the data set

    bins - boolean
        whether or not to include binned features in the data set

    scaled - boolean
        whether or not to include scaled features in the data set

    strings - boolean
        whether or not to include features that are strings in the data set

    raw - boolean
        whether or not to include raw features in the data set

    pca - boolean
        whether or not to perform PCA on the data set

    balanced - boolean
        whether or not to perform up sampling on the survived examples to balance the class distributions

    Returns
    =======
    input_df - array-like
        The labeled training data

    submit_df - array-like
        The unlabled test data to predict and submit
    """
    keep_binary = binary
    keep_bins = bins
    keep_scaled = scaled
    keep_raw = raw
    keep_strings = strings

    input_df = pd.read_csv('all/train.csv', header=0)
    test_data = pd.read_csv('all/test.csv', header=0)

    df = pd.concat([input_df, test_data], sort=True)
    df.reset_index(inplace=True)
    # print df
    df.drop('index', axis=1, inplace=True)

    df = df.reindex(input_df.columns, axis=1)
    from variable_process import *
    processCabin()
    processTicket()
    processName()
    processFare()
    processEmbarked()
    processFamily()
    processSex()
    processPClass()
    processAge()
    variable_drop()

    columns_list = list(df.columns.values)
    columns_list.remove('Survived')
    new_col_list = list(['Survived'])
    new_col_list.extend(columns_list)
    df = df.reindex(columns=new_col_list)

    # *********************************************************************************************************
    # Automated feature generation based on basic math on scaled features
    numerics = df.loc[:, ['Age_scaled', 'Fare_scaled', 'Pclass_scaled', 'Parch_scaled', 'SibSp_scaled',
                          'Names_scaled', 'CabinNumber_scaled', 'Age_bin_id_scaled', 'Fare_bin_id_scaled']]
    print "\nFeatures used for automated feature generation:\n", numerics.head(10)

    new_fields_count = 0
    for i in range(0, numerics.columns.size - 1):
        for j in range(0, numerics.columns.size - 1):
            if i <= j:
                name = str(numerics.columns.values[i]) + "*" + str(numerics.columns.values[j])
                df = pd.concat([df, pd.Series(numerics.iloc[:, i] * numerics.iloc[:, j], name=name)], axis=1)
                new_fields_count += 1
            if i < j:
                name = str(numerics.columns.values[i]) + "+" + str(numerics.columns.values[j])
                df = pd.concat([df, pd.Series(numerics.iloc[:, i] + numerics.iloc[:, j], name=name)], axis=1)
                new_fields_count += 1
            if not i == j:
                name = str(numerics.columns.values[i]) + "/" + str(numerics.columns.values[j])
                df = pd.concat([df, pd.Series(numerics.iloc[:, i] / numerics.iloc[:, j], name=name)], axis=1)
                name = str(numerics.columns.values[i]) + "-" + str(numerics.columns.values[j])
                df = pd.concat([df, pd.Series(numerics.iloc[:, i] - numerics.iloc[:, j], name=name)], axis=1)
                new_fields_count += 2

    print "\n", new_fields_count, "new features generated"

    # *********************************************************************************************************
    # Use Spearman correlation to remove highly correlated features

    # calculate the correlation matrix
    df_corr = df.drop(['Survived', 'PassengerId'], axis=1).corr(method='spearman')

    # create a mask to ignore self-
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
    df_corr = mask * df_corr

    drops = []
    # loop through each variable
    for col in df_corr.columns.values:
        # if we've already determined to drop the current variable, continue
        if np.in1d([col], drops):
            continue

        # find all the variables that are highly correlated with the current variable
        # and add them to the drop list
        corr = df_corr[abs(df_corr[col]) > 0.98].index
        # print col, "highly correlated with:", corr
        drops = np.union1d(drops, corr)

    print "\nDropping", drops.shape[0], "highly correlated features...\n"  # , drops
    df.drop(drops, axis=1, inplace=True)

    # *********************************************************************************************************
    # Split the data sets apart again, perform PCA/clustering/class balancing if necessary
    #
    input_df = df[:input_df.shape[0]]
    submit_df = df[input_df.shape[0]:]

    if pca:
        print "reducing and clustering now..."
        input_df, submit_df = reduce_dimention(input_df, submit_df)
    else:
        # drop the empty 'Survived' column for the test set that was created during set concatentation
        submit_df.drop('Survived', axis=1, inplace=1)

    print "\n", input_df.columns.size, "initial features generated...\n"  # , input_df.columns.values

    if balanced:
        # Undersample training examples of passengers who did not survive
        print 'Perished data shape:', input_df[input_df.Survived == 0].shape
        print 'Survived data shape:', input_df[input_df.Survived == 1].shape
        perished_sample = random.sample(input_df[input_df.Survived == 0].index, input_df[input_df.Survived == 1].shape[0])
        input_df = pd.concat([input_df.ix[perished_sample], input_df[input_df.Survived == 1]])
        input_df.sort(inplace=True)
        print 'New even class training shape:', input_df.shape

    return input_df, submit_df


if __name__ == "main":
    """
       titanic 主代码
    """
    input_df,submit_df = getDataSets(bins=True, scaled=True, binary=True)

    features_list = input_df.columns.values[1:]

    X = input_df.values[:, 1::]
    y = input_df.values[:, 0]

    # Set the weights to adjust for uneven class distributions (fewer passengers survived than died)
    survived_weight = .75
    y_weights = np.array([survived_weight if s == 1 else 1 for s in y])

    ##############################################################################################################
    # Reduce initial feature set with estimated feature importance
    #
    print "Rough fitting a RandomForest to determine feature importance..."
    forest = RandomForestClassifier(oob_score=True, n_estimators=10000)
    forest.fit(X, y, sample_weight=y_weights)
    feature_importance = forest.feature_importances_

    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    # print "Feature importances:\n", feature_importance

    fi_threshold = 18

    important_idx = np.where(feature_importance > fi_threshold)[0]
    # print "Indices of most important features:\n", important_idx

    important_features = features_list[important_idx]
    print "\n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance)...\n"  # , \
    # important_features

    sorted_idx = np.argsort(feature_importance[important_idx])[::-1]

    # Plot feature importance
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], align='center')
    plt.yticks(pos, important_features[sorted_idx[::-1]])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.draw()
    plt.show()

    # Remove non-important features from the feature set and submission sets
    X = X[:, important_idx][:, sorted_idx]
    # print "\nSorted (DESC) Useful X:\n", X

    submit_df = submit_df.iloc[:, important_idx].iloc[:, sorted_idx]
    print '\nTraining with', X.shape[1], "features:\n", submit_df.columns.values
    # print input_df.iloc[:,1::].iloc[:,important_idx].iloc[:,sorted_idx].head(10)


