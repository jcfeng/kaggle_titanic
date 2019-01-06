# -*- coding:utf-8 -*-
import pandas as pd
import re
import numpy  as np
from sklearn import preprocessing

keep_binary=True
keep_scaled=True
keep_bins = True
keep_strings = True
#对Tickit变量进行处理
def processTicket():
    global df

    # 将“Ticket”的前缀抽出并处理
    df['TicketPrefix'] = df['Ticket'].map(lambda x: getTicketPrefix(x.upper()))
    df['TicketPrefix'] = df['TicketPrefix'].map(lambda x: re.sub('[.?/?]', '', x))
    df['TicketPrefix'] = df['TicketPrefix'].map(lambda x: re.sub('STON', 'SOTON', x))

    # Dummy化
    prefixes = pd.get_dummies(df['TicketPrefix']).rename(columns=lambda x: 'TicketPrefix_' + str(x))
    df = pd.concat([df, prefixes], axis=1)

    # factorizing
    df['TicketPrefixId'] = pd.factorize(df['TicketPrefix'])[0]

    # 将数字抽出
    df['TicketNumber'] = df['Ticket'].map(lambda x: getTicketNumber(x))

    # 衍生一个“数字位数 ”变量
    df['TicketNumberDigits'] = df['TicketNumber'].map(lambda x: len(x)).astype(np.int)

    # 取开头的数字衍生一个变量
    df['TicketNumberStart'] = df['TicketNumber'].map(lambda x: x[0:1]).astype(np.int)

    # 去掉抽出的前缀列
    df.drop(['TicketPrefix', 'TicketNumber'], axis=1, inplace=True)


def getTicketPrefix(ticket):
    match = re.compile("([a-zA-Z./]+)").search(ticket)
    if match:
        return match.group()
    else:
        return 'U'


def getTicketNumber(ticket):
    match = re.compile("([d]+$)").search(ticket)
    if match:
        return match.group()
    else:
        return '0'



def processCabin(df):
    # 缺失值填充
    # print df['Cabin']
    df['Cabin'][df.Cabin.isnull()] = 'U0'
    df['CabinLetter'] = df['Cabin'].map(lambda x:getFirstLetter(x))
    df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]

    if keep_binary:
        deck = pd.get_dummies(df['CabinLetter']).rename(columns=lambda x: 'CabinLetter_' + str(x))
        df = pd.concat([df,deck],axis=1)
    df['CabinNum']=df['CabinNum'].map(lambda x:getNum(x)).astype(int)

    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['CabinNum_Scalered'] = scaler.fit_transform(df['CabinNum'])

def getFirstLetter(x):
    match = re.compile("([a-zA-Z]+)").search(x)
    if match:
        return match.group()
    else:
        return  'U'
def getNum(x):
    match = re.compile("([0-9]+)").search(x)
    if match:
        return match.group()
    else:
        return 0



def processFare(df):
    # 用中值填充
    df['Fare'][np.isnan(df['Fare'])] = df['Fare'].median()
    # 防止因为值为0带来的除法问题，将为0的值设置为最小值的十分之一
    df['Fare'][np.where(df['Fare']==0)[0]]=df['Fare'][df['Fare'].nonzero()[0]].min()/10
    # 对Fare变量进行处理 分块:连续数据离散化的常见手段，对于对类别敏感的算法效果好
    # 将“Fare”五等分
    # print df['Fare'].max()," ",df['Fare'].min()," ",df['Fare'].median()
    df['Fare_bin'] = pd.qcut(df['Fare'], 5)

    if keep_binary:
        df = pd.concat([df, pd.get_dummies(df['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))], axis=1)

    if keep_bins:
        df['Fare_bin_id'] = pd.factorize(df['Fare_bin'])[0] + 1

        # center and scale the fare to use as a continuous variable
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Fare_scaled'] = scaler.fit_transform(df['Fare'])

    if keep_bins and keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Fare_bin_id_scaled'] = scaler.fit_transform(df['Fare_bin_id'])


def processEmbarked(df):
    # 用众数填充
    df['Embarked'][df.Embarked.isnull()] = df['Fare'].dropna().mode().values

    # Lets turn this into a number so it conforms to decision tree feature requirements
    df['Embarked'] = pd.factorize(df['Embarked'])[0]

    if keep_binary:
        # 将定性的变量转化为哑变量,转换为哑变量适合变量值种类较少的变量
        dummies_embarked = pd.get_dummies(df['Embarked'], prefix='Embarked', prefix_sep='_')
        df = pd.concat([df,dummies_embarked],axis=1)

def processPClass(df):
    # 用众数填充
    df.PClass[df.PClass.isnull()]=df.PClass.dropna().mode().values
    if keep_binary:
        df = pd.concat([df,pd.get_dummies(df.PClass).rename(columns=lambda x: 'Pclass_' + str(x))],axis=1)

    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Pclass_scaled'] = scaler.fit_transform(df['Pclass'])

def processFamily(df):
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['SibSp_scaled'] = scaler.fit_transform(df['SibSp'])
        df['Parch_scaled'] = scaler.fit_transform(df['Parch'])

    if keep_binary:
        sibsps = pd.get_dummies(df['Sibsp']).rename(columns=lambda x:'Sibsp_'+str(x))
        parchs = pd.get_dummies(df['Parch']).rename(columns = lambda x:'Parchs_'+str(x))
        df = pd.concat([df,sibsps,parchs],axis=1)

def processSex():
    df['Gender'] = np.where(df['Sex']=='male',1,0)


### Generate features from the Name variable
def processName():
    global df
    # how many different names do they have?
    df['Names'] = df['Name'].map(lambda x: len(re.split(' ', x)))

    # what is each person's title?
    df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])

    # group low-occuring, related titles together
    df['Title'][df.Title == 'Jonkheer'] = 'Master'
    df['Title'][df.Title.isin(['Ms', 'Mlle'])] = 'Miss'
    df['Title'][df.Title == 'Mme'] = 'Mrs'
    df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
    df['Title'][df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'

    # Build binary features
    if keep_binary:
        df = pd.concat([df, pd.get_dummies(df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1)

    # process scaling
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Names_scaled'] = scaler.fit_transform(df['Names'])

    if keep_bins:
        df['Title_id'] = pd.factorize(df['Title'])[0] + 1

    if keep_bins and keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Title_id_scaled'] = scaler.fit_transform(df['Title_id'])


def processAge(df):
    import Age_predict
    Age_predict.setMissingAges(df)

    if keep_scaled:
        scaled = preprocessing.StandardScaler()
        df['Age_scaled']=scaled.fit_transform(df['Age'])

    # 特征衍生
    # bin into quartiles and create binary features
    df['Age_bin'] = pd.qcut(df['Age'], 4)
    if keep_binary:
        df = pd.concat([df, pd.get_dummies(df['Age_bin']).rename(columns=lambda x: 'Age_' + str(x))], axis=1)

    if keep_bins:
        df['Age_bin_id'] = pd.factorize(df['Age_bin'])[0] + 1

    if keep_bins and keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Age_bin_id_scaled'] = scaler.fit_transform(df['Age_bin_id'])

    if not keep_strings:
        df.drop('Age_bin', axis=1, inplace=True)

# 捕捉变量之间的特征
def variable_interaction(df):
    numerics = df.loc[:, ['Age_scaled', 'Fare_scaled', 'Pclass_scaled', 'Parch_scaled', 'SibSp_scaled',
                          'Names_scaled', 'CabinNumber_scaled', 'Age_bin_id_scaled', 'Fare_bin_id_scaled']]

    # 基本的四则运算示例
    for i in range(0, numerics.columns.size - 1):
        for j in range(0, numerics.columns.size - 1):
            col1 = str(numerics.columns.values[i])
            col2 = str(numerics.columns.values[j])
            # 乘
            if i <= j:
                name = col1 + "*" + col2
                df = pd.concat([df, pd.Series(numerics.iloc[:, i] * numerics.iloc[:, j], name=name)], axis=1)
            # 加
            if i < j:
                name = col1 + "+" + col2
                df = pd.concat([df, pd.Series(numerics.iloc[:, i] + numerics.iloc[:, j], name=name)], axis=1)
            # 除
            if not i == j:
                name = col1 + "/" + col2
                df = pd.concat([df, pd.Series(numerics.iloc[:, i] / numerics.iloc[:, j], name=name)], axis=1)
                name = col1 + "-" + col2
                df = pd.concat([df, pd.Series(numerics.iloc[:, i] - numerics.iloc[:, j], name=name)], axis=1)

# 去掉相关性强的变量，防止多重共线性
def variable_drop():
    # 计算斯皮尔曼相关系数（矩阵）
    df_corr = df.drop(['Survived', 'PassengerId'], axis=1).corr(method='spearman')

    # 将对角线变为0
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
    df_corr = mask * df_corr

    drops = []
    # 循环
    for col in df_corr.columns.values:
        # if we've already determined to drop the current variable, continue
        if np.in1d([col], drops):
            continue

        # 找出高相关的变量
        corr = df_corr[abs(df_corr[col]) > 0.98].index
        drops = np.union1d(drops, corr)

    print "nDropping", drops.shape[0], "highly correlated features...n", drops
    df.drop(drops, axis=1, inplace=True)

def reduce_dimention(input_df, submit_df):
    # 先把两个数据合并
    df = pd.concat([input_df,submit_df])
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    df.reindex_axis(input_df.columns,axis=1)

    # Split into feature and label arrays
    X = df.values[:, 1::]
    y = df.values[:, 0]

    # Series of labels
    survivedSeries = pd.Series(df['Survived'], name='Survived')

    from sklearn.decomposition import PCA
    variance_pct = .99

    # 建一个PCA的对象
    pca = PCA(n_components=variance_pct)
    X_transformed = pca.fit_transform(x, y)
    pcaDataFrame = pd.DataFrame(X_transformed)

    print pcaDataFrame.shape[1], " components describe ", str(variance_pct)[1:], "% of the variance"

    # split into separate input and test sets again
    input_df = df[:input_df.shape[0]]
    submit_df = df[input_df.shape[0]:]
    submit_df.reset_index(inplace=True)
    submit_df.drop('index', axis=1, inplace=True)
    submit_df.drop('Survived', axis=1, inplace=1)

    return input_df, submit_df