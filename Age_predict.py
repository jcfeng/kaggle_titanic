# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def setMissingAges(df):
    # 去模型变量
    age_df = df[['Age', 'Embarked', 'Fare', 'Parch', 'SibSp', 'Ticket', 'Pclass', 'Name', 'Cabin']]
    know_age = age_df.loc[(df.Age.notnull())]
    unknowAge = age_df.loc[df.Age.isnull()]
    # print age_df
    #提取所有年龄值，[]括号里左边的冒号代表从所有行中取，右边的0代表只取第一列，即年龄
    y=know_age.values[:,0]
    #取除了年龄之外的其他值：1代表找从第1列开始的值，因为第0列为年龄
    x = know_age.values[:,1::]
    # print y
    # print ".........."
    # print x
    # print "......."
    # print know_age.values[:,0::]
    print know_age.dtypes
    rtr = RandomForestRegressor(n_estimators=2000,n_jobs=-1)
    rtr.fit(x,y)
    predict_age = rtr.predict(unknowAge[:,1::])
    df.loc[df.Age.isnull()]['Age'] = predict_age

