import numpy as np
import pandas as pd
from sklearn import ensemble, preprocessing, metrics, model_selection


_use_one_hot_encoder = True

# 載入資料
titanic_train = pd.read_csv('train.csv')
titanic_train = titanic_train.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'], axis=1)

# 填補遺漏值
age_median = np.nanmedian(titanic_train["Age"])
new_Age = np.where(titanic_train["Age"].isnull(), age_median, titanic_train["Age"])
titanic_train["Age"] = new_Age

# LabelEncoder
label_encoder = preprocessing.LabelEncoder()
titanic_train["Sex"] = label_encoder.fit_transform(titanic_train["Sex"])

# 建立訓練與測試資料
if _use_one_hot_encoder:
    onehotencoder = preprocessing.OneHotEncoder(categorical_features = [2])
    data_str_ohe = onehotencoder.fit_transform(titanic_train).toarray()
    titanic_X = pd.DataFrame([data_str_ohe[:,1],
                              data_str_ohe[:,2],
                              data_str_ohe[:,3],
                              data_str_ohe[:,4]]).T
    titanic_y = titanic_train["Survived"]
else:
    titanic_X = pd.DataFrame([titanic_train["Pclass"],
                              titanic_train["Sex"],
                              titanic_train["Age"]]).T
    titanic_y = titanic_train["Survived"]


train_X, test_X, train_y, test_y = model_selection.train_test_split(titanic_X, titanic_y, test_size = 0.3)

# 建立 random forest 模型
forest = ensemble.RandomForestClassifier(n_estimators = 100)
forest_fit = forest.fit(train_X, train_y)

# 預測
test_y_predicted = forest.predict(test_X)

# 績效
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print('Accuracy: %f' % (accuracy))

fpr, tpr, thresholds = metrics.roc_curve(test_y, test_y_predicted)
auc = metrics.auc(fpr, tpr)
print('AUC     : %f' % (auc))

# 重要程度
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels = titanic_train.columns[1:]
print("\nFeature Importances: ")
for f in range(3):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
