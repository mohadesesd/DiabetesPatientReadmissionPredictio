from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('always')

def get_data():
    data = pd.read_csv('/home/msd/Downloads/dataset_diabetes/diabetic_data.csv')
    return data

def Datapreprocessing(data):
    data['readmitted'] = pd.Series([0 if val == 'NO' else 1 for val in data['readmitted']])
    data.drop(['encounter_id', 'patient_nbr', 'payer_code'], axis=1, inplace=True)
    data[data['weight'] == '?'].shape[0] * 1.0 / data.shape[0]
    data[data['medical_specialty'] == '?'].shape[0] * 1.0 / data.shape[0]
    data.drop(['weight', 'medical_specialty'], axis=1, inplace=True)
    data = data[data['race'] != '?']
    data = data[data['diag_1'] != '?']
    data = data[data['diag_2'] != '?']
    data = data[data['diag_3'] != '?']
    data = data[data['gender'] != 'Unknown/Invalid']
    data['age'] = pd.Series(['[0-50)' if val in ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)'] else val
                             for val in data['age']], index=data.index)
    data['age'] = pd.Series(['[80-100)' if val in ['[80-90)', '[90-100)'] else val
                             for val in data['age']], index=data.index)
    data['discharge_disposition_id'] = pd.Series(['Home' if val == 1 else 'Other discharge'
                                                  for val in data['discharge_disposition_id']], index=data.index)
    data['admission_source_id'] = pd.Series(
        ['Emergency Room' if val == 7 else 'Referral' if val == 1 else 'Other source'
         for val in data['admission_source_id']], index=data.index)
    data['admission_type_id'] = pd.Series(['Emergency' if val == 1 else 'Other type'
                                           for val in data['admission_type_id']], index=data.index)
    data['diag_1'] = pd.Series(['9' if val[0]=="E" or val[0]=="V" else float(val)
                              for val in data['diag_1']],index=data.index)
    data['diag_1']= pd.Series(['1' if  not isinstance(val, str) and val < 240  else val
                              for val in data['diag_1']], index=data.index)
    data['diag_1'] = pd.Series(['2' if not isinstance(val, str) and val < 290 else val
                               for val in data['diag_1']], index=data.index)
    data['diag_1'] = pd.Series(['3' if not isinstance(val, str) and val < 390 else val
                               for val in data['diag_1']], index=data.index)
    data['diag_1'] = pd.Series(['4' if not isinstance(val, str) and val < 520 else val
                               for val in data['diag_1']], index=data.index)
    data['diag_1'] = pd.Series(['5' if not isinstance(val, str) and val < 630 else val
                               for val in data['diag_1']], index=data.index)
    data['diag_1'] = pd.Series(['6' if not isinstance(val, str) and val < 710 else val
                               for val in data['diag_1']], index=data.index)
    data['diag_1'] = pd.Series(['7' if not isinstance(val, str) and val < 760 else val
                               for val in data['diag_1']], index=data.index)
    data['diag_1'] = pd.Series(['8' if not isinstance(val, str) and val <= 999 else val
                               for val in data['diag_1']], index=data.index)
    data['diag_2'] = pd.Series(['9' if val[0] == "E" or val[0] == "V" else float(val)
                                for val in data['diag_2']], index=data.index)
    data['diag_2'] = pd.Series(['1' if not isinstance(val, str) and val < 240 else val
                                for val in data['diag_2']], index=data.index)
    data['diag_2'] = pd.Series(['2' if not isinstance(val, str) and val < 290 else val
                                for val in data['diag_2']], index=data.index)
    data['diag_2'] = pd.Series(['3' if not isinstance(val, str) and val < 390 else val
                                for val in data['diag_2']], index=data.index)
    data['diag_2'] = pd.Series(['4' if not isinstance(val, str) and val < 520 else val
                                for val in data['diag_2']], index=data.index)
    data['diag_2'] = pd.Series(['5' if not isinstance(val, str) and val < 630 else val
                                for val in data['diag_2']], index=data.index)
    data['diag_2'] = pd.Series(['6' if not isinstance(val, str) and val < 710 else val
                                for val in data['diag_2']], index=data.index)
    data['diag_2'] = pd.Series(['7' if not isinstance(val, str) and val < 760 else val
                                for val in data['diag_2']], index=data.index)
    data['diag_2'] = pd.Series(['8' if not isinstance(val, str) and val < 999 else val
                                for val in data['diag_2']], index=data.index)
    data['diag_3'] = pd.Series(['9' if val[0] == "E" or val[0] == "V" else float(val)
                                for val in data['diag_3']], index=data.index)
    data['diag_3'] = pd.Series(['1' if not isinstance(val, str) and val < 240 else val
                                for val in data['diag_3']], index=data.index)
    data['diag_3'] = pd.Series(['2' if not isinstance(val, str) and val < 290 else val
                                for val in data['diag_3']], index=data.index)
    data['diag_3'] = pd.Series(['3' if not isinstance(val, str) and val < 390 else val
                                for val in data['diag_3']], index=data.index)
    data['diag_3'] = pd.Series(['4' if not isinstance(val, str) and val < 520 else val
                                for val in data['diag_3']], index=data.index)
    data['diag_3'] = pd.Series(['5' if not isinstance(val, str) and val < 630 else val
                                for val in data['diag_3']], index=data.index)
    data['diag_3'] = pd.Series(['6' if not isinstance(val, str) and val < 710 else val
                                for val in data['diag_3']], index=data.index)
    data['diag_3'] = pd.Series(['7' if not isinstance(val, str) and val < 760 else val
                                for val in data['diag_3']], index=data.index)
    data['diag_3'] = pd.Series(['8' if not isinstance(val, str) and val < 999 else val
                                for val in data['diag_3']], index=data.index)
    data['max_glu_serum']=pd.Series(['1' if val=='None' else val
                                     for val in data['max_glu_serum']])
    data['max_glu_serum'] = pd.Series(['2' if val == 'Norm' else val
                                       for val in data['max_glu_serum']])
    data['max_glu_serum'] = pd.Series(['3' if val == '>300' else val
                                       for val in data['max_glu_serum']])
    data['max_glu_serum'] = pd.Series(['4' if val == '>200' else val
                                       for val in data['max_glu_serum']])
    data['change']=pd.Series(['1' if val=='No' else '2'
                                     for val in data['change']])

    return data

def normalization(data):
    df = pd.DataFrame(data)
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)
    return df_normalized

def PCA(data):
    feature_scale_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                          'number_diagnoses', 'number_inpatient', 'number_emergency', 'number_outpatient']
    scaler = preprocessing.StandardScaler().fit(data[feature_scale_cols])
    data_scaler = scaler.transform(data[feature_scale_cols])
    data_scaler_df = pd.DataFrame(data=data_scaler, columns=feature_scale_cols, index=data.index)
    data.drop(feature_scale_cols, axis=1, inplace=True)
    data = pd.concat([data, data_scaler_df], axis=1)
    return data

def data_divider(data):
    for column in data.columns:
        if data[column].dtype == type(object):
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].astype(str))
    X = data.drop(['readmitted'], axis=1)
    X = X.drop(['A1Cresult'], axis=1)
    X = X.drop(['diag_1'],axis=1)
    X=X.values

    z = data['A1Cresult']
    z=z.values

    y = data['readmitted']
    y=y.values

    w = data['diag_1']
    w=w.values

    dataSplite={
        'X':X,
        'readmitted':y,
        'HbA1Cresult':z,
        'diag_1':w
    }
    return dataSplite

def Kmean_clustering(X,y):
    correct = 0
    for i in range(len(X)):
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = KMeans.predict(predict_me)
        if prediction[0] == y[i]:
            correct += 1
    return(correct / len(X))

def MLP(datadic):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (5, 2), random_state = 1)
    clf.fit(datadic['Xtrain'], datadic['Ytrain'])
    per = clf.predict(datadic['Xtest'])
    return metrics.accuracy_score(datadic['Ytest'], per)

def KNN(datadic):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(datadic['Xtrain'], datadic['Ytrain'])
    per=neigh.predict(datadic['Xtest'])
    return metrics.accuracy_score(datadic['Ytest'], per)

def NaiveBayes(datadic):
    gnb = GaussianNB()
    gnb.fit(datadic['Xtrain'], datadic['Ytrain'])
    y_pred= gnb.predict(datadic['Xtest'])
    return metrics.accuracy_score(datadic['Ytest'], y_pred)

def SVM(datadic):
    clf = svm.SVC(gamma='scale')
    clf.fit(datadic['Xtrain'], datadic['Ytrain'])
    y_pred_class = clf.predict(datadic['Xtest'])
    return metrics.accuracy_score(datadic['Ytest'], y_pred_class)

def Boosting(datadic):
    AdaBoost = AdaBoostClassifier(n_estimators=400, learning_rate=1, algorithm='SAMME')
    AdaBoost.fit(datadic['Xtrain'], datadic['Ytrain'])
    y_pred_class = AdaBoost.predict(datadic['Xtest'])
    return metrics.accuracy_score(datadic['Ytest'], y_pred_class)

def Voting_classifier(datadic):
    clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state = 1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (5, 2), random_state = 1)
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    eclf = eclf.fit(datadic['Xtrain'], datadic['Ytrain'])
    accuracy=eclf.score(datadic['Xtest'],datadic['Ytest'])
    return accuracy

def logesticregression(X_train,y_train,X_test,y_test):
    C_range = np.arange(0.1, 3.1, 0.2)
    param_grid = dict(C=C_range)
    clf = LogisticRegression()
    grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
    grid.fit(X_train, y_train)
    logreg = LogisticRegression(C=grid.best_params_['C'])
    logreg.fit(X_train, y_train)
    y_pred_class=logreg.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred_class)

def confusion_matrix(X,y):
    print("Result for Hb1Cresult : ")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    # clf = svm.SVC(gamma='scale')
    # clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state = 1,max_iter=500)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    # clf3 = KNeighborsClassifier(n_neighbors=3)
    # clf4 = GaussianNB()
    # clf5 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1,max_iter=400)
    # clf6 = AdaBoostClassifier(n_estimators=400, learning_rate=1, algorithm='SAMME')
    # eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
    eclf = clf2.fit(X_train, y_train)
    y_pred_class = eclf.predict(X_test)
    print("Confusion matrix:")
    confusion = metrics.confusion_matrix(y_test, y_pred_class)
    print(confusion)
    print("Recall:")
    print(metrics.recall_score(y_test, y_pred_class, average='weighted', labels=np.unique(y_pred_class)))
    print("Precision:")
    print(metrics.precision_score(y_test, y_pred_class,average='weighted', labels=np.unique(y_pred_class)))
    print("F_score:")
    print(metrics.f1_score(y_test, y_pred_class, average='weighted', labels=np.unique(y_pred_class)))


def K_fold_cross_validation(X,y):
        accuracy_KNN = 0
        accuracy_NaiveBayes = 0
        accuracy_SVM = 0
        accuracy_boosting = 0
        accuracy_logisticregression=0
        accuracy_MLP=0
        kf = KFold(n_splits=100)
        kf.get_n_splits(X, y)
        for train_index, test_index in kf.split(X):
            X_train = []
            X_test = []
            y_train = []
            y_test = []
            for i in train_index:
                X_train.append(X[i])
                y_train.append(y[i])
            for i in test_index:
                X_test.append(X[i])
                y_test.append(y[i])
            datadic = {'Xtrain': X_train, 'Ytrain': y_train, 'Xtest': X_test, 'Ytest': y_test}
            accuracy_MLP += MLP(datadic)
            accuracy_KNN += KNN(datadic)
            accuracy_NaiveBayes += NaiveBayes(datadic)
            accuracy_SVM += SVM(datadic)
            accuracy_boosting += Boosting(datadic)
            accuracy_logisticregression += logesticregression(X_train,y_train,X_test,y_test)
        accuracydic = {'Accuracy_KNN': accuracy_KNN, 'Accuracy_NaiveBayes': accuracy_NaiveBayes,
                       'Accuracy_SVM': accuracy_SVM, 'Accuracy_boosting': accuracy_boosting,'Accuracy_logisticRegression':accuracy_logisticregression,
                       'Accuracy_MLP':accuracy_MLP}
        return accuracydic

print(K_fold_cross_validation(data_divider(PCA(Datapreprocessing(get_data())))['X'],data_divider(PCA(Datapreprocessing(get_data())))['readmitted']))
confusion_matrix(data_divider(PCA(Datapreprocessing(get_data())))['X'],data_divider(PCA(Datapreprocessing(get_data())))['readmitted'])
print(K_fold_cross_validation(data_divider(PCA(Datapreprocessing(get_data())))['X'],data_divider(PCA(Datapreprocessing(get_data())))['Hb1ACresult']))
confusion_matrix(data_divider(PCA(Datapreprocessing(get_data())))['X'],data_divider(PCA(Datapreprocessing(get_data())))['Hb1ACresult'])
print(K_fold_cross_validation(data_divider(PCA(Datapreprocessing(get_data())))['X'],data_divider(PCA(Datapreprocessing(get_data())))['diag_1']))
confusion_matrix(data_divider(PCA(Datapreprocessing(get_data())))['X'],data_divider(PCA(Datapreprocessing(get_data())))['diag_1'])