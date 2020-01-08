from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import csv
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier


def get_data():
    data = csv.reader(open('/home/msd/Downloads/pd_speech_features.csv'))
    ds=[]
    for row in data:
        ds.append(list(row))
    return ds

def data_divider(ds):
    X = []
    for row in range(2, len(ds)):
        X.append(ds[row][:-1])
    y = []
    for row in range(2, len(ds)):
        y.append(ds[row][-1])
    return X,y

def KNN(datadic):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(datadic['Xtrain'], datadic['Ytrain'])
    accuracy=neigh.score(datadic['Xtest'],datadic['Ytest'])
    return accuracy

def Random_Forest_classifier(datadic):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state = 0)
    clf.fit(datadic['Xtrain'], datadic['Ytrain'])
    return clf.score(datadic['Xtest'],datadic['Ytest'])

def Voting_classifier(datadic):
    clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state = 1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (5, 2), random_state = 1)
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    eclf = eclf.fit(datadic['Xtrain'], datadic['Ytrain'])
    accuracy=eclf.score(datadic['Xtest'],datadic['Ytest'])
    return accuracy

def NaiveBayes(datadic):
    gnb = GaussianNB()
    gnb.fit(datadic['Xtrain'], datadic['Ytrain'])
    accuracy= gnb.score(datadic['Xtest'],datadic['Ytest'])
    return accuracy


def SVM(datadic):
    clf = svm.SVC(gamma='scale')
    clf.fit(datadic['Xtrain'], datadic['Ytrain'])
    accuracy =clf.score(datadic['Xtest'], datadic['Ytest'])
    return accuracy


def Boosting(datadic):
    AdaBoost = AdaBoostClassifier(n_estimators=400, learning_rate=1, algorithm='SAMME')
    AdaBoost.fit(datadic['Xtrain'], datadic['Ytrain'])
    prediction = AdaBoost.score(datadic['Xtest'], datadic['Ytest'])
    accuracy=prediction
    return accuracy

def MLP(datadic):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (5, 2), random_state = 1)
    clf.fit(datadic['Xtrain'], datadic['Ytrain'])
    accuracy = clf.score(datadic['Xtest'],datadic['Ytest'])
    return accuracy

def RBF(datadic):
    clf = svm.SVC(kernel='rbf',gamma='auto')
    clf.fit(datadic['Xtrain'], datadic['Ytrain'])
    accuracy = clf.score(datadic['Xtest'],datadic['Ytest'])
    return accuracy


def normalize(X):
    scaler=preprocessing.StandardScaler().fit(X)
    X=scaler.transform(X)
    return X

def PCA(X):
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)
    return X

def K_fold(X,y):
    accuracy_randomforest=0
    accuracy_MLP=0
    accuracy_KNN=0
    accuracy_NaiveBayes=0
    accuracy_SVM=0
    accuracy_boosting=0
    accuracy_voting=0
    kf=KFold(n_splits=10)
    kf.get_n_splits(X,y)
    for train_index, test_index in kf.split(X):
        X_train=[]
        X_test=[]
        y_train=[]
        y_test=[]
        for i in train_index:
            X_train.append(X[i])
            y_train.append(y[i])
        for i in test_index:
            X_test.append(X[i])
            y_test.append(y[i])
        datadic={'Xtrain':X_train,'Ytrain':y_train,'Xtest':X_test,'Ytest':y_test}
        accuracy_MLP+=MLP(datadic)
        accuracy_KNN += KNN(datadic)
        accuracy_NaiveBayes += NaiveBayes(datadic)
        accuracy_SVM += SVM(datadic)
        accuracy_boosting += Boosting(datadic)
        accuracy_randomforest+=Random_Forest_classifier(datadic)
        accuracy_voting+=Voting_classifier(datadic)
    accuracy_voting *=10
    accuracy_KNN *= 10
    accuracy_NaiveBayes *= 10
    accuracy_SVM *= 10
    accuracy_boosting *= 10
    accuracy_MLP*=10
    accuracy_randomforest*=10
    accuracydic={'Accuracy_Voting':accuracy_voting,'Accuracy_KNN':accuracy_KNN,'Accuracy_NaiveBayes':accuracy_NaiveBayes,'Accuracy_SVM':accuracy_SVM,'Accuracy_boosting':accuracy_boosting,'Accuracy_MLP':accuracy_MLP,'Accuracy_Random_forest':accuracy_randomforest}
    return accuracydic

if __name__ == '__main__':
    print("Accuracy for each method with normalized dataset:")
    dic=K_fold(normalize(data_divider(get_data())[0]),data_divider(get_data())[1])
    # print("Accuracy Voting:")
    # print(dic['Accuracy_Voting'])
    print('Accuracy Random Forest:')
    print(dic['Accuracy_Random_forest'])
    print('Accuracy MLP:')
    print(dic['Accuracy_MLP'])
    print('Accuracy KNN:')
    print(dic['Accuracy_KNN'])
    print('Accuracy Naivebayes:')
    print(dic['Accuracy_NaiveBayes'])
    print('Accuracy SVM :')
    print(dic['Accuracy_SVM'])
    print('Accuracy boosting:')
    print(dic['Accuracy_boosting'])
    print("Accuracy for each method with PCA:")
    dic=K_fold(PCA(normalize(data_divider(get_data())[0])),data_divider(get_data())[1])
    print('Accuracy KNN:')
    print(dic['Accuracy_KNN'])
    print('Accuracy Naivebayes:')
    print(dic['Accuracy_NaiveBayes'])
    print('Accuracy SVM :')
    print(dic['Accuracy_SVM'])
    print('Accuracy boosting:')
    print(dic['Accuracy_boosting'])
    print('Accuracy MLP')
    print(dic['Accuracy_MLP'])
    print('Accuracy Random Forest:')
    print(dic['Accuracy_Random_forest'])