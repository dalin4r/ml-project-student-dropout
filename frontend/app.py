from flask import Flask, render_template, request
from ucimlrepo import fetch_ucirepo
from markupsafe import Markup
import io
import base64

# Data Processing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

# Evaluation Metrics

from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score

# NN
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

# SVM
from sklearn.svm import SVC
from sklearn import preprocessing

# Decision Tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

app = Flask(__name__)

classifiers = [
    "Decision Tree",
    "Neural Network",
    "Support Vector Machine",
    "Random Forest",
    "Data Inspection"
]


def select_feature_chi2(ar_num, dataset, is_print=1):
    output = ""
    X = dataset.drop(columns='Target')
    y = dataset['Target']

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_rescaled = scaler.fit_transform(X)
    X = pd.DataFrame(data=X_rescaled, columns=X.columns)

    selector = SelectKBest(chi2, k=ar_num)
    X_selected = selector.fit_transform(X_rescaled, y)
    # Get the indices of the selected features
    selected_indices = selector.get_support(indices=True)

    # Get the names of the selected features
    selected_features = X.columns[selected_indices]
    if (is_print):
        output += "Top " + str(ar_num) + " significant features:<br>"
        output += str(selected_features.tolist())
        return output
    else:
        return selected_features

def DataInspection(dataset):

    def heatmap(df):
        matplotlib.use('agg')
        selected_features = select_feature_chi2(10, df, is_print=0)
        df_selected_features = df[selected_features.tolist() + ['Target']]
        mapping = {'Dropout': 1, 'Enrolled': 2, 'Graduate': 3}
        df_selected_features['Target'] = df_selected_features['Target'].replace(mapping)

        sns.heatmap(df_selected_features.corr(), vmin=-1, vmax=1, center=0, annot=True, cmap='coolwarm')

        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)

        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        img.close()
        return plot_url

    output = "Target : "
    output += str(dataset['Target'].unique()) + "<br>"

    output += "Dimensions of the dataset : "
    output = output + str(dataset.shape) + "<br>"

    # output_html = "Data Inspection<br>"
    output_html = output
    output_html = Markup(output_html)

    counts = dataset["Target"].value_counts().reset_index()
    counts_html = counts.to_html(index=False)

    # heatmap(dataset)

    return render_template('data-inspection.html', result=output_html, table=counts_html, plot_url=heatmap(dataset))

def DecisionTree(dataset):
    output_html = ""
    X = dataset.drop(columns='Target')
    y = dataset['Target']

    # normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_rescaled = scaler.fit_transform(X)
    X = pd.DataFrame(data=X_rescaled, columns=X.columns)

    def Deci_Tree_less_Features(ar_selected_features):
        output = ""
        columns = ar_selected_features.tolist()
        columns = columns + ['Target']
        df_DTC = dataset[columns]

        train, test = train_test_split(df_DTC, test_size=0.2)

        X_train, y_train = train.drop(columns=['Target']), train['Target']
        X_test, y_test = test.drop(columns=['Target']), test['Target']

        clf = DecisionTreeClassifier(random_state=1234)
        dtree_model = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        output += "Accuracy: " + str(accuracy_score(y_test, y_pred)) + "<br>"
        return output

    for i in range(X.columns.size - 20, 0,
                   -2):  # since SVM takes more time to train, we start with even less features than NN
        selector = SelectKBest(chi2, k=i)
        X_selected = selector.fit_transform(X_rescaled, y)

        # Get the indices of the selected features
        selected_indices = selector.get_support(indices=True)

        # Get the names of the selected features
        selected_features = X.columns[selected_indices]
        output_html += str(i) + " features selected:<br>"
        output_html += Deci_Tree_less_Features(selected_features) + "<br>"

    output_html = Markup(output_html)
    return render_template('output.html', model_name="Decision Tree", result=output_html)


def RandomForest(dataset):
    output_html = ""
    X = dataset.drop(columns='Target')
    y = dataset['Target']

    # normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_rescaled = scaler.fit_transform(X)
    X = pd.DataFrame(data=X_rescaled, columns=X.columns)

    def Random_forest_less_Features(ar_selected_features):
        output = ""

        columns = ar_selected_features.tolist()
        columns = columns + ['Target']
        df_RDC = dataset[columns]

        train, test = train_test_split(df_RDC, test_size=0.2)

        X_train, y_train = train.drop(columns=['Target']), train['Target']
        X_test, y_test = test.drop(columns=['Target']), test['Target']

        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        rf_classifier.fit(X_train, y_train)

        y_pred = rf_classifier.predict(X_test)
        output += "Accuracy: " + str(accuracy_score(y_test, y_pred)) + "<br>"
        return output

    for i in range(X.columns.size - 25, 0, -1):
        selector = SelectKBest(chi2, k=i)
        X_selected = selector.fit_transform(X_rescaled, y)

        # Get the indices of the selected features
        selected_indices = selector.get_support(indices=True)

        # Get the names of the selected features
        selected_features = X.columns[selected_indices]
        output_html += str(i) + " features selected:<br>"
        output_html += Random_forest_less_Features(selected_features) + "<br>"

    output_html = Markup(output_html)
    return render_template('output.html', model_name="Random Forest", result=output_html)


def NeuralNetwork(dataset):
    output_html = ""
    X = dataset.drop(columns='Target')
    y = dataset['Target']

    # normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_rescaled = scaler.fit_transform(X)
    X = pd.DataFrame(data=X_rescaled, columns=X.columns)

    def NN_Model_less_Feature(ar_selected_features):
        output = ""
        X = dataset[ar_selected_features]
        y = dataset['Target']

        categories = [['Enrolled', 'Graduate', 'Dropout']]
        encoder = OneHotEncoder(categories=categories, sparse_output=False)
        y = encoder.fit_transform(y.values.reshape(-1, 1))

        data_train, data_test, class_train, class_test = train_test_split(X, y, test_size=0.2)

        mlp = MLPClassifier(solver='sgd', random_state=42, activation='logistic', learning_rate_init=0.4,
                            batch_size=100, hidden_layer_sizes=(23, 17, 12), max_iter=500)
        mlp.fit(data_train, class_train)
        pred = mlp.predict(data_test)

        output += "Accuracy: " + str(accuracy_score(class_test, pred)) + "<br>"
        return output

    # from tqdm import tqdm
    for i in range(X.columns.size - 20, 0, -2):
        selector = SelectKBest(chi2, k=i)
        X_selected = selector.fit_transform(X_rescaled, y)

        # Get the indices of the selected features
        selected_indices = selector.get_support(indices=True)

        # Get the names of the selected features
        selected_features = X.columns[selected_indices]
        # print(f'{i} features selected:')
        output_html += str(i) + " features selected:<br>"
        output_html += NN_Model_less_Feature(selected_features) + "<br>"

    output_html += select_feature_chi2(8, dataset=dataset)
    output_html = Markup(output_html)
    return render_template('output.html', model_name="Neural Network", result=output_html)


def SupportVectorMachine(dataset):
    output_html = ""
    X = dataset.drop(columns='Target')
    y = dataset['Target']

    # normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_rescaled = scaler.fit_transform(X)
    X = pd.DataFrame(data=X_rescaled, columns=X.columns)

    def report_confusion_matrices_and_other_measurement(ar_class_test, ar_pred):
        output = ""
        output += "Accuracy: " + str(accuracy_score(ar_class_test, ar_pred)) + "<br>"
        mcm = multilabel_confusion_matrix(ar_class_test, ar_pred)
        categories = [['Enrolled', 'Graduate', 'Dropout']]
        unique_labels = categories[0]
        for i, label in enumerate(unique_labels):
            output += "Confusion Matrix for label " + label + "<br>"
            cm = mcm[i]
            output += str(cm) + "<br>"
            tp = cm[1, 1]
            fn = cm[1, 0]
            fp = cm[0, 1]
            tn = cm[0, 0]

            output += "True Positive (TP)  : " + str(tp) + "<br>"
            output += "False Negative (FN)  : " + str(fn) + "<br>"
            output += "False Positive (FP)  : " + str(fp) + "<br>"
            output += "True Negative (TN)  : " + str(tn) + "<br>"
            output += "<br>"

        # print("Mean Square Error : ", mean_squared_error(ar_class_test, ar_pred))
        output += "Classification Report : "
        report = classification_report(ar_class_test, ar_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_html = report_df.to_html(classes='table table-striped')

        return output, report_html

    def SVM_Model_less_Features(ar_selected_features, includes_classification_report):
        output = ""
        # X = df[ar_selected_features]
        # y = df['Target']
        columns = ar_selected_features.tolist()
        columns.append('Target')
        df_svm = dataset[columns]
        df_svm = pd.get_dummies(df_svm, columns=ar_selected_features)
        # df_svm = df_svm.astype(int)

        svm_train, svm_test = train_test_split(df_svm, test_size=0.2)

        X_svm_train, y_svm_train = svm_train.drop(columns=['Target']), svm_train['Target']
        X_svm_test, y_svm_test = svm_test.drop(columns=['Target']), svm_test['Target']

        scaler = preprocessing.StandardScaler()
        scaler.fit(X_svm_train)
        Z_svm_train = scaler.transform(X_svm_train)
        Z_svm_test = scaler.transform(X_svm_test)

        svm_li = SVC(kernel='linear')
        svm_li.fit(Z_svm_train, np.asarray(y_svm_train))
        y_pred_li = svm_li.predict(Z_svm_test)
        output += "Linear Kernel<br>"
        if includes_classification_report:
            output_result, report_html = report_confusion_matrices_and_other_measurement(y_svm_test, y_pred_li)
            output += output_result
            return output, report_html
        else:
            output += "Accuracy: " + str(accuracy_score(y_svm_test, y_pred_li)) + "<br>"

        svc_rbf = SVC(kernel='rbf')
        svc_rbf.fit(Z_svm_train, np.asarray(y_svm_train))
        y_pred_rbf = svc_rbf.predict(Z_svm_test)
        output += "RBF Kernel<br>"
        if includes_classification_report:
            output_result, report_html = report_confusion_matrices_and_other_measurement(y_svm_test, y_pred_rbf)
            output += output_result
            return output, report_html
        else:
            output += "Accuracy: " + str(accuracy_score(y_svm_test, y_pred_rbf)) + "<br>"

        return output, ""

    for i in range(X.columns.size - 26, 0,
                   -1):  # since SVM takes more time to train, we start with even less features than NN Model
        selector = SelectKBest(chi2, k=i)
        X_selected = selector.fit_transform(X_rescaled, y)

        # Get the indices of the selected features
        selected_indices = selector.get_support(indices=True)

        # Get the names of the selected features
        selected_features = X.columns[selected_indices]
        # print(f'{i} features selected:')
        output_html += str(i) + " features selected:<br>"
        output_res, _ = SVM_Model_less_Features(selected_features, includes_classification_report=False)
        output_html += output_res + "<br>"

    output_html += select_feature_chi2(4, dataset=dataset) + "<br>"
    output_res, report_html = SVM_Model_less_Features(select_feature_chi2(4, dataset=dataset, is_print=False),
                                                      includes_classification_report=True)
    output_html += output_res + "<br>"
    output_html = Markup(output_html)
    return render_template('output.html', model_name="Support Vector Machine", result=output_html, report=report_html)


@app.route('/', methods=['GET', 'POST'])
def index():
    dataset = pd.read_csv("data.csv", sep=';')
    if request.method == 'POST':

        selected_classifier = request.form['classifier']

        if selected_classifier == 'Decision Tree':
            return DecisionTree(dataset)
        elif selected_classifier == 'Random Forest':
            return RandomForest(dataset)
        elif selected_classifier == 'Neural Network':
            return NeuralNetwork(dataset)
        elif selected_classifier == 'Support Vector Machine':
            return SupportVectorMachine(dataset)
        elif selected_classifier == 'Data Inspection':
            return DataInspection(dataset)
        else:
            output_html = "<h1>No classifier selected</h1>"
            output_html = Markup(output_html)
            return render_template('output.html', result=output_html)

    return render_template('index.html', classifiers=classifiers)


if __name__ == '__main__':
    app.run(debug=True)

    # Here you can perform actions based on the selected classifier

