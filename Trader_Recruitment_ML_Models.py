
# coding: utf-8

# In[1]:

# Import libraries
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:

# Load dataframe and encode categorical features

traders_df = pd.read_csv('xxx_2018-Quant_Trader_Applicant_Data_Cleaned_3.csv')
traders_df = traders_df.rename(columns={'index': 'Applicant #'})
traders_df = traders_df[traders_df['Candidate_Status'] != 'In Progress']
traders_df = traders_df.drop(['Applicant #', 'Unnamed: 0', 'ID', 'Jobs Applied', ' Applications'], axis=1)
traders_df


# In[42]:

# Queries
print((traders_df[(traders_df['Last Stage'] == 'Offer')].groupby('University').size() / traders_df.groupby('University').size()).nlargest(20))
print('\n')
print(traders_df[(traders_df['Last Stage'] == 'Offer')].groupby('University').size().nlargest(20))
print('\n')
print(traders_df.groupby('University').size().nlargest(20))


# ### Classification steps:
# 1. Normalize/standardize data
# 2. Categorical feature encoding/transformation
# 3. Feature engineering: generation, selection, removal (correlation analysis, add higher-order terms)
# 4. Apply different models, tune hyperparameters via grid search (e.g. regularization param)

# In[3]:

features = np.array(traders_df.columns)
continuous_features = ['Codility Battleship', 'Saville Diagramatic', 'Saville Numerical', 'Best Score', 'GPA']
categorical_features = np.setdiff1d(features, continuous_features)
categorical_features


# In[4]:

# More data cleaning

# Ensures that missing values are unique across columns so one-hot encoding method doesn't complain about duplicate columns
def replace_missing_vals(val, prefix):
    if val == 'Not Listed':
        return prefix + ' Not Listed'
    return val

# Drop unnecessary columns
traders_df = traders_df.drop(['Candidate_Status', 'Best standardized test', 'Disposition Reason'], axis=1)

traders_df['Sponsorship'] = traders_df['Sponsorship'].apply(lambda sponsorship : 'Sponsorship ' + sponsorship)
traders_df['Will require visa sponsorship'] = traders_df['Will require visa sponsorship'].apply(lambda sponsorship : 'Visa Required ' + sponsorship)
traders_df['Major'] = traders_df['Major'].apply(lambda major : replace_missing_vals(major, 'Major'))
traders_df['Last Degree'] = traders_df['Last Degree'].apply(lambda degree : replace_missing_vals(degree, 'Degree'))

# Split into full-time and intern applicants
full_time_traders_appl = traders_df[traders_df['Job_Posting_Title'] == 'REQ-00438 Graduate Quant Trader (Open)']
intern_traders_appl = traders_df[traders_df['Job_Posting_Title'] == 'REQ-00446 Quant Trader Intern - Summer 2019 (Fill Date: 06/10/2019)']
full_time_traders_appl_labels = full_time_traders_appl['Last Stage']
intern_traders_appl_labels = intern_traders_appl['Last Stage']
full_time_traders_appl = full_time_traders_appl.drop('Last Stage', axis=1)
intern_traders_appl = intern_traders_appl.drop('Last Stage', axis=1)
print(full_time_traders_appl.shape)
print(intern_traders_appl.shape)


# ### Intern Trader Applicants Models

# In[5]:

# 1. Standardize (0 mean, unit variance) or normalize data
from sklearn.preprocessing import StandardScaler

continuous_data = full_time_traders_appl[continuous_features]

std = StandardScaler()
scaler = std.fit(continuous_data)
continuous_data = scaler.transform(continuous_data)
full_time_traders_appl[continuous_features] = continuous_data
full_time_traders_appl


# ### Types of Features:
# 1. Ordinal features: features with values that can be ordered (e.g. age bins, preference scale from 1-10, dates, economic status [low, medium, high], etc.) 
# 2. Nominal features: features with values that cannot be ordered (e.g. sex, sport, university, product type, etc.)
# 
# ### Types of Classic Encodings:
# 1. One-hot encoding (dummy encoding): used for nominal (very often) and ordinal features
# 2. Ordinal encoding: used for ordinal features (1 to k)
# 3. Binary encoding
# 4. BaseN encoding
# 5. Hashing encoding

# ### Encoding for this dataset:
# 1. Applied On (ordinal): date ordinal encoding (done)
# 2. Graduation Date (ordinal): date ordinal encoding (done)
# 3. Major (nominal): one-hot encoding
# 4. Last Degree (ordinal): ordinal encoding
# 5. University (nominal): one-hot encoding
# 6. Sponsorship (nominal): one-hot encoding
# 7. Will require visa sponsorship (nominal): one-hot encoding
# 8. Last Stage (ordinal - CLASS LABEL): ordinal encoding
# 
# Should only be standardized (like continuous data) if regression model is used; can leave as is for tree-based

# In[6]:

# 2. Categorical feature encodings

ordinal_features = ['Applied On', 'Graduation Date', 'Last Degree']
nominal_features = ['Major', 'University', 'Sponsorship', 'Will require visa sponsorship']
class_label = 'Last Stage'

for ordinal_feature in ordinal_features:
    print(full_time_traders_appl.groupby(ordinal_feature).size())
for nominal_feature in nominal_features:
    print(full_time_traders_appl.groupby(nominal_feature).size())


# In[7]:

from datetime import datetime
from dateutil.relativedelta import relativedelta

def convert_date_to_ordinal(date):
    if date == 'Not Lis':
        date = '2019-05' # Replaced with second mode of data (will replace with date mean later) for Graduation Date
    date = datetime.strptime(date, '%Y-%m').date()
    base_date = datetime.strptime('1990-01', '%Y-%m').date()
    date_diff = relativedelta(date, base_date)
    months_diff = date_diff.years * 12 + date_diff.months 
    return months_diff

full_time_traders_appl['Applied On'] = full_time_traders_appl['Applied On'].apply(lambda date : convert_date_to_ordinal(date))
full_time_traders_appl['Graduation Date'] = full_time_traders_appl['Graduation Date'].apply(lambda date : convert_date_to_ordinal(date))


# In[8]:

# Using pd.get_dummies for one-hot encoding since we're dealing with pandas dataframes

def one_hot_encode_features(df, features):
    for feature in features:
        one_hot_feature = pd.get_dummies(df[feature])
        df = df.drop(feature, axis=1)
        df = df.join(one_hot_feature)
    return df

full_time_traders_appl = one_hot_encode_features(full_time_traders_appl, nominal_features)
full_time_traders_appl


# In[9]:

degree_mapping = {
    'Degree Not Listed' : 0,
    'Other' : 0,
    'Bachelor Degree' : 1,
    'Masters Degree' : 2,
    'PhD' : 3
}

stage_mapping = {
    'Applied' : 0,
    'Assessment' : 1,
    'First Round Interview' : 2,
    'Final Round' : 3,
    'Offer' : 4
}

full_time_traders_appl['Last Degree'] = full_time_traders_appl['Last Degree'].apply(lambda degree : degree_mapping[degree])
full_time_traders_appl_labels = full_time_traders_appl_labels.apply(lambda stage : stage_mapping[stage])


# In[10]:

full_time_traders_appl = full_time_traders_appl.drop('Job_Posting_Title', axis=1)
full_time_traders_appl


# In[11]:

# Feature Correlation Analysis

import seaborn as sns

corr = full_time_traders_appl.corr()
corr.to_csv('Feature_Correlations.csv')

corr_file = open('Feature_Correlation_Analysis.txt', 'w+')
for i in range(0, corr.shape[0]):
    row = corr.iloc[i]
    directly_correlated_features = 'Highest correlations: %s\n' % str(row.nlargest(10))
    corr_file.write(directly_correlated_features)
    inversely_correlated_features = 'Smallest correlations: %s\n' % str(row.nsmallest(10))
    corr_file.write(inversely_correlated_features)
    corr_file.write('\n\n')

plt.figure(figsize=(50,50))
heatmap = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.savefig("Feature_Correlation_Heatmap.png")
plt.show()


# In[18]:

# Feature-Label Correlation Analysis

full_time_traders_appl_with_labels = full_time_traders_appl.copy()
full_time_traders_appl_with_labels['Last Stage'] = full_time_traders_appl_labels

corr = full_time_traders_appl_with_labels.corr()
class_corr = corr['Last Stage'][:]
print('Inverse correlations: ', class_corr.nsmallest(30))
print('Direct correlations: ', class_corr.nlargest(30))


# ### Important Considerations when Choosing ML Algo
# 1. Dimensionality of data (after cleaning and encoding)
# 2. Number of training examples
# 3. Presence/absence of outliers

# ### Multi-class Classification Algos
# 1. SVM (works well in datasets with high # features relative to training examples)
# 2. Neural networks (requires a LOT of training examples)
# 3. kNN (works well in low-dimensional space with normalized/standardized values)
# 4. Random forests
# 5. GBM: Adaboost, XGBoost
# 6. Linear, logistic, ordinal, ridge regression
# 7. Naive Bayes

# ### To improve classification model:
# 1. Remove some features (i.e. remove irrelevant features with little variance or weight, build 2 regression models: one for full-time applicants, one for intern applicants, remove rows with Disposition Reason as 'Application Received too Late')
# 2. Better encoding
# 3. Feature engineering/selection to remove collinearity: 
# https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
# Coursera Course: https://www.coursera.org/learn/competitive-data-science/home/welcome
# 4. Try different regression models and decision trees (RandomForestRegressor, XGBoost, Lasso)
# 5. Tune regularization parameters
# 6. Rerun cleaning data script with modifications (further cleans University and GPA feature vals)
# 7. Run regression with k-fold cross validation
# 
# Questions:
# 1. Should I standardize/normalize data after encoding so I don't give the encoded (categorical) features too much weight compared to the continuous features?
# 2. What kind of baseline algo should I use?
# 3. Should I just completely remove features of very low importance (according to the rf model)?

# In[13]:

# Method to plot confusion matrix of model's predictions

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(title, y_pred, y_true, normalize):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    classes = ['Applied', 'Assessment', 'First Round Interview', 'Final Round', 'Offer']
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax


# In[14]:

# ML Model #1: Random Forest

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

import matplotlib.backends.backend_pdf
cm_pdf = matplotlib.backends.backend_pdf.PdfPages("RF_CMs.pdf")

rf = RandomForestClassifier(max_depth=25, n_estimators=125, max_features=200, min_samples_leaf=1)

# 1) Use k-fold cross validation (k = 10)
k = 10
avg_accuracy = 0.0
cm_fig = plt.figure()
for i in range(0, k):
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        full_time_traders_appl, 
        full_time_traders_appl_labels, 
        stratify=full_time_traders_appl_labels, 
        test_size=0.2
    )
    
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    model_accuracy = accuracy_score(y_pred_rf, y_test)
    avg_accuracy += model_accuracy
    print('Accuracy: ', model_accuracy)
    
    # 2) Out of Fold Validation (OOF) - *usually* more representative of test accuracy than k-fold cross validation
    '''
    oof_preds_log = cross_val_predict(rf, X_train, y_train.values, cv=10, n_jobs=1, method="predict")                              
    # Calculate RMSLE (RMSE of Log(1+y))
    cv_rmsle = np.sqrt(mean_squared_error(np.log1p(y_train.values), oof_preds_log))
    print("OOF RMSLE Score: {:.4f}\n".format(cv_rmsle))
    '''
    
    fig, ax = plot_confusion_matrix('Random Forest Model Confusion Matrix', y_pred_rf, y_test, True)
    cm_pdf.savefig(fig)
    
cm_pdf.close()
print('Average accuracy: ', avg_accuracy / k)


# In[15]:

feature_importances = pd.DataFrame(rf.feature_importances_, 
                                   index = full_time_traders_appl.columns, 
                                   columns=['importance'])
feature_importances = feature_importances.sort_values('importance', ascending=False)
print(feature_importances)
plt.rcParams['figure.figsize'] = [20,20]
feature_importances[0:50].plot(kind='bar')
plt.show()


# In[16]:

# ML Model #2: XGBoost
import xgboost as xgb

cm_pdf = matplotlib.backends.backend_pdf.PdfPages("GBT_CMs.pdf")

# 1) Use k-fold cross validation (k = 10)
k = 10
avg_accuracy = 0.0
cm_fig = plt.figure()
for i in range(0, k):
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        full_time_traders_appl, 
        full_time_traders_appl_labels, 
        stratify=full_time_traders_appl_labels, 
        test_size=0.2
    )
    
    # Train XGBoost model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {'max_depth': 10, 'eta': .4, 'silent': 1, 'objective': 'multi:softprob', 'num_class': 5}
    num_rounds = 100
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_rounds, evallist)
    
    # Get predictions
    y_probs = bst.predict(dtest)
    sorted_y_probs_idx = [np.argsort(probs, kind='mergesort')[::-1] for probs in y_probs]
    y_diff = []
    for i in range(0, y_probs.shape[0]):
        probs = y_probs[i]
        sorted_y_prob_idx = sorted_y_probs_idx[i]
        y_diff.append(probs[sorted_y_prob_idx[0]] - probs[sorted_y_prob_idx[1]])
    y_pred = [probs_idx[0] for probs_idx in sorted_y_probs_idx]
    model_accuracy = accuracy_score(y_pred, y_test)
    print('Accuracy: ', model_accuracy)
    avg_accuracy += model_accuracy
    
    print('Prob diffs: ', np.sort(y_diff))
    diff_idx = np.where(y_pred != y_test)[0]
    y_diff = np.array(y_diff)[diff_idx]
    print('Misclassified prob diffs: ', np.sort(y_diff))
    
    fig, ax = plot_confusion_matrix('XGBoost Confusion Matrix', y_pred, y_test, True)
    cm_pdf.savefig(fig)
    
cm_pdf.close()
print('Average accuracy: ', avg_accuracy / k)


# In[17]:

plt.figure(figsize=(50,50))
xgb.plot_importance(bst)
plt.show()


# #### TODO 6: Try encoding the majors differently (don't treat double majors as separte applicants; should have 1 for TWO majors instead in one-hot encoding!)
# #### TODO 7: Identify where y_pred != y_test and get index of X_test. Look at X_test data point and see if there are any interesting trends or patterns by hand so you can put in some manual rules. May solve the issue of model classifying as first-round interview instead of final round/offer.
# #### TODO 9: Try removing some features (like Visa Required or Sponsorship) and see how model performs.

# In[173]:

# ML Model #3: SVM

from sklearn import svm
from sklearn import grid_search

cm_pdf = matplotlib.backends.backend_pdf.PdfPages("GBT_CMs.pdf")

k = 10
avg_accuracy = 0.0
cm_fig = plt.figure()
for i in range(0, k):
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        full_time_traders_appl, 
        full_time_traders_appl_labels, 
        stratify=full_time_traders_appl_labels, 
        test_size=0.2
    )
    
    # Perform gridsearch to find ideal hyperparameters
    '''
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = grid_search.GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=2)
    grid_search.fit(X_train, y_train)
    grid_search.best_params_
    print(grid_search.best_params_)
    '''
    
    clf = svm.SVC(gamma=.05, C=10, decision_function_shape='ovr', degree=5)
    clf.fit(X_train.values, y_train.values)
    
    y_pred_svm = clf.predict(X_test)
    model_accuracy = accuracy_score(y_pred_svm, y_test)
    avg_accuracy += model_accuracy
    print('Accuracy: ', model_accuracy)
    
    fig, ax = plot_confusion_matrix('SVM Confusion Matrix', y_pred, y_test, True)
    cm_pdf.savefig(fig)
    
cm_pdf.close()
print('Average accuracy: ', avg_accuracy / k)


# In[ ]:



