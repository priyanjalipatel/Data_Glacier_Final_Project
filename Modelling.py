#!/usr/bin/env python
# coding: utf-8

# ## Dataset link: https://archive.ics.uci.edu/dataset/222/bank+marketing

# In[1]:


#importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#reading the dataset

df = pd.read_csv('./bank-additional-full.csv', delimiter = ';')
df.head()



# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.info()


# ## There are zero missing values in the dataset.

# In[6]:


df['duration']


# In[7]:


print(df.nunique().sort_values(ascending=False))


# In[8]:


#dropping duplicates 

df.drop_duplicates()
#df.drop_duplicates(keep = 'first', inplace = True)
print(df.duplicated().sum())
df= df.drop_duplicates()

print(df.shape)



# In[9]:


df.describe()


# In[10]:


df.drop('duration', axis=1)


# In[11]:


df.columns


# Duration : Length of the last call (in seconds). Important: This strongly influences whether the client subscribed, but since it's only known after the call, it shouldn't be used in a real predictive model.

# In[12]:


##Develop model with Duration and without duration feature and report the performance of the model.
## We exclude duration columns. For building a realistic predictive model, 
df.drop('duration', axis = 1, inplace = True)


# In[13]:


#https://www.geeksforgeeks.org/data-science/detect-and-remove-the-outliers-using-python/


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt


#outlier detection
cols = ['age', 'campaign', 'pdays',
       'previous', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed']

       
plt.figure(figsize=(10,15))

for i, col in enumerate(cols):
    plt.subplot(4,3,i+1)
    df.boxplot(col)
    plt.grid()
    plt.tight_layout()
plt.savefig("./plots/outlier.png")


# | Row       | Meaning                                                                                                |
# | --------- | ------------------------------------------------------------------------------------------------------ |
# | **count** | Number of **non-null (non-missing)** values in each column. Here, all have 41,176 values, so no nulls. |
# | **mean**  | The **average** value of the column. Can be distorted by outliers.                                     |
# | **std**   | The **standard deviation**, measuring how spread out the values are. Larger std = more variability.    |
# | **min**   | The **minimum** value observed in the column.                                                          |
# | **25%**   | The **first quartile (Q1)** — 25% of the data falls **below** this value.                              |
# | **50%**   | The **median (Q2)** — 50% of the data falls **below** this value.                                      |
# | **75%**   | The **third quartile (Q3)** — 75% of the data falls **below** this value.                              |
# | **max**   | The **maximum** value observed — potential outlier if very far from Q3.                                |
# 

# In[15]:


#checking statistics of outlier features

df[['age', 'campaign', 'pdays','previous','cons.conf.idx']].describe()



# In[16]:


df['pdays'].unique


# In[17]:


#contact , pdays, poutcome : missing values


# In[18]:


#https://www.geeksforgeeks.org/machine-learning/ml-handling-missing-values/


# ### Ranges from 17 to 98, with a mean of ~40 years.
# 
# ### Looks normally distributed; some clients are quite old, but values seem realistic.

# In[19]:


np.max(df['age'])


# In[20]:


#The value which is outside the whisker
print(df['campaign'].quantile(0.95))


# ### Outlier removal using median values
# ### 50%	The median (Q2) — 50% of the data falls below this value. The median of df['campaign'] = 2.000

# In[21]:


#replacing the values which are greater than the 95th percentile with the median value
median_val = df['campaign'].median()
threshold1 = df['campaign'].quantile(0.95)
df['campaign_median_capped'] = np.where(df['campaign'] > threshold1, median_val, df['campaign'])
df[['campaign', 'campaign_median_capped']].describe()


# ### Outlier removal using mean values
# ### The mean of df['campaign'] = 2.567879

# In[22]:


#replacing the values which are greater than the 95th percentile with the median value
mean_val = df['campaign'].mean()
threshold2 = df['campaign'].quantile(0.95)
df['campaign_mean_capped'] = np.where(df['campaign'] > threshold2, mean_val, df['campaign'])
df[['campaign', 'campaign_mean_capped']].describe()


# In[23]:


cols = ['campaign','campaign_median_capped','campaign_mean_capped']

plt.figure(figsize=(10,15))

for i, col in enumerate(cols):
    plt.subplot(4,3,i+1)
    df.boxplot(col)
    plt.grid()
    plt.tight_layout()



# ### Replace campaign Outliers Within Each Job Group (Using IQR)

# In[24]:


def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series.clip(lower, upper)

df['campaign_IQR'] = df.groupby('job')['campaign'].transform(cap_outliers)
df[['campaign','campaign_median_capped','campaign_IQR']].describe()


# Outlier handling is often more robust when done within context, like by job, education, or region.
# 
# A 98-year-old student may be an outlier, but an 80-year-old retired person may not.

# ### Cap campaign using 95th percentile per job group
# 
# This:
# 
#     Calculates the 95th percentile of age for each job group.
# 
#     Caps all values above that threshold within the group.

# In[25]:


def cap_age(series):
    cap = series.quantile(0.95)
    return series.apply(lambda x: cap if x > cap else x)

df['campaign_95th_percentile'] = df.groupby('job')['campaign'].transform(cap_age)
df[['campaign','campaign_IQR', 'campaign_95th_percentile']].describe()


# In[26]:


cols = ['campaign','campaign_IQR','campaign_95th_percentile']

plt.figure(figsize=(10,15))

for i, col in enumerate(cols):
    plt.subplot(4,3,i+1)
    df.boxplot(col)
    plt.grid()
    plt.tight_layout()


# In[27]:


cols = ['age','campaign','pdays','previous','emp.var.rate']

plt.figure(figsize=(10, 15))

for i, col in enumerate(cols):
    plt.subplot(4, 3, i + 1)
    sns.histplot(df[col], kde=True)  # use df[col] to access column data
    plt.title(f'Distribution of {col}')
    plt.grid(True)

plt.tight_layout()
plt.show()


#sns.histplot(df['campaign'])
#sns.histplot(df['campaign'], bins=20)
#df['campaign'].describe()


# ### Final Recommendation:
# #### Use IQR or 95th percentile to identify outliers
# #### Then use median to replace them for robustness

# In[28]:


#replacing the values which are greater than the 95th percentile with the median value
median_previous = df['previous'].median()
threshold3 = df['previous'].quantile(0.95)
df['previous_median_capped'] = np.where(df['previous'] > threshold3, median_previous, df['previous'])
df[['previous', 'previous_median_capped']].describe()


# In[29]:


#replacing the values which are greater than the 95th percentile with the median value
median_emp_var_rate = df['cons.conf.idx'].median()
threshold4 = df['cons.conf.idx'].quantile(0.95)
df['cons.conf.idx_median_capped'] = np.where(df['cons.conf.idx'] > threshold4, median_emp_var_rate , df['cons.conf.idx'])
df[['cons.conf.idx', 'cons.conf.idx_median_capped']].describe()


# In[30]:


def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series.clip(lower, upper)

df['cons.conf.idx_IQR'] = df.groupby('job')['cons.conf.idx'].transform(cap_outliers)
df[['cons.conf.idx','cons.conf.idx_IQR']].describe()


# In[31]:


# Drop multiple columns in-place
df.drop(columns=['cons.conf.idx','campaign','previous','campaign_mean_capped', 'campaign_95th_percentile','campaign_IQR','cons.conf.idx_IQR'], inplace=True)


# Save to CSV
#df.to_csv('final_cleaned_bank-additional-full.csv', index=False)


# In[32]:


# Settings
sns.set(style='whitegrid')

# --- Dataset Overview ---
print("\n--- Dataset Info ---")
df.info()

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Summary Statistics ---")
print(df.describe(include='all'))


#


# In[33]:


# --- Target Variable ---
sns.countplot(x='y', data=df)
plt.title('Target Variable Distribution')
plt.xlabel('Subscription (y)')
plt.ylabel('Count')
plt.savefig('plots/Target_Variable_Distribution.png')

plt.show()


# In[34]:


# --- Numerical Features ---
numeric_cols = ['age', 'campaign_median_capped', 'pdays', 'previous_median_capped',
                'emp.var.rate', 'cons.price.idx', 'cons.conf.idx_median_capped', 'euribor3m', 'nr.employed']

# Histograms
_ = df[numeric_cols].hist(figsize=(14, 10), bins=20)
plt.suptitle('Numerical Features Distribution')
plt.savefig('plots/numerical_distribution.png')

plt.show()



# # Correlation heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.show()
# plt.savefig('plots/correlation_heatmap.png')
# 

# In[35]:


# Encode target as binary (if not already done)
df['y_binary'] = df['y'].map({'yes': 1, 'no': 0})

# Add it to numeric columns for correlation
corr_cols = numeric_cols + ['y_binary']

# Compute correlations
corr = df[corr_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (with target)")

# Rotate axis labels
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0,ha='right')

plt.tight_layout()
plt.savefig('plots/correlation_with_target.png')
plt.show()


# Feature-wise Explanation
# 
#     emp.var.rate (Employment Variation Rate)
# 
#         More employment variation → less likely to subscribe (economic uncertainty?)
# 
#     euribor3m (Euro Interbank Offered Rate - 3 Months)
# 
#         Higher interest rates → less subscriptions (people may avoid locking into term deposits)
# 
#     nr.employed (Number of Employees)
# 
#         High employment → possibly lower urgency to invest or switch products
# 
#     pdays
# 
#         If recently contacted → more likely to subscribe
# 
#         999 = not contacted → less likely → explains negative correlation

# In[36]:


# Boxplots vs Target
target = 'y'
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=target, y=col, data=df)
    plt.title(f"{col} vs {target}")
    plt.tight_layout()
    plt.savefig(f'plots/boxplot_{col}.png')
    plt.show()
    


# In[37]:


# --- Categorical Features ---
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                    'contact', 'month', 'day_of_week', 'poutcome']

# Count plots
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index)
    plt.title(f'{col} Distribution')
    plt.tight_layout()
    plt.savefig(f'plots/count_{col}.png')
    plt.show()



# In[38]:


# Target vs Categorical
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, hue=target, data=df)
    plt.title(f"{col} vs {target}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'plots/target_vs_{col}.png')

    plt.show()
    


# In[39]:


# --- Class Balance ---
class_dist = df[target].value_counts(normalize=True)
print("\n--- Class Distribution ---")
print(class_dist)
class_dist.to_csv("target_distribution.csv")


# In[40]:


# --- Age Group Analysis ---
age_bins = [15, 25, 35, 45, 55, 65, 75, 100]
age_labels = ['<25', '25–35', '35–45', '45–55', '55–65', '65–75', '75+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

group_stats = df.groupby('age_group')['y'].value_counts(normalize=True).unstack().fillna(0)
group_stats['count'] = df.groupby('age_group')['y'].count()
group_stats['conversion_rate'] = group_stats['yes']

# Plot conversion rate
plt.figure(figsize=(8, 5))
sns.barplot(x=group_stats.index, y=group_stats['conversion_rate'], palette='viridis')
plt.title("Conversion Rate by Age Group")
plt.ylabel("Conversion Rate (y = yes)")
plt.xlabel("Age Group")
plt.ylim(0, 1)
plt.savefig("plots/conversion_by_age_group.png")
plt.show()

# Plot count by age group and target
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='age_group', hue='y')
plt.title("Subscription Count by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.savefig("plots/count_by_age_group.png")
plt.show()

group_counts = df.groupby(['age_group', 'y']).size().unstack().fillna(0)
group_counts['count'] = group_counts['yes'] + group_counts['no']
group_counts['conversion_rate'] = group_counts['yes'] / group_counts['count']

conversion_rate: It tells you which age groups are more likely to Customer says yes to subscribing to the bank's term deposit offer. → helps focus marketing efforts where the probability is higher.
# In[41]:


df.info()


# In[42]:


# Drop age group column
df.drop(columns=['age_group'], inplace=True)


# ###  One-hot encoding for categorical variables

# categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
#                     'contact', 'month', 'day_of_week', 'poutcome']

# In[43]:


# One-hot encode with drop_first=True to avoid dummy variable trap
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Check shape
print("Original shape:", df.shape)
print("Encoded shape:", df_encoded.shape)

# Save encoded dataset (optional)
#df_encoded.to_csv("data/bank_marketing_encoded.csv", index=False)


# In[44]:


print(df_encoded.head())


# In[45]:


print(df_encoded['previous_median_capped'])


# In[46]:


df['pdays_contacted'] = df['pdays'].apply(lambda x: 0 if x == 999 else 1)
df.drop(columns=['pdays'], inplace=True)


# In[47]:


df_encoded['y'].replace({'yes':1,'no':0}, inplace=True)


# In[48]:


print(df_encoded['y'])


# In[49]:


df_encoded['campaign_median_capped']


# ## Modelling using Machine Learning

# In[50]:


X = df_encoded.drop(['y'],axis=1)
y = df_encoded.y


# In[51]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[53]:


## Train and test split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =42)


# In[54]:


print("X and Y train shape:", X_train.shape, y_train.shape)
print("X and Y test shape:", X_test.shape, y_test.shape)


# Feature Scaling is the process of bringing all independent features to the same scale.
# It is done during data preprocessing because:
# 
# Different features may have very different ranges or units (e.g., age in years vs. salary in dollars).
# 
# Without scaling, algorithms like Logistic Regression, SVM, or Neural Networks may treat larger values as more important just because of their scale.
# 
# Scaling helps the algorithm converge faster when using optimization methods like gradient descent.
# In short: Feature scaling ensures all features contribute equally by putting them on the same scale (using normalization or standardization).

# ## Step: Scaling (for LR only) Feature Engineering: Scaling, 
# 
# ## Standardize the dataset: The method of scaling is based on the central tendencies and variance of the data. 

# In[55]:


#every feature is calculated with different units to make the algorithm converge faster(to reach global minimum) using gradient decent,
#all the datapoints should be normalized or standardize to the same scale
#Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[56]:


print(X_test)


# In[57]:


from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB 
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# # Step 4: Handle imbalance (SMOTE)
# # -------------------------------
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Step 5: Models
# # -------------------------------
# 
# # Logistic Regression
# log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
# log_reg.fit(X_train_scaled, y_train)
# y_pred_lr = log_reg.predict(X_test_scaled)
# print("Logistic Regression Report:")
# print(classification_report(y_test, y_pred_lr))
# print("ROC-AUC:", roc_auc_score(y_test, log_reg.predict_proba(X_test_scaled)[:,1]))
# 
# # Random Forest
# rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
# rf.fit(X_train, y_train)
# y_pred_rf = rf.predict(X_test)
# print("\nRandom Forest Report:")
# print(classification_report(y_test, y_pred_rf))
# print("ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))
# 
# # XGBoost (with imbalance handling)
# xgb = XGBClassifier(scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
#                     n_estimators=300, random_state=42, use_label_encoder=False, eval_metric="logloss")
# xgb.fit(X_train, y_train)
# y_pred_xgb = xgb.predict(X_test)
# print("\nXGBoost Report:")
# print(classification_report(y_test, y_pred_xgb))
# print("ROC-AUC:", roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1]))

# In[58]:


#creating the objects for the models 
#1. Logistic Regression
logreg = LogisticRegression()

#2. Decision Tree
dt=DecisionTreeClassifier()

#3 Naive Bayes
nb=BernoulliNB()

#4. Random Forest
rf=RandomForestClassifier()

#5. Gradient Boosting
gb=GradientBoostingClassifier()

cv_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'Naive Bayes', 3: 'Random Forest', 4: 'Gradient Boosting'}
cv_models=[logreg,dt,nb,rf,gb]


for i,model in enumerate(cv_models):
    print("{} Test Accuracy: {}".format(cv_dict[i],cross_val_score(model, X_scaled, y, cv=10, scoring ='accuracy').mean()*100))


# In[59]:


# Models
logreg = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier()
nb = BernoulliNB()
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()

cv_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'Naive Bayes', 3: 'Random Forest', 4: 'Gradient Boosting'}
cv_models = [logreg, dt, nb, rf, gb]

# Loop over models
for i, model in enumerate(cv_models):
    name = cv_dict[i]
    print(f"\n{name}")
    print("=" * len(name))

    # Cross-validation scores
    acc = cross_val_score(model, X_scaled, y, cv=10, scoring='accuracy').mean()
    roc = cross_val_score(model, X_scaled, y, cv=10, scoring='roc_auc').mean()
    f1 = cross_val_score(model, X_scaled, y, cv=10, scoring='f1').mean()

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC : {roc:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Predictions for confusion matrix
    y_pred = cross_val_predict(model, X_scaled, y, cv=10)

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['No', 'Yes']))


# Decision Tree and Gradient Boosting showed 100% accuracy, likely due to overfitting or data leakage (e.g., use of duration feature). 
# These models were excluded from final selection to ensure fair evaluation.

# In[60]:


#creating the objects for the models 
#1. Logistic Regression
logreg = LogisticRegression()

#3 Naive Bayes
nb=BernoulliNB()

#4. Random Forest
rf=RandomForestClassifier()


cv_dict = {0: 'Logistic Regression', 1:'Naive Bayes', 2: 'Random Forest'}
cv_models=[logreg,nb,rf]


for i,model in enumerate(cv_models):
    print("{} Test Accuracy: {}".format(cv_dict[i],cross_val_score(model, X_scaled, y, cv=10, scoring ='accuracy').mean()*100))


# C: Controls regularization. Large values (100, 10) mean weaker regularization, smaller values (0.1, 0.01) mean stronger.
# 
# penalty:
# 
# 'l1': forces sparsity (feature selection effect).
# 
# 'l2': shrinks coefficients but keeps all.
# 
# solver: Different optimization algorithms; some only work with certain penalties (e.g. lbfgs doesn’t support l1).
# 
# max_iter: Ensures convergence; Logistic Regression might need more iterations with l1 or large datasets.
# 
# cv=5: Uses 5-fold cross-validation for tuning.
# 
# n_jobs=-1: Uses all CPU cores for faster training.

# Fits Logistic Regression with all combinations from param_grid.
# 
# Prints the best hyperparameters found.
# 
# Evaluates the tuned model on the test set.

# param_grid = {
#     'C': [100, 10, 1.0, 0.1, 0.01],
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear', 'saga'],  # solvers compatible with 'l1'
#     'max_iter': range(80, 120)
# }

# ## LogisticRegression is the best model.

# In[66]:


#Training the model with the best parameters


logreg = LogisticRegression(C=1.0, max_iter=87, penalty='l1', random_state=0, solver='saga')
logreg.fit(X_train, y_train)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)*100))


# In[62]:


confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n",confusion_matrix)
print("Classification Report:\n",classification_report(y_test, y_pred))


# 
# 
#     Confusion matrix results tell us that we have 7187 + 121 Correct predictions and 741+187 incorrect
# 
#     Classification report shows precision as 90% which is the ability of a classification model to identify only the relevant data points,
# that is in this case people who would be subscribing to the term deposit is correctly classified.
# 
# 

# Business Interpretation
# 
# Precision (for Yes): Of all customers we target (predicted Yes), how many actually subscribe?
# 
# Recall (for Yes): Of all real subscribers, how many did we actually find?
# 
# In marketing, recall is often more important → we don’t want to miss potential customers.

# Business interpretation
# 
# Model is very good at identifying No’s (non-subscribers).
# 
# Model is struggling to catch Yes’s (subscribers), which is the main business goal.
# 
# Even though overall accuracy is 90%, the recall of 20% for class 1 means the bank misses 80% of potential customers → not acceptable for marketing use case.

# In[76]:


y_pred = logreg.predict_proba(X_test)[:,1]  
logit_roc_auc = roc_auc_score(y_test, y_pred)
# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[72]:


logreg_bal = LogisticRegression(class_weight='balanced', solver='saga', max_iter=2000, random_state=0)
logreg_bal.fit(X_train, y_train)
y_pred_bal = logreg_bal.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg_bal.score(X_test, y_test)*100))


# In[73]:


from sklearn.metrics import confusion_matrix, classification_report

# Do NOT assign to confusion_matrix variable
cm_bal = confusion_matrix(y_test, y_pred_bal)
print("Confusion Matrix:\n", cm_bal)

print("Classification Report:\n", classification_report(y_test, y_pred_bal))


# ROC curve plots True Positive Rate (Recall) vs. False Positive Rate at different thresholds.
# 
# AUC (Area Under Curve) is a single number summarizing the curve:
# 
# 1.0 → perfect model
# 
# 0.5 → random guessing
# 
# Higher is always better

# In[75]:


# Use probabilities, not labels
y_pred_proba = logreg_bal.predict_proba(X_test)[:,1]  
logreg_bal_roc_auc = roc_auc_score(y_test, y_pred_proba)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logreg_bal_roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

