# Last amended: 22nd May, 2025

# File: airways_deploy.py
# Original file is located at
#    C:\Users\ashok\OneDrive\Documents\airways



# Install: To encode categorical features:
# pip install category_encoders



# Target column: cormack lehane
#        Values: 1,2,3,4




#1.0  Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pickle import load,dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import category_encoders as ce
from sklearn.ensemble import ExtraTreesClassifier



# 2.0 Read data from github repo

df=pd.read_csv('https://raw.githubusercontent.com/harnalashok/classification/refs/heads/main/data/airways_mod_ver4.csv')
df.shape   # (1000, 13)
df.head()  # cormacl lehane is the Target


# 3.0
df.isna().sum()
# We have only two rows where edentulous is null
# Drop those two rows
df = df.dropna()

# 3.1 Change the type of 'edentulous' from float64 to int64
df['edentulous'] = df['edentulous'].astype('int64')
df.dtypes

#4.0 Map 'male' and 'female' to 1 and 0
mappings = {
            'female': 0,
            'male': 1
           }
df['sex'] = df['sex'].map(mappings)

#5.0 Our cat features and num features:
# unique values  2         2             2             4              2
cat_features = ['sex', 'buck teeth', 'edentulous', 'mallampatti', 'mouth opening',
#                         2                           4
                'head and neck movement','subluxation of mandible']

# min-max       0-90   120-200  10-40      10-25                    3-13
num_features = ["age", "height",  "bmi", "sternomental distance", "thyromental distance"]



#6.0 Split dfc into target and predictors
y = df.pop('cormack lehane')
X = df



# 7.0 Label encode cromack lahane

le = LabelEncoder()
y = le.fit_transform(y)




# 8.0 Split into train/test. We may need it at some places
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 stratify = y,
                                                 test_size = 0.20,
                                                 )
# 9.0 Train model
clf = ExtraTreesClassifier(criterion="entropy",
                           n_estimators=400,
                           max_depth=None,
                           min_samples_split=2,
                           warm_start=False,
                           n_jobs=-1
                           )


# 10.0
encoder = ce.CatBoostEncoder(cols=["sex", "edentulous", "mallampatti",
                                   "mouth opening", "head and neck movement"]
                                   )

# 11.0 Transform X_train
encoder.fit(X_train, y_train)
X_train = encoder.transform(X_train)

# 11.1 Transform X_test
X_test = encoder.transform(X_test)

# 11.2 Fit X_train,y_train
clf.fit(X_train, y_train)

# 12.0 Evaluate model
acc = clf.score(X_test, y_test)
roc_auc = roc_auc_score(y_test,
                    clf.predict_proba(X_test),
                    multi_class='ovr', # Ref: https://stackoverflow.com/a/66022487
                    average = None)

prob=clf.predict_proba(X_test)
ll=log_loss(y_test, prob)


# 13.0 PRint results
print ("\n=============")
print("logloss for this fold/split:", ll)
print(f"Calculated roc_auc: {roc_auc}")
print(f"Accuracy is: {acc}")
print ("=============\n")


print(classification_report(y_test,
                            clf.predict(X_test),
                            output_dict = False  # dict format and not tabular format
                            ))


# 14.0 Save models
extra_classifier = clf
pathToSave = "C:\\Users\\ashok\\OneDrive\\Documents\\airways\\"

# 14.1 Dump encoder
with open(pathToSave+"encoder.pkl", "wb") as f:
    dump(encoder, f, protocol=5)

# 14.2 Dump model
with open(pathToSave + "extratrees.pkl", "wb") as f:
    dump(extra_classifier, f, protocol=5)

# 15.0 Reload saved model

with open(pathToSave + "encoder.pkl", "rb") as f:
    encoder = load(f)

with open(pathToSave + "extratrees.pkl", "rb") as f:
    clf = load(f)

###########



"""
Processing:
    
1. Read input data
2. Columns names be:

'age', 'sex', 'height', 'subluxation of mandible',
'head and neck movement', 'buck teeth', 'edentulous', 'bmi',
'sternomental distance', 'thyromental distance', 'mallampatti',
'mouth opening', 'cormack lehane'

# 3. Map 'male' and 'female' to 1 and 0

# 4.
Read clf
Read encoder

X_test = encoder.transform(X_test)
cormack_lehane = clf.predict(X_test)
 

# C. Decode cormack lehane, as:
0=>1
1=>2
2=>3
3=>4

Do not allow nulls here:
    
["mallampatti", "age", "sex", "bmi", "strenomental diatnce", "buck teeth",
 "height", "thromental distance", "edentulous",
 "mouth opening", "head and neck movement"]


"""


