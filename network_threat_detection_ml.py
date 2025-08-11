## DATAPREPROCESSING AND DATACLEANING 
import pandas as pd 
import numpy as np 
import warnings 
warnings.filterwarnings('ignore') 
df=pd.read_csv('cyber1.csv') 
del df['Unnamed: 0'] 
del df['id.orig_p'] 
del df['id.resp_p'] 
del df['proto'] 
del df['service'] 
del df['fwd_iat.min'] 
del df['fwd_iat.max'] 
del df['fwd_iat.tot'] 
del df['fwd_iat.avg'] 
del df['fwd_iat.std'] 
del df['bwd_iat.min'] 
del df['bwd_iat.max'] 
del df['bwd_iat.tot'] 
del df['bwd_iat.avg'] 
del df['bwd_iat.std'] 
del df['flow_iat.min'] 
del df['flow_iat.max'] 
del df['flow_iat.tot'] 
del df['flow_iat.avg'] 
del df['flow_iat.std'] 
deldf['payload_bytes_per_second'] 
del df['fwd_subflow_pkts'] 
del df['bwd_subflow_pkts'] 
del df['fwd_subflow_bytes'] 
del df['bwd_subflow_bytes'] 
del df['fwd_bulk_bytes'] 
del df['bwd_bulk_bytes'] 
del df['active.min'] 
deldf['active.max'] 
del df['active.tot'] 
del df['active.avg'] 
del df['active.std'] 
del df['idle.min'] 
del df['idle.max'] 
del df['idle.tot'] 
del df['idle.avg'] 
del df['idle.std'] 
df.head() 
df.tail()  
df.shape 
df.size 
df.columns 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
var = ['Attack_type'] 
for i in var: 
df[i] = le.fit_transform(df[i]).astype(int) 
df.isnull() 
df = df.dropna() 
df['Attack_type'].unique() 
df.describe() 
df.corr() 
df.info() 
df["Attack_type"].value_counts() 
df.duplicated() 
sum(df.duplicated()) 
df = df.drop_duplicates() 
sum(df.duplicated()) 
## DATAVISUALIZATION AND DATAANALYSIS 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
df=pd.read_csv('cyber1.csv') 
del df['Unnamed: 0'] 
del df['id.orig_p'] 
del df['id.resp_p'] 
del df['proto'] 
del df['service'] 
del df['fwd_iat.min'] 
del df['fwd_iat.max'] 
del df['fwd_iat.tot'] 
del df['fwd_iat.avg'] 
del df['fwd_iat.std'] 
del df['bwd_iat.min'] 
del df['bwd_iat.max'] 
del df['bwd_iat.tot'] 
del df['bwd_iat.avg'] 
del df['bwd_iat.std'] 
del df['flow_iat.min'] 
del df['flow_iat.max'] 
del df['flow_iat.tot'] 
del df['flow_iat.avg'] 
del df['flow_iat.std'] 
deldf['payload_bytes_per_second'] 
del df['fwd_subflow_pkts'] 
del df['bwd_subflow_pkts'] 
del df['fwd_subflow_bytes'] 
del df['bwd_subflow_bytes'] 
del df['fwd_bulk_bytes'] 
del df['bwd_bulk_bytes'] 
del df['active.min'] 
deldf['active.max'] 
del df['active.tot'] 
del df['active.avg'] 
del df['active.std'] 
del df['idle.min'] 
del df['idle.max'] 
del df['idle.tot'] 
del df['idle.avg'] 
del df['idle.std'] 
df.head() 
df.columns 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
var = ['Attack_type'] 
for i in var: 
df[i] = le.fit_transform(df[i]).astype(int) 
plt.figure(figsize=(12,7)) 
sns.countplot(x='Attack_type',data=df) 
plt.figure(figsize=(15,5)) 
plt.subplot(1,2,1) 
plt.hist(df['bwd_data_pkts_tot'],color='red') 
plt.subplot(1,2,2) 
plt.hist(df['flow_pkts_per_sec'],color='blue') 
df.hist(figsize=(15,55),layout=(15,4), color='green') 
plt.show() 
df['flow_pkts_per_sec'].hist(figsize=(10,5),color='yellow') 
sns.lineplot(df['flow_ACK_flag_count'], color='brown') # scatter, plot, triplot, 
stackplot 
sns.violinplot(df['fwd_header_size_max'], color='purple') 
fig, ax = plt.subplots(figsize=(20,15)) 
sns.heatmap(df.corr(),annot = True, fmt='0.2%',cmap = 'autumn',ax=ax) 
def plot(df, variable): 
dataframe_pie = df[variable].value_counts() 
ax = dataframe_pie.plot.pie(figsize=(9,9), autopct='%1.2f%%', fontsize = 10) 
ax.set_title(variable + ' \n', fontsize = 10) 
return np.round(dataframe_pie/df.shape[0]*100,2) 
plot(df, 'Attack_type')  
# BERNOULLINB CLASSIFIER ALGORITHEM 
# Import the necessary libraries. 
import pandas as pd 
import numpy as npy
import matplotlib.pyplot as plt 
import seaborn as sns 
# Avoid unnecessary warnings, (EX: software updates, version mismatch, and so 
on.) 
import warnings 
warnings.filterwarnings('ignore') 
# Load the datasets 
df=pd.read_csv('cyber1.csv') 
del df['Unnamed: 0'] 
del df['id.orig_p'] 
del df['id.resp_p'] 
del df['proto'] 
del df['service'] 
del df['fwd_iat.min'] 
del df['fwd_iat.max'] 
del df['fwd_iat.tot'] 
del df['fwd_iat.avg'] 
del df['fwd_iat.std'] 
del df['bwd_iat.min'] 
del df['bwd_iat.max'] 
del df['bwd_iat.tot'] 
del df['bwd_iat.avg'] 
del df['bwd_iat.std'] 
del df['flow_iat.min'] 
del df['flow_iat.max'] 
del df['flow_iat.tot'] 
del df['flow_iat.avg'] 
del df['flow_iat.std'] 
del df['payload_bytes_per_second'] 
del df['fwd_subflow_pkts'] 
del df['bwd_subflow_pkts'] 
del df['fwd_subflow_bytes'] 
del df['bwd_subflow_bytes'] 
del df['fwd_bulk_bytes'] 
del df['bwd_bulk_bytes'] 
del df['active.min'] 
deldf['active.max'] 
del df['active.tot'] 
del df['active.avg'] 
del df['active.std'] 
del df['idle.min'] 
del df['idle.max'] 
del df['idle.tot'] 
del df['idle.avg'] 
del df['idle.std'] 
# Check the columns of dataset 
df.columns 
df.info() 
# Check the top5 values 
df.head() 
# Remove the null value 
df=df.dropna() 
df['Attack_type'].value_counts() 
# Transform the columns value(ex: int to str, str to int) for classification purpose. 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
var = ['Attack_type'] 
for i in var: 
df[i] = le.fit_transform(df[i]).astype(int) 
df['Attack_type'].value_counts()
# Check the top5 values 
df.head() 
# Split the datasets into depended and independed variable 
# X is independend variable (Input features) 
x1 = df.drop(labels='Attack_type', axis=1) 
# Y is dependend variable (Target variable) 
y1 = df.loc[:,'Attack_type'] 
# This process execute to balanced the datasets features. 
import imblearn 
from imblearn.over_sampling import RandomOverSampler 
from collections import Counter 
ros =RandomOverSampler(random_state=42) 
x,y=ros.fit_resample(x1,y1)
print("OUR DATASET COUNT 
: ", Counter(y1)) 
print("OVER SAMPLING DATA COUNT : ", Counter(y)) 
# Split the datasets into two parts like trainng and testing variable. 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, 
random_state=42, stratify=y) 
print("NUMBER OF TRAIN DATASET : ", len(x_train)) 
print("NUMBER OF TEST DATASET : ", len(x_test)) 
print("TOTAL NUMBER OF DATASET : ", len(x_train)+len(x_test)) 
print("NUMBER OF TRAIN DATASET : ", len(y_train)) 
print("NUMBER OF TEST DATASET : ", len(y_test)) 
print("TOTAL NUMBER OF DATASET : ", len(y_train)+len(y_test)) 
# Implement Catboost classifier algorithm learning patterns 
from sklearn.naive_bayes import BernoulliNB 
RFR = BernoulliNB() 
RFR.fit(x_train, y_train) 
# Predict is the test function for this algorithm 
predicted = RFR.predict(x_test) 
# Check classification report for this algorithm 
from sklearn.metrics import classification_report 
cr = classification_report(y_test,predicted) 
print('THE CLASSIFICATION REPORT OF BERNOULLIINB 
CLASSIFIER:\n\n',cr) 
# Check the confusion matrix for this algorithms. 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,predicted) 
print('THE CONFUSION MATRIX SCORE OF BERNOULLIINB 
CLASSIFIER:\n\n\n',cm) 
# Check the accuracy score of this algorithms. 
from sklearn.metrics import accuracy_score 
a = accuracy_score(y_test,predicted) 
print("THE ACCURACY SCORE OF BERNOULLIINB CLASSIFIER IS 
:",a*100) 
# Check the hamming loss of this algorithm. 
from sklearn.metrics import hamming_loss 
hl = hamming_loss(y_test,predicted) 
print("THE HAMMING LOSS OF BERNOULLIINB CLASSIFIER IS :",hl*100) 
# Plot a Confusion matrix for this algorithms. 
def plot_confusion_matrix(cm, title='THE CONFUSION MATRIX SCORE OF 
BERNOULLIINB CLASSIFIER\n\n', cmap=plt.cm.cool): 
plt.imshow(cm, interpolation='nearest', cmap=cmap) 
plt.title(title) 
plt.colorbar() 
cm1=confusion_matrix(y_test, predicted) 
print('THE CONFUSION MATRIX SCORE OF BERNOULLIINB 
CLASSIFIER:\n\n') 
print(cm) 
plot_confusion_matrix(cm) 
# Plot the worm plot for this model. 
import matplotlib.pyplot as plt 
df2 = pd.DataFrame() 
df2["y_test"] = y_test 
df2["predicted"] = predicted 
df2.reset_index(inplace=True) 
plt.figure(figsize=(20, 5)) 
plt.plot(df2["predicted"][:100], marker='x', linestyle='dashed', color='red') 
plt.plot(df2["y_test"][:100], marker='o', linestyle='dashed', color=)
plt.show()  
# ADABOOST CLASSIFIER ALGORITHEM 
# Import the necessary libraries. 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
# Avoid unnecessary warnings, (EX: software updates, version mismatch, and so 
on.) 
import warnings 
warnings.filterwarnings('ignore') 
# Load the datasets 
df=pd.read_csv('cyber1.csv') 
del df['Unnamed: 0'] 
del df['id.orig_p'] 
del df['id.resp_p'] 
del df['proto'] 
del df['service'] 
del df['fwd_iat.min'] 
del df['fwd_iat.max'] 
del df['fwd_iat.tot'] 
del df['fwd_iat.avg'] 
del df['fwd_iat.std'] 
del df['bwd_iat.min'] 
del df['bwd_iat.max'] 
del df['bwd_iat.tot'] 
del df['bwd_iat.avg'] 
del df['bwd_iat.std'] 
del df['flow_iat.min'] 
del df['flow_iat.max'] 
del df['flow_iat.tot'] 
del df['flow_iat.avg'] 
del df['flow_iat.std'] 
deldf['payload_bytes_per_second'] 
del df['fwd_subflow_pkts'] 
del df['bwd_subflow_pkts'] 
del df['fwd_subflow_bytes'] 
del df['bwd_subflow_bytes'] 
del df['fwd_bulk_bytes'] 
del df['bwd_bulk_bytes'] 
del df['active.min'] 
deldf['active.max'] 
del df['active.tot'] 
del df['active.avg'] 
del df['active.std'] 
del df['idle.min'] 
del df['idle.max'] 
del df['idle.tot'] 
del df['idle.avg'] 
del df['idle.std'] 
# Check the columns of dataset 
df.columns 
df.info() 
# Check the top5 values 
df.head() 
# Remove the null value 
df=df.dropna() 
df['Attack_type'].value_counts() 
# Transform the columns value(ex: int to str, str to int) for classification purpose. 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
var = ['Attack_type'] 
for i in var: 
df[i] = le.fit_transform(df[i]).astype(int) 
71 
df['Attack_type'].value_counts() 
# Check the top5 values 
df.head() 
# Split the datasets into depended and independed variable 
# X is independend variable (Input features) 
x1 = df.drop(labels='Attack_type', axis=1) 
# Y is dependend variable (Target variable) 
y1 = df.loc[:,'Attack_type'] 
# This process execute to balanced the datasets features. 
import imblearn 
from imblearn.over_sampling import RandomOverSampler 
from collections import Counter 
ros =RandomOverSampler(random_state=42) 
x,y=ros.fit_resample(x1,y1) 
print("OUR DATASET COUNT 
: ", Counter(y1)) 
print("OVER SAMPLING DATA COUNT : ", Counter(y)) 
# Split the datasets into two parts like trainng and testing variable. 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, 
random_state=42, stratify=y) 
print("NUMBER OF TRAIN DATASET : ", len(x_train)) 
print("NUMBER OF TEST DATASET : ", len(x_test)) 
print("TOTAL NUMBER OF DATASET : ", len(x_train)+len(x_test)) 
print("NUMBER OF TRAIN DATASET : ", len(y_train)) 
print("NUMBER OF TEST DATASET : ", len(y_test)) 
print("TOTAL NUMBER OF DATASET : ", len(y_train)+len(y_test)) 
# Implement AdaBoost classifier algorithm learning patterns 
from sklearn.ensemble import AdaBoostClassifier 
RFR = AdaBoostClassifier() 
RFR.fit(x_train, y_train) 
# Predict is the test function for this algorithm 
predicted = RFR.predict(x_test) 
# Check classification report for this algorithm 
from sklearn.metrics import classification_report 
cr = classification_report(y_test,predicted) 
print('THE CLASSIFICATION REPORT OF ADABOOST CLASSIFIER:\n\n',cr) 
# Check the confusion matrix for this algorithms. 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,predicted) 
print('THE CONFUSION MATRIX SCORE OF ADABOOST 
CLASSIFIER:\n\n\n',cm) 
# Check the accuracy score of this algorithms. 
from sklearn.metrics import accuracy_score 
a = accuracy_score(y_test,predicted) 
print("THE ACCURACY SCORE OF ADABOOST CLASSIFIER IS :",a*100) 
# Check the hamming loss of this algorithm. 
from sklearn.metrics import hamming_loss 
hl = hamming_loss(y_test,predicted) 
print("THE HAMMING LOSS OF ADABOOST CLASSIFIER IS :",hl*100) 
# Plot a Confusion matrix for this algorithms. 
def plot_confusion_matrix(cm, title='THE CONFUSION MATRIX SCORE OF 
ADABOOST CLASSIFIER\n\n', cmap=plt.cm.cool): 
plt.imshow(cm, interpolation='nearest', cmap=cmap) 
plt.title(title) 
plt.colorbar() 
cm1=confusion_matrix(y_test, predicted) 
print('THE CONFUSION MATRIX SCORE OF ADABOOST 
CLASSIFIER:\n\n') 
print(cm) 
plot_confusion_matrix(cm) 
# Plot the worm plot for this model. 
import matplotlib.pyplot as plt 
df2 = pd.DataFrame() 
df2["y_test"] = y_test 
df2["predicted"] = predicted 
df2.reset_index(inplace=True) 
plt.figure(figsize=(20, 5)) 
plt.plot(df2["predicted"][:100], marker='x', linestyle='dashed', color='red') 
plt.plot(df2["y_test"][:100], marker='o', linestyle='dashed', color='green') 
plt.show() 
# RANDOMFOREST CLASSIFIER ALGORITHEM 
# Import the necessary libraries. 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
# Avoid unnecessary warnings, (EX: software updates, version mismatch, and so 
on.) 
import warnings 
warnings.filterwarnings('ignore') 
# Load the datasets 
df=pd.read_csv('cyber1.csv') 
del df['Unnamed: 0'] 
del df['id.orig_p'] 
del df['id.resp_p'] 
del df['proto'] 
del df['service'] 
del df['fwd_iat.min'] 
del df['fwd_iat.max'] 
del df['fwd_iat.tot'] 
del df['fwd_iat.avg'] 
del df['fwd_iat.std'] 
del df['bwd_iat.min'] 
del df['bwd_iat.max'] 
del df['bwd_iat.tot'] 
del df['bwd_iat.avg'] 
del df['bwd_iat.std'] 
del df['flow_iat.min'] 
del df['flow_iat.max'] 
del df['flow_iat.tot'] 
del df['flow_iat.avg'] 
del df['flow_iat.std'] 
del df['payload_bytes_per_second'] 
del df['fwd_subflow_pkts'] 
del df['bwd_subflow_pkts'] 
del df['fwd_subflow_bytes'] 
del df['bwd_subflow_bytes'] 
del df['fwd_bulk_bytes'] 
del df['bwd_bulk_bytes'] 
del df['active.min'] 
deldf['active.max'] 
del df['active.tot'] 
del df['active.avg'] 
del df['active.std'] 
del df['idle.min'] 
del df['idle.max'] 
del df['idle.tot'] 
del df['idle.avg'] 
del df['idle.std'] 
# Check the columns of dataset 
df.columns 
df.info() 
# Check the top5 values 
df.head() 
# Remove the null value 
df=df.dropna() 
df['Attack_type'].value_counts() 
# Transform the columns value(ex: int to str, str to int) for classification purpose. 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
var = ['Attack_type'] 
for i in var: 
df[i] = le.fit_transform(df[i]).astype(int) 
df['Attack_type'].value_counts() 
# Check the top5 values 
df.head() 
# Split the datasets into depended and independed variable 
# X is independend variable (Input features) 
x1 = df.drop(labels='Attack_type', axis=1) 
# Y is dependend variable (Target variable) 
y1 = df.loc[:,'Attack_type'] 
# This process execute to balanced the datasets features. 
import imblearn 
from imblearn.over_sampling import RandomOverSampler 
from collections import Counter 
ros =RandomOverSampler(random_state=42) 
x,y=ros.fit_resample(x1,y1) 
print("OUR DATASET COUNT 
: ", Counter(y1)) 
print("OVER SAMPLING DATA COUNT : ", Counter(y)) 
# Split the datasets into two parts like trainng and testing variable. 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, 
random_state=42, stratify=y) 
print("NUMBER OF TRAIN DATASET : ", len(x_train)) 
print("NUMBER OF TEST DATASET : ", len(x_test)) 
print("TOTAL NUMBER OF DATASET : ", len(x_train)+len(x_test)) 
print("NUMBER OF TRAIN DATASET : ", len(y_train)) 
print("NUMBER OF TEST DATASET  : ", len(y_test)) 
print("TOTAL NUMBER OF DATASET : ", len(y_train)+len(y_test)) 
# Implement Catboost classifier algorithm learning patterns 
from sklearn.ensemble import RandomForestClassifier 
RFR = RandomForestClassifier() 
RFR.fit(x_train, y_train) 
# Predict is the test function for this algorithm 
predicted = RFR.predict(x_test) 
# Check classification report for this algorithm 
from sklearn.metrics import classification_report 
cr = classification_report(y_test,predicted) 
print('THE CLASSIFICATION REPORT OF RANDOMFOREST 
CLASSIFIER:\n\n',cr) 
# Check the confusion matrix for this algorithms. 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,predicted) 
print('THE CONFUSION MATRIX SCORE OF RANDOMFOREST 
CLASSIFIER:\n\n\n',cm) 
# Check the cross value score of this algorithm. 
from sklearn.model_selection import cross_val_score 
accuracy = cross_val_score(RFR, x, y, scoring='accuracy') 
print('THE CROSS VALIDATION TEST RESULT OF ACCURACY :\n\n\n', 
accuracy*100) 
# Check the accuracy score of this algorithms. 
from sklearn.metrics import accuracy_score 
a = accuracy_score(y_test,predicted) 
print("THE ACCURACY SCORE OF RANDOMFOREST CLASSIFIER IS 
:",a*100) 
# Check the hamming loss of this algorithm. 
from sklearn.metrics import hamming_loss 
hl = hamming_loss(y_test,predicted) 
print("THE HAMMING LOSS OF RANDOMFOREST CLASSIFIER IS 
:",hl*100) 
# Plot a Confusion matrix for this algorithms. 
def plot_confusion_matrix(cm, title='THE CONFUSION MATRIX SCORE OF 
RANDOMFOREST CLASSIFIER\n\n', cmap=plt.cm.cool): 
plt.imshow(cm, interpolation='nearest', cmap=cmap) 
plt.title(title) 
plt.colorbar() 
cm1=confusion_matrix(y_test, predicted) 
print('THE CONFUSION MATRIX SCORE OF RANDOMFOREST 
CLASSIFIER:\n\n') 
print(cm) 
plot_confusion_matrix(cm) 
# Plot the worm plot for this model. 
import matplotlib.pyplot as plt 
df2 = pd.DataFrame() 
df2["y_test"] = y_test 
df2["predicted"] = predicted 
df2.reset_index(inplace=True) 
plt.figure(figsize=(20, 5)) 
plt.plot(df2["predicted"][:100], marker='x', linestyle='dashed', color='red') 
plt.plot(df2["y_test"][:100], marker='o', linestyle='dashed', color='green') 
plt.show() 






