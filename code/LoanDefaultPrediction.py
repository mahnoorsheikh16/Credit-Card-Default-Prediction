#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from scipy import stats
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import hiplot as hip
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from imblearn.over_sampling import SMOTE
import json
from sklearn.model_selection import train_test_split


# In[2]:


#import data
data = pd.read_csv(r"https://raw.githubusercontent.com/mahnoorsheikh16/Credit-Card-Default-Prediction/refs/heads/main/datasets/UCI_Credit_Card.csv")
data_macro = pd.read_excel(r"https://raw.githubusercontent.com/mahnoorsheikh16/Credit-Card-Default-Prediction/refs/heads/main/datasets/Macroeconomic%20indicators.xlsx")
data_income = pd.read_excel(r"https://raw.githubusercontent.com/mahnoorsheikh16/Credit-Card-Default-Prediction/refs/heads/main/datasets/Income%20level.xlsx")
print(data.head())
print(data.shape)
print(data_macro)
print(data_macro.shape)
print(data_income)
print(data_income.shape)


# #### Data Cleaning

# In[3]:


### Variable identification and classification

#ID is a unique variable so drop it
data =  data.drop(columns = ['ID'])

#rename columns
data.rename(columns={'default.payment.next.month': 'Default'}, inplace=True)
data.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)


# Target (dependent) variable: Default
# 
# Explanatory (independent) variable: All other variables for now. Relationships will be further explored during EDA.

# In[4]:


#classify variables in credit data
data.info()


# 14 quantitative variables with discrete data (integers): LIMIT_BAL, AGE, BILL_AMT1-6, PAY_AMT1-6 
# 
# 10 categorical variables: EDUCATION & MARRIAGE are nominal. SEX & Default are binary. PAY_1-6 are ordinal.

# In[5]:


#classify variables in macro data
data_macro.info()


# 1 categorical variable: Month
# 
# 2 quantitative variables with continuous data: CPI, Unemployment Rate

# In[6]:


#classify variables in income data
data_income.info()


# 1 categorical variable: Demographic
# 
# 1 quantitative variable with continuous data: Avg Income level

# In[7]:


### Data Quality Assessment

#check unique values of categorical variables for inconsistencies and compare with data description to verify data accuracy
data['SEX'].value_counts() #all labels match


# In[8]:


data['EDUCATION'].value_counts() #all labels match


# In[9]:


data['MARRIAGE'].value_counts() #all labels match


# In[10]:


data['PAY_1'].value_counts() #labels don't match


# -2 and 0 instances are undocumented but make up a significant chunk of the data so cannot be treated as unknowns
# 
# Upon inspection of the order and row data, we can infer -2=no payment due and 0=payment delay for <1 month (however we cannot be sure) 

# In[11]:


data['Default'].value_counts() #all labels match


# In[12]:


### Basic Descriptive Statistics

data.describe()


# In[4]:


#check for duplicate values
data.duplicated().sum() #35
data.drop_duplicates(inplace=True)


# In[5]:


##10 categorical variables: EDUCATION & MARRIAGE are nominal. SEX & Default are binary. PAY_1-6 are ordinal.
#Encode categorical variables for imputation
categorical = data[['MARRIAGE', 'SEX', 'Default','EDUCATION']]
le = LabelEncoder() #first convert columns to numerical labels
data['SEX_label'] = le.fit_transform(data['SEX'])
data['EDUCATION_label'] = le.fit_transform(data['EDUCATION'])
data['MARRIAGE_label'] = le.fit_transform(data['MARRIAGE'])
data['Default_label'] = le.fit_transform(data['Default'])

data['MARRIAGE_label'] = data['MARRIAGE_label'].replace({3: np.nan}) #introduce NaNs again
data['EDUCATION_label'] = data['EDUCATION_label'].replace({4: np.nan})


# In[15]:


data.info() #recheck data types for consistecy


# In[45]:


### Missing data analysis

total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()


# In[47]:


#visualize missing data
plt.figure(figsize=(12, 7))
sns.heatmap(data.isna().transpose(),cmap='magma')
plt.show()

#create correlation matrix for missing rows
data_dup = data.copy()
data_dup['EDUCATION_missing'] = data['EDUCATION'].isna().astype(int)
data_dup['MARRIAGE_missing'] = data['MARRIAGE'].isna().astype(int)
numeric_cols = data_dup.select_dtypes(include=[np.number])
corr_matrix = numeric_cols.corr()
corr = corr_matrix[['EDUCATION_missing', 'MARRIAGE_missing']]
fig = px.imshow(corr.T, 
                text_auto=True, 
                color_continuous_scale='RdBu_r', 
                aspect='auto',
                title="Correlation Heatmap for Missing Values")
fig.show()
IDA_missing_corr = fig.to_json()
with open("IDA_missing_corr_plot.json", "w") as f:
    f.write(IDA_missing_corr)


# In[41]:


#use pairplot to get additional insights into missingness
data["was_NaN"] = False
data.loc[data["EDUCATION"].isnull() == True, "was_NaN"] = True
sns.pairplot(data, hue="was_NaN",vars=["LIMIT_BAL", "AGE","BILL_AMT1","PAY_AMT1"])


# In[42]:


data["was_NaN"] = False
data.loc[data["MARRIAGE"].isnull() == True, "was_NaN"] = True
sns.pairplot(data, hue="was_NaN",vars=["LIMIT_BAL", "AGE","BILL_AMT1","PAY_AMT1"])


# In[46]:


#remove new column added
data.drop('was_NaN', axis=1, inplace=True)


# No significant correlation between the variables and missing data so MCAR missingness of general pattern (most complex to handle). 
# 
# Missing data is an insignificant percentage of the overall data & we can safely drop columns. Will also impute to not lose significant information.

# In[7]:


#drop missing rows
data_dropped = data.copy()
data_dropped.dropna(inplace=True)
data['NaN'] = 'With Missing'
data_dropped['NaN'] = 'Without Missing'
combined_data = pd.concat([data, data_dropped], ignore_index=True)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

sns.countplot(x='EDUCATION_label', hue='NaN', data=combined_data, palette='magma', ax=axs[0])
axs[0].set_title('EDUCATION Distribution: Before and After Dropping NaNs', fontsize=8)
axs[0].set_xlabel('Education Level')
axs[0].set_ylabel('Count')
axs[0].tick_params(axis='x', rotation=45)
axs[0].legend(title='Data Status', fontsize='small')

sns.countplot(x='MARRIAGE_label', hue='NaN', data=combined_data, palette='magma', ax=axs[1])
axs[1].set_title('MARRIAGE Distribution: Before and After Dropping NaNs', fontsize=8)
axs[1].set_xlabel('Marriage Status')
axs[1].set_ylabel('Count')
axs[1].tick_params(axis='x', rotation=45)
axs[1].legend(title='Data Status', fontsize='small')

plt.tight_layout()
plt.show()

data.drop('NaN', axis=1, inplace=True)


# Distribution remains identical so no significant loss after dropping rows.

# In[6]:


#imputation using KNN
#MICE might be overkill here since it’s best suited for scenarios with intricate relationships and more extensive missing data patterns
data_copy = data.copy()
removed_columns = data_copy[['MARRIAGE', 'EDUCATION', 'SEX', 'Default']]
data_copy.drop(['MARRIAGE', 'EDUCATION', 'SEX', 'Default'], axis=1, inplace=True)
imputer = KNNImputer(n_neighbors=15)
df = imputer.fit_transform(data_copy)
data_imputed = pd.DataFrame(df) #convert back to dataframe
data_imputed.columns = ['LIMIT_BAL', 'AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5',
       'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
       'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
       'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'SEX_label', 'EDUCATION_label',
       'MARRIAGE_label', 'Default_label']
for col in ['MARRIAGE_label', 'EDUCATION_label']:
    data_imputed[col] = data_imputed[col].round(0).astype(int)
data_imputed = pd.concat([data_imputed, removed_columns.reset_index(drop=True)], axis=1)


# In[9]:


#compare distributions
data['NaN'] = 'With Missing'
data_imputed['NaN'] = 'Without Missing'
combined_data = pd.concat([data, data_imputed], ignore_index=True)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

sns.countplot(x='EDUCATION_label', hue='NaN', data=combined_data, palette='magma', ax=axs[0])
axs[0].set_title('EDUCATION Distribution: Before and After KNN Imputation', fontsize=8)
axs[0].set_xlabel('Education Level')
axs[0].set_ylabel('Count')
axs[0].tick_params(axis='x', rotation=45)
axs[0].legend(title='Data Status', fontsize='small')

sns.countplot(x='MARRIAGE_label', hue='NaN', data=combined_data, palette='magma', ax=axs[1])
axs[1].set_title('MARRIAGE Distribution: Before and After KNN Imputation', fontsize=8)
axs[1].set_xlabel('Marriage Status')
axs[1].set_ylabel('Count')
axs[1].tick_params(axis='x', rotation=45)
axs[1].legend(title='Data Status', fontsize='small')

plt.tight_layout()
plt.show()

data_imputed.drop('NaN', axis=1, inplace=True)


# Distribution remains identical so no significant loss after KNN imputation.

# In[7]:


data = data_imputed.copy() #moving forward with imputed data so more values to work with
#fix data types
cols = ['LIMIT_BAL', 'AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5',
       'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
       'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
       'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'SEX_label', 'EDUCATION_label',
       'MARRIAGE_label', 'Default_label']
data[cols] = data[cols].astype('int64')


# In[8]:


### Encoding

#remove bias by binary encoding
#binary Encoding can be more efficient than one-hot encoding since it generates fewer features and our data is already of high dimension and has many categories
def custom_binary_encode(value): #remove bias of labels
    return format(value, '02b')
data['SEX_binary'] = data['SEX_label'].apply(custom_binary_encode)
data['EDUCATION_binary'] = data['EDUCATION_label'].apply(custom_binary_encode)
data['MARRIAGE_binary'] = data['MARRIAGE_label'].apply(custom_binary_encode)
data['Default_binary'] = data['Default_label'].apply(custom_binary_encode)

#drop repetitive columns
data.drop(['MARRIAGE_label', 'EDUCATION_label', 'SEX_label', 'Default_label'], axis=1, inplace=True)


# In[120]:


### Outlier Detection

#check outliers in dataset using Z-Score
from scipy import stats
numerical_cols = data.select_dtypes(include=['int64']).columns
z = np.abs(stats.zscore(data[numerical_cols]))                           
threshold = 3
outliers = np.where(z > threshold)
count = np.count_nonzero(z > threshold)
count


# 7476 rows are classified as outliers. Since huge chunk of data, will not remove.

# In[9]:


### Scaling

#Scale LIMIT_BAL, BILL_AMT1-6, PAY_AMT1-6, Age data since extreme ranges
#Robust scaling (uses median and IQR) so robust to outliers
cols = ['LIMIT_BAL','BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
       'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
       'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'AGE']
scaler = RobustScaler(copy=True)
data_scaled = scaler.fit_transform(data[cols])
data_scaled = pd.DataFrame(data_scaled, columns=cols)
data_scaled = pd.concat([data_scaled, data.drop(columns=cols)], axis=1)


# In[19]:


#compare distribution of scaled data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
## create scatter plot using features 1 and 2 of original data
ax1.scatter(data['LIMIT_BAL'], data['BILL_AMT1'], alpha=0.5)
ax1.set_title('Original Data')
ax1.set_xlabel('LIMIT_BAL')
ax1.set_ylabel('BILL_AMT1')

## create scatter plot using features 1 and 2 of scaled data
ax2.scatter(data_scaled['LIMIT_BAL'], data_scaled['BILL_AMT1'], alpha=0.5)
ax2.set_title('Scaled Data')
ax2.set_xlabel('LIMIT_BAL')
ax2.set_ylabel('BILL_AMT1')

plt.tight_layout()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
## create scatter plot using features 1 and 2 of original data
ax1.scatter(data['LIMIT_BAL'], data['PAY_AMT6'], alpha=0.5)
ax1.set_title('Original Data')
ax1.set_xlabel('LIMIT_BAL')
ax1.set_ylabel('PAY_AMT6')

## create scatter plot using features 1 and 2 of scaled data
ax2.scatter(data_scaled['LIMIT_BAL'], data_scaled['PAY_AMT6'], alpha=0.5)
ax2.set_title('Scaled Data')
ax2.set_xlabel('LIMIT_BAL')
ax2.set_ylabel('PAY_AMT6')

plt.tight_layout()


# Data follows identical distribution after scaling.

# In[10]:


#combine datasets
#since credit data is for Apr-Sep 2005, remove additional rows from macro dataset
data_macro = data_macro[data_macro['Month'].isin(['Apr','May','Jun','Jul','Aug','Sep'])]

#add count of defaults in macro data
data_macro['defaults'] = 0
cols = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
rows = [8, 7, 6, 5, 4, 3]
for col, row_index in zip(cols, rows):
    count = (data[col].isin([1, 2, 3, 4, 5, 6, 7, 8, 9])).sum()
    data_macro.loc[row_index, 'defaults'] = count
data_macro


# In[11]:


#add count of defaults in income level data
#clean data
rows = data_income.loc[[0, 1, 3], 'Avg Income level']
avg = rows.mean()
data_income.loc[13, 'Demographic'] = 'others'
data_income.loc[13, 'Avg Income level'] = avg
data_income = data_income.drop([0, 1, 3])
data_income.loc[4, 'Demographic'] = 'University'
data_income.loc[5, 'Demographic'] = 'Graduate school'


# In[12]:


#add count of defaults
data_income['defaults'] = 0
count = ((data['EDUCATION_binary'] == '01') & (data['Default_binary'] == '01')).sum()
data_income.loc[2, 'defaults'] = count
count = ((data['EDUCATION_binary'] == '11') & (data['Default_binary'] == '01')).sum()
data_income.loc[4, 'defaults'] = count
count = ((data['EDUCATION_binary'] == '00') & (data['Default_binary'] == '01')).sum()
data_income.loc[5, 'defaults'] = count
count = ((data['EDUCATION_binary'] == '10') & (data['Default_binary'] == '01')).sum()
data_income.loc[13, 'defaults'] = count
count = ((data['SEX_binary'] == '01') & (data['Default_binary'] == '01')).sum()
data_income.loc[6, 'defaults'] = count
count = ((data['SEX_binary'] == '00') & (data['Default_binary'] == '01')).sum()
data_income.loc[7, 'defaults'] = count
count = ((data['AGE'] < 30) & (data['Default_binary'] == '01')).sum()
data_income.loc[8, 'defaults'] = count
count = ((data['AGE'] > 29) & (data['AGE'] < 35) & (data['Default_binary'] == '01')).sum()
data_income.loc[9, 'defaults'] = count
count = ((data['AGE'] > 34) & (data['AGE'] < 40) & (data['Default_binary'] == '01')).sum()
data_income.loc[10, 'defaults'] = count
count = ((data['AGE'] > 39) & (data['AGE'] < 45) & (data['Default_binary'] == '01')).sum()
data_income.loc[11, 'defaults'] = count
count = ((data['AGE'] > 44) & (data['AGE'] < 55) & (data['Default_binary'] == '01')).sum()
data_income.loc[12, 'defaults'] = count
count = ((data['AGE'] > 64) & (data['Default_binary'] == '01')).sum()
data_income.loc[14, 'defaults'] = count
data_income


# #### Exploratory Data Analysis

# In[76]:


### Statistical summary
data.describe()


# In[64]:


### Visualisations

all = ['LIMIT_BAL','AGE','PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3',
       'BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','SEX_binary',
       'EDUCATION_binary','MARRIAGE_binary','Default_binary']

#correlation heatmap for all variables
corr = data[all].corr()
fig1 = go.Figure(data=go.Heatmap(z=corr.values,
                  x=corr.index.values,
                  y=corr.columns.values,
                  showscale=True,
                  colorscale='Plasma',
                  colorbar=dict(
                  len=1.0,        
                  thickness=10,
                  xanchor='left'
                  )))
fig1.update_layout(title_text="Interactive Correlation Heatmap", title_font_size=15, height=600, width=1000)
fig1.show()

fig1_json = fig1.to_json()
with open("correlation_heatmap.json", "w") as f:
    f.write(fig1_json)


# In[86]:


#density plot for LIMIT_BAL
plt.figure(figsize=(8, 5))
fig2 = sns.displot(data, x="LIMIT_BAL", fill=True, kde=True, hue="Default", palette={'no': 'green', 'yes': 'red'})
plt.title("Default Percentage grouped by Amount of Credit Limit")
plt.show()

fig2.savefig("density_plot.png")


# This shows that higher the credit limit, lower is the chance of default. This is sensible as richer people tend to have higher credit limit and are so less likely to default on loans. The highest defaulters are for credit limit 0 to 100,000, with the highest being for credit limit 50,000, and the density for this interval is larger for defaulters than for non-defaulters.

# In[ ]:


#hiplot for defaults on previous payments
col = ['PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
fig = hip.Experiment.from_dataframe(data[col]).display()
fig3 = hip.Experiment.from_dataframe(data[col])

fig3.to_html('hiplot.html')


# In[91]:


#bill & pay amount distributions for last 3 months (joint kde plot)
col = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','Default']

sns.jointplot(data=data[col], x="BILL_AMT2", y="PAY_AMT1", kind="scatter", hue='Default')
plt.suptitle("Joint KDE Plot of Aug Bill & Payment Status", y=1.02)

sns.jointplot(data=data[col], x="BILL_AMT3", y="PAY_AMT2", kind="scatter", hue='Default')
plt.suptitle("Joint KDE Plot of Jul Bill & Payment Status", y=1.02)

sns.jointplot(data=data[col], x="BILL_AMT4", y="PAY_AMT3", kind="scatter", hue='Default')
plt.suptitle("Joint KDE Plot of Jun Bill & Payment Status", y=1.02)

plt.show()


# In[96]:


#bar charts for demographic data
filtered_data = filtered_data = data_income[data_income['Demographic'].isin(['Male', 'Female'])]
fig7 = px.bar(filtered_data, x='Demographic', y='defaults',
             hover_data=['defaults', 'Avg Income level'], color='Avg Income level',
             labels={'defaults':'Defaults'}, height=400,color_continuous_scale=px.colors.sequential.Viridis)
fig7.update_layout(title_text="SEX: Income Level vs Defaults", title_font_size=15, height=400, width=500, xaxis_title='')
fig7.show()
sex = fig7.to_json()
with open("sex_plot.json", "w") as f:
    f.write(sex)

filtered_data = filtered_data = data_income[data_income['Demographic'].isin(['High school', 'University', 'Graduate school', 'others'])]
fig8 = px.bar(filtered_data, x='Demographic', y='defaults',
             hover_data=['defaults', 'Avg Income level'], color='Avg Income level',
             labels={'defaults':'Defaults'}, height=400,color_continuous_scale=px.colors.sequential.Viridis)
fig8.update_layout(title_text="EDUCATION: Income Level vs Defaults", title_font_size=15, height=400, width=500, xaxis_title='')
fig8.show()
education = fig8.to_json()
with open("education_plot.json", "w") as f:
    f.write(education)

filtered_data = filtered_data = data_income[data_income['Demographic'].isin(['Under 30 yrs','30-34 yrs','35-39 yrs','40-44 yrs','45-54 yrs','65 yrs and above'])]
fig9 = px.bar(filtered_data, x='Demographic',  y='defaults',
             hover_data=['defaults', 'Avg Income level'], color='Avg Income level',
             labels={'defaults':'Defaults'}, height=400,color_continuous_scale=px.colors.sequential.Viridis)
fig9.update_layout(title_text="AGE: Income Level vs Defaults", title_font_size=15, height=400, width=500, xaxis_title='')
fig9.show()
age = fig9.to_json()
with open("age_plot.json", "w") as f:
    f.write(age)

grouped_data = data.groupby(['MARRIAGE', 'Default']).size().reset_index(name='Count')
fig10 = px.bar(grouped_data, x='MARRIAGE', y='Count', color='Default', barmode='group', labels={'MARRIAGE': 'MARRIAGE', 'Count': 'Count'}, 
               height=400,hover_data=['Count'],color_discrete_sequence=px.colors.qualitative.Set2)
fig10.update_layout(title_text="Relationship Status vs Defaults",title_font_size=15,height=400,width=600,xaxis_title='')
fig10.show()
marriage = fig10.to_json()
with open("marriage_plot.json", "w") as f:
    f.write(marriage)


# Female population is in majority and is more likely to default on the loan. This could be explained by their high percentage in the dataset.
# 
# University education level in highest in the dataset, and is most likely to default. This could be explained by their high percentage in the dataset. Next are Graduates, and then Highschools. This should be the opposite, but again could be explained by the difference in their numbers in the dataset. Though, those least likely to default are unknown labels (others). Logically speaking, they need to be above graduate level atleast as education level and likeliness of default has an inverse relationship.
# 
# Single people are highest in number and are most likely to default. This makes sense. Type 'other' may be people in a relationship.
# 
# Young people are more likely to default than the older population.

# In[65]:


#correlation heatmap for macro data
all = ['CPI', 'Unemployment Rate', 'defaults']

#correlation heatmap for all variables
corr = data_macro[all].corr()
fig32 = go.Figure(data=go.Heatmap(z=corr.values,
                  x=corr.index.values,
                  y=corr.columns.values,
                  showscale=True,
                  colorscale='Plasma',
                  colorbar=dict(
                  len=1.0,        
                  thickness=10,
                  xanchor='left'
                  )))
fig32.update_layout(title_text="Interactive Correlation Heatmap", title_font_size=15, height=300, width=600)
fig32.show()

fig32_json = fig32.to_json()
with open("correlation_heatmapmacro.json", "w") as f:
    f.write(fig32_json)


# In[99]:


#line chart for macro data
fig5 = make_subplots(specs=[[{"secondary_y": True}]])
fig5.add_trace(
    go.Scatter(x=data_macro["Month"], y=data_macro["Unemployment Rate"], mode='lines', name='Unemployment Rate',line=dict(color='green')),
    secondary_y=False
)
fig5.add_trace(
    go.Scatter(x=data_macro["Month"], y=data_macro["defaults"], mode='lines', name='Defaults',line=dict(color='orange')),
    secondary_y=True
)
fig5.update_layout(
    title='Unemployment Rate vs Defaults',
    xaxis_title='Month',
    yaxis_title='Unemployment Rate',
    yaxis2_title='Defaults',
    legend=dict(x=1.05, y=1, traceorder='normal', orientation='v'),
    showlegend=True,
    yaxis2=dict(showgrid=False),
)
fig5.show()


# In[101]:


#line chart for macro data
fig6 = make_subplots(specs=[[{"secondary_y": True}]])
fig6.add_trace(
    go.Scatter(x=data_macro["Month"], y=data_macro["CPI"], mode='lines', name='Inflation Rate',line=dict(color='green')),
    secondary_y=False
)
fig6.add_trace(
    go.Scatter(x=data_macro["Month"], y=data_macro["defaults"], mode='lines', name='Defaults',line=dict(color='orange')),
    secondary_y=True
)
fig6.update_layout(
    title='Inflation Rate vs Defaults',
    xaxis_title='Month',
    yaxis_title='Inflation Rate',
    yaxis2_title='Defaults',
    legend=dict(x=1.05, y=1, traceorder='normal', orientation='v'),
    showlegend=True,
    yaxis2=dict(showgrid=False)  
)
fig6.show()


# #### Feature Engineering

# In[172]:


numerical = ['LIMIT_BAL','AGE','PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3',
             'BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
categorical = ['SEX_binary','EDUCATION_binary','MARRIAGE_binary']

#T Test for numerical columns
p=[]
from scipy.stats import ttest_ind
for i in numerical:
    df1=data_scaled.groupby('Default_binary').get_group('00')
    df2=data_scaled.groupby('Default_binary').get_group('01')
    t,pvalue=ttest_ind(df1[i],df2[i])
    p.append(1-pvalue)
plt.figure(figsize=(7,7))
sns.barplot(x=p, y=numerical)
plt.title('Best Numerical Features')
plt.axvline(x=(1-0.05),color='r')
plt.xlabel('1-p value')
plt.show()


# All features show importance, and almost all of them show statistical significance so won't drop any.

# In[173]:


#Chi Square test for Categorical Columns
from scipy.stats import chi2_contingency
l=[]
for i in categorical:
    pvalue  = chi2_contingency(pd.crosstab(data_scaled['Default_binary'],data_scaled[i]))[1]
    l.append(1-pvalue)
plt.figure(figsize=(7,7))
sns.barplot(x=l, y=categorical)
plt.title('Best Categorical Features')
plt.axvline(x=(1-0.05),color='r')
plt.show()


# All categorical features show statistical significance so won't drop any.

# In[66]:


cols = ['BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
       'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
       'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
X = data_scaled[cols]

#perform SVD
U, s, Vt = np.linalg.svd(X)
V = Vt.T


# In[ ]:


#Scree plot to choose number of principle components
fig = plt.figure(figsize=(15, 6))
var_exp = s**2 / np.sum(s**2)
cum_var_exp = np.cumsum(var_exp)

plt.plot(range(1, len(var_exp) + 1), var_exp, 'bo-', label='Individual')
plt.plot(range(1, len(cum_var_exp) + 1), cum_var_exp, 'ro-', label='Cumulative')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.title('Scree Plot')
plt.legend()
plt.grid(True)


# In[67]:


#Apply PCA for 6 components
labels = data_scaled['Default_binary']
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X)
scores = X.to_numpy() @ V

#Feature loadings
print("\nFeature Loadings (scaled by singular values):")
for feature, loadings in zip(cols, pca.components_.T): 
    loading_str = ", ".join([f"PC{i+1} = {loading:7.4f}" for i, loading in enumerate(loadings)])
    print(f"{feature:25s}: {loading_str}")


# In[68]:


#PCA plot
labels = data_scaled['Default_binary']
components_df = pd.DataFrame(
    X_pca, 
    columns=[f"PC{i+1}" for i in range(6)]
)
components_df['Default_binary'] = labels

# Create a scatter plot matrix
fig = px.scatter_matrix(
    components_df,
    dimensions=[f"PC{i+1}" for i in range(6)],
    color='Default_binary',
    color_discrete_sequence=px.colors.qualitative.Plotly,
    title="PCA Scatter Matrix for 6 Principal Components",
    labels={f"PC{i+1}": f"PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})" for i in range(6)},
)

# Customize the plot
fig.update_traces(diagonal_visible=False, marker=dict(size=6, opacity=0.7))
fig.update_layout(
    title_font_size=18,
    height=900,
    width=900,
    coloraxis_colorbar=dict(title="Default Binary"),
    xaxis=dict(tickangle=45),  
    yaxis=dict(automargin=True)
)

fig.update_layout(
    margin=dict(l=50, r=50, t=50, b=50),
)
fig.show()

pca_plot = fig.to_json()
with open("pca_plot.json", "w") as f:
    f.write(pca_plot)


# There is significant overlap between the Default_binary classes (e.g., no clear separation between defaulted and non-defaulted groups). This might imply that the first two PCA components do not strongly distinguish between those who defaulted and those who did not. This could be a sign that the features (bill and payment amounts) alone do not provide a strong signal for predicting default risk, at least in the two-dimensional space captured by PCA.

# In[69]:


#transform dataset to include pca variables
pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)

other_columns = data_scaled.drop(['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
                                  'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
                                  'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6','MARRIAGE','EDUCATION',
                                  'SEX','Default'], axis=1)

data_transformed = pd.concat([X_pca_df, other_columns], axis=1)


# #### Pre-Processing

# In[70]:


#Split test and train data
x = data_transformed.drop(columns = 'Default_binary')
y = data_transformed['Default_binary']

#will train on 70% of the data, test on the remaining 30%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=50)


# In[71]:


### Class Imbalance

#check imbalance
temp = y_train.value_counts()
df = pd.DataFrame({'Default': temp.index,'values': temp.values})
plt.figure(figsize = (6,6))
plt.title('Default Credit Card Clients - target value - data unbalance\n (Not Default = 00, Default = 01)')
sns.set_color_codes("pastel")
sns.barplot(x = 'Default', y="values", data=df)
locs, labels = plt.xticks()
plt.show()


# Majority class (not default) has 77% samples. Minority class (default) has 22% samples. Data is highly imbalanced. The best method here would be to undersample majority class using one-class classification. As we have not learnt it yet, we will apply SMOTE for demonstration purposes.

# In[72]:


# display original class distribution
print("\nOriginal class distribution:")
print(y_train.value_counts())

# apply SMOTE
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

# display new class distribution
print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())


# In[78]:


# Bar chart for class distribution after SMOTE
sns.countplot(x=y_train_resampled)
plt.title('Class Distribution After SMOTE')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()

#check distribution for one variable
plt.figure(figsize=(12,5))

resampled_data = pd.DataFrame(x_train_resampled, columns=x_train_resampled.columns)
resampled_data['Default_binary'] = y_train_resampled

plt.subplot(1,2,1)
sns.histplot(data[data['Default_binary']=='00']['LIMIT_BAL'], color='blue', label='Not Default', kde=True) 
sns.histplot(data[data['Default_binary']=='01']['LIMIT_BAL'], color='red', label='Default', kde=True) 
plt.title('Credit Limit Distribution Before SMOTE')
plt.legend()

plt.subplot(1,2,2)
sns.histplot(resampled_data[resampled_data['Default_binary']=='00']['LIMIT_BAL'], color='blue', label='Not Default', kde=True)
sns.histplot(resampled_data[resampled_data['Default_binary']=='01']['LIMIT_BAL'], color='red', label='Default', kde=True)
plt.title('Credit Limit Distribution After SMOTE')
plt.legend()

plt.tight_layout()
plt.show()


# SMOTE made an impact on the shape of the histograms since the smoothed-out line now follows an exaggerated pattern for default class, i.e. the mean, median and mode have changed. There is increase in count for some values of LIMIT_BAL, but not relative to the proportion of the difference in the count for each rectangle, which has raised the height of the histograms. So the values now occur more frequently than before.

# In[79]:


#fix encoding for target variable for ROC curve calculation
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_test = le.fit_transform(y_test)
y_train = le.fit_transform(y_train)
y_train_resampled = le.fit_transform(y_train_resampled)


# Now we have 3 sub-datasets: 
# 1. x_train, y_train: training data with class inbalance
# 2. x_train_resampled, x_train_resampled: training data with balanced classes
# 3. x_test, y_test: test data

# #### Modelling

# In[80]:


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn import datasets
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report,confusion_matrix


# ##### **Logistic Regression**

# In[81]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 1, 10]}

lr = LogisticRegression(penalty='l2', class_weight='balanced')
lr = GridSearchCV(estimator = lr, param_grid = param_grid, n_jobs = -1)


# In[82]:


#apply on imbalanced data
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[83]:


# Coefficients for feature importance
best_lr_model = lr.best_estimator_
y_pred = best_lr_model.predict(x_test)

coefficients = best_lr_model.coef_[0] 
feature_names = x_train.columns
coeff_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})
coeff_df['Abs_Coefficient'] = coeff_df['Coefficient'].abs()
coeff_df = coeff_df.sort_values(by='Abs_Coefficient', ascending=False)
print(coeff_df)


# In[84]:


#apply on balanced data
lr.fit(x_train_resampled, y_train_resampled)
y_pred = lr.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[133]:


from sklearn.metrics import roc_curve, auc
y_prob = lr.predict_proba(x_test)[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# Same result on both training sets: Accuracy = 0.67, AUC = 0.72

# ##### **Support Vector Machine**

# In[ ]:


from sklearn import svm
from sklearn import metrics

#Radial Basis Function Kernel with low gamma 0.01
clf = svm.SVC(kernel='rbf', random_state=0, gamma=.01, C=1) 

#apply on imbalanced data
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))


# In[160]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[161]:


#apply on balanced data
clf.fit(x_train_resampled, y_train_resampled)
y_pred = clf.predict(x_test)

# model accuracy:
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[162]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# Best result on imbalanced training set: Accuracy = 0.82
# 
# Best ability to distinguish between classes on balanced training set: AUC = 0.70

# ##### **XGBoost Classifier**

# In[29]:


from xgboost import XGBClassifier
import joblib

model = XGBClassifier(verbosity=1, 
                      scale_pos_weight=1,
                      learning_rate=0.01,  
                      colsample_bytree = 0.8,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=100, 
                      reg_alpha = 0.3,
                      max_depth=4, 
                      gamma=5)


# In[ ]:


#fix encoding to make compatible with xgboost
x_train_label = pd.get_dummies(x_train, columns=['SEX_binary', 'EDUCATION_binary', 'MARRIAGE_binary'])
x_test_label = pd.get_dummies(x_test, columns=['SEX_binary', 'EDUCATION_binary', 'MARRIAGE_binary'])
x_train_resampled_label = pd.get_dummies(x_train_resampled, columns=['SEX_binary', 'EDUCATION_binary', 'MARRIAGE_binary'])
x_test_label = x_test_label.reindex(columns=x_train_label.columns, fill_value=0)
x_train_resampled_label = x_train_resampled_label.reindex(columns=x_train_label.columns, fill_value=0)


# In[ ]:


#apply on imbalanced data
model.fit(x_train_label, y_train)
y_pred = model.predict(x_test_label)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred))


# In[30]:


#apply on balanced data
model.fit(x_train_resampled_label, y_train_resampled)

#save model for streamlit
xgb_model = model.fit(x_train_resampled_label, y_train_resampled)
joblib.dump(xgb_model, 'xgb_model.pkl')
    
# Make predictions and evalute
y_pred = model.predict(x_test_label)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred))


# In[156]:


y_prob = model.predict_proba(x_test_label)[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# Best result on imbalanced training set: Accuracy = 0.81, AUC = 0.77

# Best Model:
# 
# XGBoost Classifier on imbalanced training set:
# 
# Accuracy: 0.813, AUC: 0.77
# 

# #### Time Series Forecasting for CPI

# In[13]:


from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose


# In[14]:


cpi = pd.read_excel(r"C:\Users\manos\OneDrive\Desktop\MSU\Fall 2024\CMSE 830 Foundations of Data Science\Project\Midterm\CPI data yearly.xlsx")
cpi['Month'] = pd.to_datetime(cpi['Month'], format='%b %Y', errors='coerce')
cpi.set_index('Month', inplace=True)
cpi.index.freq = 'MS'

# Split into training and test sets
train_size = len(cpi) - 3
train = cpi[:train_size]
test = cpi[train_size:]

# Plot original data with train/test split
plt.figure(figsize=(15, 6))
plt.plot(train.index, train['CPI'], label="Training Data", color='blue')
plt.plot(test.index, test['CPI'], label="Test Data", color='red')
plt.xlabel("Date")
plt.ylabel("CPI")
plt.xticks(rotation=45)
plt.legend()
plt.title("Consumer Price Index (2000-2005)")
plt.grid()
plt.tight_layout()
plt.show()


# In[15]:


# STL Decomposition
decomposition = seasonal_decompose(train, period=12)
plt.figure(figsize=(50, 10))
decomposition.plot()
plt.xticks([])
plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

#ACF plot
plot_acf(train, lags=35, alpha=0.05, ax=ax1)
ax1.set_title('ACF')

#PACF plot
plot_pacf(train, lags=30, alpha=0.05, ax=ax2)
ax2.set_title('PACF')

plt.tight_layout()
plt.show()


# In[16]:


#ADF test for stationarity
from statsmodels.tsa.stattools import adfuller

def adf_test(series, name="Time Series"):
    result = adfuller(series.dropna())
    adf_statistic = result[0]
    p_value = result[1]
    used_lags = result[2]
    num_observations = result[3]
    critical_values = result[4]
    
    # Print the results
    print(f"Results of Augmented Dickey-Fuller Test for {name}:")
    print(f"ADF Statistic: {adf_statistic}")
    print(f"p-value: {p_value}")
    print(f"Lags Used: {used_lags}")
    print(f"Number of Observations: {num_observations}")
    print("Critical Values:")
    for key, value in critical_values.items():
        print(f"  {key}: {value}")
    
    if p_value < 0.05:
        print("\nThe series is stationary (reject the null hypothesis).")
        stationary = True
    else:
        print("\nThe series is non-stationary (fail to reject the null hypothesis).")
        stationary = False

    return {
        "ADF Statistic": adf_statistic,
        "p-value": p_value,
        "Used Lags": used_lags,
        "Number of Observations": num_observations,
        "Critical Values": critical_values,
        "Stationary": stationary
    }

adf_results_original = adf_test(train['CPI'], name="Original CPI")


# In[17]:


#first difference since series is non-stationary
train.loc[:, 'CPI_diff'] = train['CPI'] - train['CPI'].shift(1)
adf_results_diff = adf_test(train['CPI_diff'], name="First Differenced CPI")


# In[26]:


# Fit ARIMA(p,d,q) model and forecast
model = ARIMA(train['CPI'], order=(1, 1, 3))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))

# Evaluate
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"RMSE: {rmse}")

# Plot forecasts
plt.figure(figsize=(15, 6))
plt.plot(train.index, train['CPI'], label="Training Data", color='blue')
plt.plot(test.index, test['CPI'], label="Test Data", color='red')
plt.plot(test.index, forecast, label=f"ARIMA(1,1,3) Forecast", linestyle="--", color='green')
plt.xlabel("Date")
plt.ylabel("CPI")
plt.xticks(rotation=45)
plt.legend()
plt.title("One-Step-Ahead Forecast")
plt.grid()
plt.show()

