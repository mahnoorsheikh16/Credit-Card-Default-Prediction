# Credit Card Default Prediction
This classification problem aims to predict whether a customer will default on next month's credit card payment using a [UCI dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

Access the [streamlit](https://credit-card-default-prediction-msu-cmse830.streamlit.app/) web app to explore insights derived from the analysis.

## Table of Contents:
1. [Introduction](#introduction)
2. [Environment](#environment)
3. [Datasets](#datasets)
4. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Future Work](#future-work)

## Introduction:
Credit risk refers to the cardholder's inability to make required payments on their credit card debt, leading to 'credit default.' This poses a major concern for financial institutions, as defaults can result in significant losses and, in severe cases, even bankruptcy. Conducting thorough evaluations and verifying a borrower's ability to repay can help prevent the over-issuance of credit cards to unqualified applicants, thereby minimizing credit risk.

Machine learning models can be deployed to identify risky customers and minimise lenders' losses. By using algorithms to study historical transactions and customer demographics, we can apply the findings to future customers, effectively distinguishing between risky and non-risky profiles. This approach leads to more efficient loan lending practices.

## Environment
The analysis has been conducted in Python and the source code requires the following libraries: `pandas`, `numpy`, `sklearn`, `scipy`, and `imblearn`.

Primary libraries used for visualizations are `matplotlib`, `seaborn`, `plotly`, and `hiplot`.

`streamlit` must be installed to run the `streamlit_code.py` file. Run it using the following command in the terminal:
```
streamlit run streamlit_code.py
```

## Datasets:
**I. Default of Credit Card Clients**

Dataset contains information on credit card clients in Taiwan from April 2005 to September 2005. It has 30,000 instances across 25 attributes, contains multivariate characteristics, and the attributes have both integer, categorical and real data types. The attribute summary is as follows:

`ID`: ID of each client

`LIMIT_BAL`: Amount of given credit in NT dollars (includes individual and family/supplementary credit)

`SEX`: Gender (male, female)

`EDUCATION`: Level of education (graduate school, university, high school, others)

`MARRIAGE`: Marital status (married, single, others)

`AGE`: Age in years

`PAY_0`: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)

`PAY_2`: Repayment status in August, 2005 (scale same as above)

`PAY_3`: Repayment status in July, 2005 (scale same as above)

`PAY_4`: Repayment status in June, 2005 (scale same as above)

`PAY_5`: Repayment status in May, 2005 (scale same as above)

`PAY_6`: Repayment status in April, 2005 (scale same as above)

`BILL_AMT1`: Amount of bill statement in September, 2005 (NT dollar)

`BILL_AMT2`: Amount of bill statement in August, 2005 (NT dollar)

`BILL_AMT3`: Amount of bill statement in July, 2005 (NT dollar)

`BILL_AMT4`: Amount of bill statement in June, 2005 (NT dollar)

`BILL_AMT5`: Amount of bill statement in May, 2005 (NT dollar)

`BILL_AMT6`: Amount of bill statement in April, 2005 (NT dollar)

`PAY_AMT1`: Amount of previous payment in September, 2005 (NT dollar)

`PAY_AMT2`: Amount of previous payment in August, 2005 (NT dollar)

`PAY_AMT3`: Amount of previous payment in July, 2005 (NT dollar)

`PAY_AMT4`: Amount of previous payment in June, 2005 (NT dollar)

`PAY_AMT5`: Amount of previous payment in May, 2005 (NT dollar)

`PAY_AMT6`: Amount of previous payment in April, 2005 (NT dollar)

`default payment next month`: Default payment (yes, no)

**II. Macroeconomic Data for Taiwan**

Data on labour, income, and inflation for Taiwan in 2005 have been sourced from the [National Statistics Republic of China (Taiwan)](https://eng.stat.gov.tw/cl.aspx?n=2324) and [DGBAS Government Bureau](https://www.dgbas.gov.tw/default.aspx).

`CPI`: Consumer Price Index representing the average change over time in the prices paid by consumers for a representative basket of consumer goods and services

`Unemployment Rate`: Percentage of people in the labour force who are unemployed (includes civilians age 15 & above who were: (i) jobless (ii) available for work (iii) seeking a job or waiting for results after job seeking during the reference week (iv) waiting for a recall after layoff (v) having a job offer but have not started to work)

`Avg Income Level`: Disposable income of employees (including those having: (i) full-time, part-time, or another payroll (ii) entrepreneurial income (iii) property income (iv) imputed rent income (v) current transfer receipts)

## Data Cleaning and Preprocessing:
To prepare the dataset for model evaluation, I start with variable identification and classification. I first remove unique variables like 'ID' and rename columns for better understanding. The target variable is the binary variable 'Default,' and the explanatory variables have information about customer demographics and payment history. These are 14 quantitative variables with discrete data (integers), i.e. LIMIT_BAL, AGE, BILL_AMT1-6, PAY_AMT1-6, and 10 categorical variables, where EDUCATION and MARRIAGE are nominal; SEX and Default are binary, and PAY_1-6 are ordinal. The macroeconomic and income datasets have continuous numerical data. I further check the unique values of each categorical variable for inconsistencies during data quality assessment and find all labels to match their given data descriptions, except for PAY_1-6 variables. '-2' and '0' instances are undocumented but make up a significant chunk of the data so they cannot be treated as unknowns. Upon inspection of the label order and observations, we can infer '-2' to be no payment due and '0' to represent a payment delay for <1 month (however we cannot be sure).

As part of data cleaning, I started by removing 35 duplicate rows and using the Label Encoder for variables 'MARRIAGE', 'SEX', 'Default' and 'EDUCATION' so their values can be compatible with further analytical techniques. The 'EDUCATION' and 'MARRIAGE' variables have 345 and 54 missing values respectively, which make up less than 2% of the dataset. To classify the type of feature missingness, three methods are employed: heatmap, correlation matrix and pair plots:
![missing_heatmap](https://github.com/user-attachments/assets/5affd66f-8fec-4107-a7b6-908e32fa83c7)
![missingcor_heatmap](https://github.com/user-attachments/assets/98a7bc03-6950-49bd-ad18-8b8b36fbca23)
#### EDUCATION Pairplot:
![education_pairplot](https://github.com/user-attachments/assets/a540aed2-70fa-40ed-bc87-411c0a91c118)
#### MARRIAGE Pairplot
![marriage_pairplot](https://github.com/user-attachments/assets/135b64d0-79df-4281-b837-b524380fd722)

No significant correlation is found between between the variables and missing data so I classify it as MCAR missingness of general pattern (most complex to handle). 
Since missing data is an insignificant percentage of the overall data, we can safely drop columns. However, I also employ KNN imputation to not lose any significant information. The reason for choosing KNN is that other techniques like MICE might be overkill here since it’s best suited for scenarios with intricate relationships and more extensive missing data patterns. Also, since categorical features have missing data, they cannot be imputed using numerical methods. KNN imputation is tested with various n_neighbors and is set to 15 neighbours for maximum accuracy.

To verify changes to the distribution of data post-handling of missing values, I visualize using count plots. 

![data_drop](https://github.com/user-attachments/assets/e5961f23-1115-4619-8b81-f4dc40848249)
![data_imputation](https://github.com/user-attachments/assets/eb80082f-a82f-43c1-b66c-6061ebbd2917)

The distributions remain identical after both methods so no significant loss is recorded. I move forward with imputed data so there are more values to work with and fix integer data types for features.

Since Label Encoder has introduced bias in the model (higher labels will be given more weightage), I use Binary Encoding on 'MARRIAGE', 'SEX', 'Default' and 'EDUCATION' variables. Binary Encoding can be more efficient than One-Hot Encoding since it generates fewer features and our data is already of a high dimension with variables having many categories. I then drop repetitive columns from the dataset to make it cleaner. 

To identify outliers in the dataset, I employ the Z-Score method and set the threshold to 'z > 3'. 7476 rows are classified as outliers and since this makes up 25% of the dataset, I do not remove them. Further, I scale LIMIT_BAL, BILL_AMT1-6, PAY_AMT1-6 variables due to their large ranges. Robust Scaler, which uses median and IQR as benchmarks, is employed as it is robust to outliers. Scatterplots are used for LIMIT_BAL, BILL_AMT and PAY_AMT variables to visualize changes in their distributions after scaling.
#### LIMIT_BAL vs BILL_AMT1
![scaleddata_billamt](https://github.com/user-attachments/assets/d487a29d-bd16-486e-869a-21ba3f6bc805)
#### LIMIT_BAL vs PAY_AMT6
![scaleddata_payamt](https://github.com/user-attachments/assets/d3c7ac1a-fd9b-4c0b-b710-337615bf2856)

The data is found to follow an identical relationship after scaling.

Finally, the credit card default dataset is combined with the macroeconomic and income datasets using default counts. This is used to further explore the relationships between variables using exploratory data analysis.

## Exploratory Data Analysis
Visit the [streamlit web app](https://credit-card-default-prediction-msu-cmse830.streamlit.app/) to explore the insights derived from the dataset. The app features summaries of the T-test and Chi-Square test, which identify significant variables, as well as methods like SMOTE used to manage class imbalance.

## Future Work:
This project is a work in progress. The goal is to apply these insights to different machine learning models to predict whether a customer will default on their next credit card payment. I also plan to use regression to explore how changes in specific variables affect the probability of default. Scaling the data will take this analysis further by uncovering true relationships without the noise, especially since the dataset contains many outliers.
