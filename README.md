# Credit Card Default Prediction
This classification problem aims to predict whether a customer will default on next month's credit card payment using a [UCI dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

Access the [streamlit](https://fds-project-cmse830-credit-card-default-prediction-r2f68jop9pe.streamlit.app/) web app to explore the insights derived from the analysis.

## Table of Contents:
1. [Introduction](#introduction)
2. [Environment](#environment)
3. [Datasets](#datasets)
4. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
5. [Exploratory Data Analysis](https://fds-project-cmse830-credit-card-default-prediction-r2f68jop9pe.streamlit.app/)
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

![missing heatmap](https://github.com/user-attachments/assets/c728f8d7-3e8d-4ad0-bb2e-d720b4e1de41)
[View missingcorrelation_heatmap.json](./missingcorrelation_heatmap.json)

## Future Work:
