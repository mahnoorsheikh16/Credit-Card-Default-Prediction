import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import hiplot as hip
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import plotly.io as pio
from PIL import Image
import streamlit.components.v1 as components
import joblib

#add navigation sidebar
st.sidebar.title("🔎Explore")
page = st.sidebar.selectbox("Select a page:", ["📄Overview", "🕵🏻Identify Defaults", "📊Data Analysis & Insights", "🖥️Decoding the Algorithm"], index=0)
for _ in range(15):  # Change 10 to the number of empty lines you want
    st.sidebar.write("")
st.sidebar.write("View the code and dataset details: https://github.com/mahnoorsheikh16/Credit-Card-Default-Prediction")

#import data
data = pd.read_csv("https://raw.githubusercontent.com/mahnoorsheikh16/Credit-Card-Default-Prediction/refs/heads/main/datasets/UCI_Credit_Card.csv")
data_macro = pd.read_excel("https://raw.githubusercontent.com/mahnoorsheikh16/Credit-Card-Default-Prediction/main/data_macro.xlsx")
data_income = pd.read_excel("https://raw.githubusercontent.com/mahnoorsheikh16/Credit-Card-Default-Prediction/main/data_income.xlsx")
test_set = pd.read_csv("https://raw.githubusercontent.com/mahnoorsheikh16/Credit-Card-Default-Prediction/refs/heads/main/test_set.csv")

data.rename(columns={'default.payment.next.month': 'Default'}, inplace=True)
data.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)

#set page content
if page == "📄Overview":
    st.header('Credit Default Predictor: Identify Next Month’s High-Risk Clients')
    st.write("")
    st.write("")
    total_customers = len(data)
    total_defaults = len(data[data['Default'] == 'yes'])
    money_lost = data['BILL_AMT1'].sum()
    perc = (total_defaults / total_customers)*100
    col1, col2, col3, col4 = st.columns([0.5, 0.5, 0.5, 1])
    with col1:
        st.metric("**Total Customers**", total_customers)
    with col2:
        st.metric("**Total Defaults**", total_defaults)
    with col3:
        st.metric("**Default Percentage**", f"{perc:,}%")
    with col4:
        st.metric("**Total NTD Lost**", f"${money_lost:,}")
    st.write("")
    st.write("Welcome to the Credit Default Prediction Dashboard! Designed for risk managers and financial teams, this tool allows you to upload client data and predict next month’s credit card defaults. By leveraging advanced machine learning, it provides clear insights to identify high-risk clients early, enabling proactive decisions before monthly reviews or client assessments to minimize financial risk.")
    st.write("Navigate to the 'Identify Defaults' page to upload client data and predict next month's defaulting customers.")

elif page == "🕵🏻Identify Defaults":
    st.subheader("Upload client data for next month's defaults")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Or download the sample test set and upload it back for predictions:")
    with col2:
        csv_data = test_set.to_csv(index=False)
        st.download_button(label="Download Test Set", data=csv_data, file_name="test_set.csv", mime="text/csv",)
    uploaded_file = st.file_uploader("Upload your CSV file for prediction", type=["csv"])
    st.write("")
    st.write("")
    xgb_model = joblib.load('xgb_model.pkl')
    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        try:
            #check file has all columns
            required_columns = ['PC1','PC2','PC3','PC4','PC5','PC6','LIMIT_BAL','AGE','PAY_1','PAY_2',
                                'PAY_3','PAY_4','PAY_5','PAY_6','SEX_binary_00','SEX_binary_01',
                                'EDUCATION_binary_00','EDUCATION_binary_01','EDUCATION_binary_10',
                                'EDUCATION_binary_11','MARRIAGE_binary_00','MARRIAGE_binary_01',
                                'MARRIAGE_binary_10']
            missing_cols = [col for col in required_columns if col not in user_data.columns]
            if missing_cols:
                st.error(f"The following required columns are missing: {missing_cols}")
            else:
                predictions = xgb_model.predict(user_data[required_columns])
                user_data['Default_Prediction'] = ['Default' if pred == 1 else 'No Default' for pred in predictions]

                #calculate summary
                total_rows = len(user_data)
                num_defaults = sum(user_data['Default_Prediction'] == 'Default')
                num_non_defaults = sum(user_data['Default_Prediction'] == 'No Default')
                percent_defaults = (num_defaults / total_rows) * 100
                percent_non_defaults = (num_non_defaults / total_rows) * 100
    
                st.write("**Prediction Results:**")
                col1, col2 = st.columns([0.75, 1])
                with col1:
                    st.dataframe(user_data[['Default_Prediction']])
                with col2:
                    st.write(f"Total Observations: {total_rows}")
                    st.write(f"Defaults: {num_defaults} ({percent_defaults:.2f}%)")
                    st.write(f"Non-Defaults: {num_non_defaults} ({percent_non_defaults:.2f}%)")
                    csv_output = user_data.to_csv(index=False)
                    st.download_button("Download Full Dataset with Defaults", csv_output, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif page=='📊Data Analysis & Insights':
    st.subheader("Explore Detailed Analysis Across Tabs")
    st.write("**Datasets:** Summary of the datasets, key metrics and the data sources used.")
    st.write("**IDA:** Explores datasets' structure, quality, and key patterns to guide further analysis.")
    st.write("**EDA: Demographic Data:** Defaults in relation to Gender, Relationship Status, Age, & Education Level")
    st.write("**EDA: Credit Limit & Balance:** Defaults in relation to Credit Limit and Monthly Repayment History")
    st.write("**EDA: Macroeconomic Factors:** Defaults in relation to Unemployment Rate and Inflation")
    st.write("")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Datasets", "IDA", "EDA: Demographic Data", "EDA: Credit Limit & Balance", "EDA: Macroeconomic Factors"])
    
    with tab1:
        st.write("**I. Default of Credit Card Clients**")
        st.write("[UCI dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) contains information on credit card clients in Taiwan from April 2005 to September 2005. It has 30,000 instances across 25 attributes, contains multivariate characteristics, and the attributes have both integer, categorical and real data types. The attribute summary is as follows:")
        st.write("ID: ID of each client")
        st.write("LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit)")
        st.write("SEX: Gender (male, female)")
        st.write("EDUCATION: Level of education (graduate school, university, high school, others)")
        st.write("MARRIAGE: Marital status (married, single, others)")
        st.write("AGE: Age in years")
        st.write("PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)")
        st.write("PAY_2: Repayment status in August, 2005 (scale same as above)")
        st.write("PAY_3: Repayment status in July, 2005 (scale same as above)")
        st.write("PAY_4: Repayment status in June, 2005 (scale same as above)")
        st.write("PAY_5: Repayment status in May, 2005 (scale same as above)")
        st.write("PAY_6: Repayment status in April, 2005 (scale same as above)")
        st.write("BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)")
        st.write("BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)")
        st.write("BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)")
        st.write("BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)")
        st.write("BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)")
        st.write("BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)")
        st.write("PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)")
        st.write("PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)")
        st.write("PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)")
        st.write("PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)")
        st.write("PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)")
        st.write("PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)")
        st.write("default payment next month: Default payment (yes, no)")
        st.write("")
        st.write("**II. Macroeconomic Data for Taiwan**")
        st.write("Data on labour, income, and inflation for Taiwan in 2005 have been sourced from the [National Statistics Republic of China (Taiwan)](https://eng.stat.gov.tw/cl.aspx?n=2324) and [DGBAS Government Bureau](https://www.dgbas.gov.tw/default.aspx).")
        st.write("CPI: Consumer Price Index representing the average change over time in the prices paid by consumers for a representative basket of consumer goods and services")
        st.write("Unemployment Rate: Percentage of people in the labour force who are unemployed (includes civilians age 15 & above who were: (i) jobless (ii) available for work (iii) seeking a job or waiting for results after job seeking during the reference week (iv) waiting for a recall after layoff (v) having a job offer but have not started to work)")
        st.write("Avg Income Level: Disposable income of employees (including those having: (i) full-time, part-time, or another payroll (ii) entrepreneurial income (iii) property income (iv) imputed rent income (v) current transfer receipts)")

    with tab2:
        st.write("**Variable Identification and Classification**")
        st.write("Unique variable 'ID' is removed and columns are renamed for better understanding. The target variable is the binary variable 'Default,' and the explanatory variables have information about customer demographics and payment history. These are 14 quantitative variables with discrete data (integers), i.e. LIMIT_BAL, AGE, BILL_AMT1-6, PAY_AMT1-6, and 10 categorical variables, where EDUCATION and MARRIAGE are nominal; SEX and Default are binary, and PAY_1-6 are ordinal. The macroeconomic and income datasets have continuous numerical data.")
        st.write("**Data Quality Assessment**")
        st.write("Checking the unique values of each categorical variable for inconsistencies reveals all labels to match their given data descriptions, except for PAY_1-6 variables. '-2' and '0' instances are undocumented but make up a significant chunk of the data so they cannot be treated as unknowns. Upon inspection of the label order and observations, we can infer '-2' to be no payment due and '0' to represent a payment delay for <1 month (however we cannot be sure).")
        st.write("35 duplicate rows are removed and Label Encoder is used for variables 'MARRIAGE', 'SEX', 'Default' and 'EDUCATION' so their values can be compatible with further analytical techniques.")
        st.write("**Missing Data Analysis**")
        st.write("The 'EDUCATION' and 'MARRIAGE' variables have 345 and 54 missing values respectively, which make up less than 2% of the dataset. To classify the type of feature missingness, three methods are employed: heatmap, correlation matrix and pair plots:")
        imageida1 = Image.open("IDA_missing_heatmap.png")
        st.image(imageida1, use_column_width=True)
        with open("IDA_missing_corr_plot.json", "r") as f:
            IDA_missing_corr_json = f.read()
            fig1 = pio.from_json(IDA_missing_corr_json)
        st.plotly_chart(fig1, use_container_width=True)
        st.write("EDUCATION Pairplot:")
        imageida1 = Image.open("IDA_education_pairplot.png")
        st.image(imageida1, use_column_width=True)
        st.write("MARRIAGE Pairplot:")
        imageida1 = Image.open("IDA_marriage_pairplot.png")
        st.image(imageida1, use_column_width=True)
        st.write("No significant correlation is found between between the variables and missing data so I classify it as MCAR missingness of general pattern (most complex to handle). Since missing data is an insignificant percentage of the overall data, we can safely drop columns. However, I also employ KNN imputation to not lose any significant information. The reason for choosing KNN is that other techniques like MICE might be overkill here since it’s best suited for scenarios with intricate relationships and more extensive missing data patterns. Also, since categorical features have missing data, they cannot be imputed using numerical methods. KNN imputation is tested with various n_neighbors and is set to 15 neighbours for maximum accuracy.")
        st.write("To verify changes to the distribution of data post-handling of missing values, I visualize using count plots.")
        st.write("")
        imageida1 = Image.open("IDA_drop_countplot.png")
        st.image(imageida1, use_column_width=True)
        imageida1 = Image.open("IDA_knn_countplot.png")
        st.image(imageida1, use_column_width=True)
        st.write("The distributions remain identical after both methods so no significant loss is recorded. I move forward with imputed data so there are more values to work with and fix integer data types for features.")
        st.write("**Encoding**")
        st.write("Since Label Encoder has introduced bias in the model (higher labels will be given more weightage), I use Binary Encoding on 'MARRIAGE', 'SEX', 'Default' and 'EDUCATION' variables. Binary Encoding can be more efficient than One-Hot Encoding since it generates fewer features and our data is already of a high dimension with variables having many categories. Repetitive columns are then dropped from the dataset to make it cleaner.")
        st.write("**Outlier Detection and Scaling**")
        st.write("Z-Score method is employed and the threshold is set to 'z > 3'. 7476 rows are classified as outliers and since this makes up 25% of the dataset, I do not remove them. Further, I scale LIMIT_BAL, BILL_AMT1-6, PAY_AMT1-6, AGE variables due to their large ranges. Robust Scaler, which uses median and IQR as benchmarks, is employed as it is robust to outliers. Scatterplots are used for LIMIT_BAL, BILL_AMT and PAY_AMT variables to visualize changes in their distributions after scaling.")
        st.write("LIMIT_BAL vs BILL_AMT1:")
        imageida1 = Image.open("IDA_bill_scatterplot.png")
        st.image(imageida1, use_column_width=True)
        st.write("LIMIT_BAL vs PAY_AMT6:")
        imageida1 = Image.open("IDA_pay_scatterplot.png")
        st.image(imageida1, use_column_width=True)
        st.write("The data is found to follow an identical relationship after scaling.")
        st.write("**Merging Datasets**")
        st.write("The credit card default dataset is combined with the macroeconomic and income datasets using default counts. This is used to further explore the relationships between variables using exploratory data analysis.")

    with tab3:
        with open("sex_plot.json", "r") as f:
            sex_json = f.read()
            fig1 = pio.from_json(sex_json)
        st.plotly_chart(fig1, use_container_width=True)
        st.write("Female population is in majority and is more likely to default on the payments. This could be explained by their high percentage in the dataset and lower income levels.")
        st.write("")
        st.write("")
        with open("education_plot.json", "r") as f:
            edu_json = f.read()
            fig2 = pio.from_json(edu_json)
        st.plotly_chart(fig2, use_container_width=True)
        st.write("Those with a university education level are most likely to default. Next are those with graduate level education having the highest income level, followed by highschool with the lowest income level. The default count follows an inverse relationship of being higher for those with a higher education level. Though unexpected, this can be explained by the difference in their numbers in the dataset since those with a highschool level education and lower income levels will be less likely to qualify for a credit card. The unknown labels (others) are insignificant in number and can be ignored.")
        st.write("")
        st.write("")
        with open("age_plot.json", "r") as f:
            age_json = f.read()
            fig4 = pio.from_json(age_json)
        st.plotly_chart(fig4, use_container_width=True)
        st.write("As age increases, the income level also rises. This could explain the increaisngly lower count of defaults as age progresses. Those in the 45-54 age range are a bit higher in percentage in the data and this may be why their count of defaults breaks from the pattern and is slightly higher.")
        st.write("")
        st.write("")
        with open("marriage_plot.json", "r") as f:
            mar_json = f.read()
            fig3 = pio.from_json(mar_json)
        st.plotly_chart(fig3, use_container_width=True)
        st.write("Single people are the highest in number and are more likely to default in comparison to married people. Type 'other' may be people in a relationship.")
    
    with tab4:
        with open("correlation_heatmap.json", "r") as f:
            fig1_json = f.read()
            fig1 = pio.from_json(fig1_json)
        st.plotly_chart(fig1, use_container_width=True)
        st.write("Highest positive correlation exists between the BILL_AMT features, where each month's bill statement is correlated with the other months, i.e. if a person spends more in one month, they are like to spend more in the next months. This is followed by the high correlations between the PAY features which represent the repayment status. If a person defaults on one month's payment, they are likely to default on the next as well.")
        st.write("LIMIT_BAL and PAY features have a slight negative correlation, i.e. higher the credit limit, lower is the chance of defaulting on credit repayment. Age and Marriage Status also follow a slight negative correlation where a higher age indicates the client is likely to be married.")
        st.write("")
        st.write("")
        st.write("**Defaults grouped by Amount of Credit Limit**")
        image = Image.open("density_plot.png")  
        col1, col2 = st.columns([3, 1])  
        with col1:
            st.image(image, use_column_width=True) 
        with col2:
            st.write("")
            st.write("")
            st.markdown("""
            As credit limits increase, the density of non-defaulters remains higher relative to defaulters, indicating that non-defaulters tend to have higher credit limits. 
            Defaulters are relatively more frequent in the credit limit range of 0 to 100,000, with the highest being for credit limit 50,000. 
            """)
        st.write("")
        st.write("")
        st.write("**Joint Relationship of Bill Amounts and Payment Amounts Across Defaults**")
        image1 = Image.open("kde_june.png")  
        col1, col2 = st.columns([1, 1])  
        with col1:
            st.image(image1, use_column_width=True) 
        with col2:
            st.write("")
            st.write("")
            st.markdown("""
            No clear pattern can be seen in the default pattern in relation to monthly bill statements and payments made. There is a slight trend of defaults being more concentrated around lower payments amounts, as those making higher monthly payments are more likely to not default. 
            However, defaults observations are scattered and outliers can be seen in the data as well.
            """)
        st.write("")
        st.write("")
        image2 = Image.open("kde_july.png") 
        image4 = Image.open("kde_aug.png")
        col1, col2 = st.columns([1, 1])  
        with col1:
            st.image(image2, use_column_width=True) 
        with col2:
            st.image(image4, use_column_width=True)
        st.write("")
        st.write("")
        st.write("**Credit Repayment History Snapshot**")
        st.write("KEY:") 
        st.write("PAY_1 = Repayment status in Sep, PAY_2 = Repayment status in Aug, ... so on")
        st.write("-2 = No payment due, -1 = Paid duly, 0 = Payment delay <1 month")
        st.write("1 = Payment delay of 1 month, 2 = Payment delay of 2 months, ... so on")
        with open('hiplot.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=800, scrolling=True)
        
    with tab5:
        option = st.radio("Choose a factor", ("Unemployment Rate", "Inflation Rate"))
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=data_macro["Month"], y=data_macro["defaults"], mode='lines', name='Defaults',line=dict(color='orange')),secondary_y=True)
        if option == "Unemployment Rate":
            fig.add_trace(go.Scatter(x=data_macro["Month"], y=data_macro["Unemployment Rate"], mode='lines', name='Unemployment Rate',line=dict(color='green')),secondary_y=False)
            st.write("The count of defaults is steadily rising. When compared with the unemployment rate in Taiwan, the number of defaults increase as unemployment increases. September shows a drastic fall in the unemployment rate and we can expect to see a fall in the default rate as well as more customers will be expected to repay their debts.")
        elif option == "Inflation Rate":
            fig.add_trace(go.Scatter(x=data_macro["Month"], y=data_macro["CPI"], mode='lines', name='Inflation Rate',line=dict(color='green')),secondary_y=False)
            st.write("The count of defaults is steadily rising. When compared with the inflation rate in Taiwan, the number of defaults increase as the inflation rate increases. The buying power of the population is falling and so customers are less likely to repay their credit card payments.")    
        fig.update_layout(xaxis_title='Month',yaxis_title='Rate',yaxis2_title='Defaults')
        st.plotly_chart(fig, use_container_width=True)
        with open("correlation_heatmapmacro.json", "r") as f:
            fig1_json = f.read()
            fig1 = pio.from_json(fig1_json)
        st.plotly_chart(fig1, use_container_width=True)
        st.write("CPI demonstrates a strong positive correlation with defaults, suggesting that higher inflation rates significantly strain finances, leading to increased defaults. Unemployment Rate shows a moderate positive correlation with CPI, indicating that rising inflation often coincides with economic slowdowns and higher unemployment. However, defaults have a weak correlation with Unemployment Rate, implying that inflation (CPI) is likely a more critical driver of defaults.")


elif page == "🖥️Decoding the Algorithm":
    st.subheader("Understanding the Model's Inner Workings")
    st.write("Identifying defaulting customers is the primary focus and XGBoost classifier trained on balanced data is used for this problem.")
    st.write("Nagivate to the tabs below to gain a deeper understanding of the model's structure.")
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Engineering", "Pre-Processing", "Modelling & Results", "Time Series Forecasting for CPI"])
    st.write("")
    
    with tab1:
        st.write("**T-Test for numerical columns**")
        st.write("Evaluated whether a feature significantly affects observed differences. Each bar represents a feature's significance level, with the red line marking the threshold for statistical significance. Bars crossing the line indicate statistically significant features, i.e. meaningful impact on the prediction model.")
        image1 = Image.open("ttest.png")  
        col1, col2 = st.columns([4, 1])  
        with col1:
            st.image(image1, use_column_width=True) 
        with col2:
            st.write("")
            st.write("")
            st.markdown("""
            Most features show statistical significance. BILL_AMT4-6 are highly correlated with other BILL_AMT features and are not significant, so they can be dropped. 
            """)
        st.write("")
        st.write("")
        st.write("**Chi-Square test for Categorical Columns**")
        st.write("Similarly, evaluated whether a categorical feature has a significant association with the outcome.")
        image2 = Image.open("chi-squaretest.png")  
        col1, col2 = st.columns([4, 1])  
        with col1:
            st.image(image2, use_column_width=True) 
        with col2:
            st.write("")
            st.write("")
            st.markdown("""
            All categorical features show statistical significance so won't drop any.
            """)
        st.write("")
        st.write("")
        st.write("**Principle Component Analysis for PAY_AMT1-6 and BILL_AMT1-6**")
        st.write("Since dataset contains 6 sets each of PAY_AMT and BILL_AMT, used PCA to reduce this high-dimensional data to a lower-dimensional space, where each point represents an observation.")
        st.write("First, found the optimal number of principle components using Scree plot.")
        image2 = Image.open("Scree plot.png")
        st.image(image2, use_column_width=True)
        st.write("Since the first six principal components account for approximately 90% of the dataset's variance, the dimensionality can be reduced to these components. This approach retains maximum information while minimizing redundancy. While five PCs might simplify the model further, it risks losing subtle yet potentially valuable patterns present in the additional variance explained by the sixth component.")
        st.write("Next, dimensionality was reduced to six principal components.")
        with open("pca_plot.json", "r") as f:
            fig5_json = f.read()
            fig5 = pio.from_json(fig5_json)
        st.plotly_chart(fig5, use_container_width=True)
        st.write("The significant overlap between Default_binary classes (defaulted vs. non-defaulted), even with six principal components, suggests that bill and payment features lack the discriminative power to predict default risk effectively. Additional or alternative features may be needed.")


    with tab2:
        st.write("Scaled dataset was split in 70-30 ratio for training and testing data for model evaluation.")
        st.write("**Class Imbalance**")
        image3 = Image.open("class_imbalance.png")  
        col1, col2 = st.columns([1.5, 1])  
        with col1:
            st.image(image3, use_column_width=True)  
        with col2:
            st.write("")
            st.write("")
            st.markdown("""
            The plot illustrates the distribution of classes in the training set. 
            The majority class "not default" has 77% samples, whereas, the minority class "default" has 22% samples. Data is highly imbalanced. 
            Addressing this imbalance is vital to improve model performance.
            """)
        st.write("")
        st.write("")
        st.write("**Dataset Transformation After SMOTE**")
        image4 = Image.open("imbalance_smote.png")  
        col1, col2 = st.columns([1.5, 1])  
        with col1:
            st.image(image4, use_column_width=True)  
        with col2:
            st.write("")
            st.write("")
            st.markdown("""
            Both classes are now balanced in the training set, i.e. we have an equal count of observations for default and non-default cases.
            """)
        st.write("")
        st.write("SMOTE made an impact on the shape of the histograms since the smoothed-out line now follows an exaggerated pattern for the default class, i.e. the mean, median and mode have changed. There is an increase in the count for some values of LIMIT_BAL, but not relative to the proportion of the difference in the count for each rectangle, which has raised the height of the histograms. So the values now occur more frequently than before.")
        st.write("")
        image5 = Image.open("distribution_smote.png")
        st.image(image5, width=800)

    with tab3:
        st.write("For comprehensive results, the performance of three machine learning models is evaluated using both imbalanced and balanced training datasets.")
        st.write("**Best Model Overall:** XGBoost Classifier for accurately distinguising between classes and having the highest precision for predicting defaulting customers. Though, it struggles with recall for class 1, balanced data achieves a better balance between precision and recall for class 1.")
        st.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        st.write("**I. Logistic Regression**")
        st.latex(r'''P(y=1|x) = \frac{1}{1+e^{-(A+Bx)}}''')
        st.write("Rantionale: Uses the logistic (sigmoid) function to output probabilities that map to binary outcomes, making it more suitable than linear regression.")
        st.write("Model parameters: 'l2' penalty helps regularize the model to prevent overfitting, class_weight='balanced' adjusts for imbalanced classes by giving more weight to the minority class (defaults), and GridSearchCV with different values of C helps optimize the regularization strength to control the trade-off between model complexity and fit to data.")
        st.write("Imbalanced data results: Model accurately predicted 67% observations. Precision=0.36 for class 1 (defaults) is very low, meaning it falsely labels non-defaulting customers as defaulting. Recall=0.65 for class 1 is decent but still misses 35% of actual defaulting clients.")
        st.write("Balanced data results: Model accurately predicted 67% observations. Performance metrics are nearly identical to imbalanced results, with precision and recall slightly reduced for class 1. SO balancing data did not have a significant impact on model.")
        col1, col2 = st.columns([1.5, 1]) 
        with col1:
            imageida1 = Image.open("Logistics_reg.png")
            st.image(imageida1, width=350)
        with col2:
            st.write("AUC=0.72 indicates that the model is moderately good at distinguishing between the two classes. This is same for imbalanced and balanced sets.")
        
        st.write("**II. Support Vector Machine**")
        st.write("Rantionale: Effective in high-dimensional spaces and works well for classification tasks with clear margins.")
        st.write("Model parameters: Radial Basis Function kernel handles non-linear relationships well (exists in this data), low gamma (0.01) ensures the decision boundary is not too sensitive, avoiding overfitting, and C=1 balances the trade-off between maximizing the margin and minimizing classification errors.")
        st.write("Imbalanced data results: Model accurately predicted 82% observations. Precision=0.66 for class 1 is better than Logistic Regression, i.e. predicts defaulting customers more accurately. Recall=0.32 for class 1 is low indicating a big portion of defaulting customers are not identified.")
        col1, col2 = st.columns([1.5, 1]) 
        with col1:
            imageida1 = Image.open("SVM_imbalanced.png")
            st.image(imageida1, width=350)
        with col2:
            st.write("AUC=0.64 is lower than Logistic Regression so SVM struggles to differentiate between the classes as effectively.")
        st.write("Balanced data results: Model accurately predicted 78% observations. Precision=0.49 for class 1 drops, so model is predicting defaulting customers less accurately than before. Recall=0.56 for class 1 improves compared to imbalanced set, indicating model captures a higher number of defaulting customers.")
        col1, col2 = st.columns([1.5, 1]) 
        with col1:
            imageida1 = Image.open("SVM_balanced.png")
            st.image(imageida1, width=350)
        with col2:
            st.write("AUC=0.70 improves for balanced data so model is better able to separate the classes than using imbalanced data.")
        
        st.write("**III. XGBoost Classifier**")
        st.write("Rantionale: Powerful gradient boosting algorithm that models complex, non-linear relationships in data by building an ensemble of weak learners (decision trees) and optimizes the prediction through boosting.")
        st.write("Model parameters: Learning rate=0.01 ensures gradual learning to avoid overfitting, Colsample_bytree=0.8 and Subsample=0.8 introduce randomness, Max depth=4 controls the depth of individual trees to avoid overly complex model, Gamma=5 adds regularization to control complexity, and Scale_pos_weight=1 balances weight of both classes in imbalanced datasets.")
        st.write("Imbalanced data results: Model accurately predicted 81% observations. Precision=0.72 for class 1 is highest among all models so this is most accurate for predicting defaulting customers. Recall=0.24 for class 1 is very low so the model misclassifies a large portion of defaulting customers.")
        st.write("Balanced data results: Model accurately predicted 79% observations. Precision=0.51 for class 1 drops compared to imbalanced data, but still outperforms Logistic Regression and SVM. Recall=0.58 for class 1 improves so the model catches a larger portion of defaultign customers than using the imbalanced set.")
        col1, col2 = st.columns([1.5, 1]) 
        with col1:
            imageida1 = Image.open("XGBoost.png")
            st.image(imageida1, width=350)
        with col2:
            st.write("AUC=0.77 is highest among all model indictaing that this model is the best at distinguishing between the classes. This is same for imbalanced and balanced sets.")

    with tab4:
        st.write("Given the strong positive correlation of 0.84 between CPI and defaults observed during the EDA, understanding future CPI trends can enable us to anticipate potential changes in default percentages and proactively adjust strategies to minimize financial risks.")
        st.write("**Time Series Decomposition**")
        st.write("This plot represents Taiwanese CPI training data from Jan 2000 to Sep 2005. CPI shows a clear upward trend, indicating that the series is not stationary (mean and variance are not constant over time). Seasonal variations may exist.")
        imageida1 = Image.open("CPI_dist.png")
        st.image(imageida1, use_column_width=True)
        st.write("To further model these variations, STL Decomposition is employed.")
        col1, col2 = st.columns([1.5, 1]) 
        with col1:
            imageida1 = Image.open("STL.png")
            st.image(imageida1, width=400)
        with col2:
            st.write("The trend component highlights a steady increase in CPI, indicating long-term growth. The seasonal component shows regular fluctuations, suggesting seasonal patterns in the data. The residual component captures irregularities not explained by trend or seasonality. These are relatively small, suggesting that majority of the variability in the time series is explained by the trend and seasonal components.")
        st.write("")
        st.write("**Pre-processing**")
        st.write("Autoregressive Integrated Moving Average, i.e. ARIMA(p, d, q) model is employed, which requires differencing the series to remove trend and achieve stationarity.")
        st.write("To identify p (Autoregressive (AR) Order) and q (Moving Average (MA) Order), ACF and PACF plots are evaluated. The PACF plot shows the correlation of the time series with its lagged values after removing the effects of intermediate lags and the ACF plot shows the overall correlation of the time series with its lagged values.")
        imageida1 = Image.open("PACF.png")
        st.image(imageida1, use_column_width=True)
        st.write("ACF plot displays a high initial positive lag (at lag 1), indicating strong autocorrelation, confirming a non-stationary series. Gradual decay suggests that differencing might be needed to make the series stationary. The last significant spike is at lag 3 so a q=MA(3) component might be suitable.")
        st.write("PACF plot displays the last significant spike at lag 1 and diminishing spikes afterward, suggesting that an p=AR(1) component might be suitable.")
        st.write("Stationarity is evaluated using the Augmented Dickey-Fuller Test. The test concludes that the series becomes stationary after the first difference so d=1.")
        st.write("")
        st.write("**Modelling**")
        st.write("ARIMA(1,1,3) model is hence used to forecast CPI for the next 3 months, i.e. Oct, Nov, Dec 2005.")
        imageida1 = Image.open("forecast.png")
        st.image(imageida1, use_column_width=True)
        st.write("The model, despite yielding a relatively low RMSE of 1.1356, fails to accurately capture the data's structure. While the small RMSE suggests low prediction errors, this might be due to nonlinear relationships and noise in the data that the ACF/PACF plots did not account for. Even higher-order ARIMA models fail to adequately model the underlying relationship effectively. Hence, a more optimal method would have to be employed.")
