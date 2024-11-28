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

#add navigation sidebar
st.sidebar.title("🔎Explore")
page = st.sidebar.selectbox("Select a page:", ["🏡Home", "👪Demographic Data", "💳Credit Limit & Balance", "📊Macroeconomic Factors","🤑Sep Defaulters","🖥️Decoding the Algorithm"], index=0)
for _ in range(15):  # Change 10 to the number of empty lines you want
    st.sidebar.write("")
st.sidebar.write("View the code and dataset details: https://github.com/mahnoorsheikh16/FDS-Project-CMSE830-Credit-Card-Default-Prediction")

#import data
data = pd.read_csv("https://raw.githubusercontent.com/mahnoorsheikh16/FDS-Project-CMSE830-Credit-Card-Default-Prediction/refs/heads/main/UCI_Credit_Card.csv")
data_macro = pd.read_excel("https://raw.githubusercontent.com/mahnoorsheikh16/FDS-Project-CMSE830-Credit-Card-Default-Prediction/main/data_macro.xlsx")
data_income = pd.read_excel("https://raw.githubusercontent.com/mahnoorsheikh16/FDS-Project-CMSE830-Credit-Card-Default-Prediction/main/data_income.xlsx")

data.rename(columns={'default.payment.next.month': 'Default'}, inplace=True)
data.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)

#set page content
if page == "🏡Home":
    st.title('Beyond the Client')
    st.subheader("Enhancing Credit Card Default Prediction with Behavioral Analysis and Macroeconomic Data")
    st.write("")
    st.write("")
    st.write("The Bank of Taiwan is facing losses due to increasing credit card defaults. This poses a major concern as defaults can result in significant losses and, in severe cases, even bankruptcy. This dashboard provides a means to identify factors associated with credit card defaults and help identify potential faulty clients to curb losses.")
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

elif page == "👪Demographic Data":
    st.subheader("Defaults in relation to Gender, Relationship Status, Age, & Education Level")
    st.write("")
    st.write("")
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
    
elif page == "💳Credit Limit & Balance":
    st.subheader("Defaults in relation to Credit Limit and Monthly Repayment History")
    st.write("")
    st.write("")
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
    st.write("")
    st.write("")
    st.write("**Principle Component Analysis for PAY_AMT1-6 and BILL_AMT1-6**")
    st.write("PCA is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space. Each point represents an observation in the reduced space, with axes corresponding to principal components that capture the most significant patterns in the data, revealing clusters or trends among the observations.")
    with open("pca_plot.json", "r") as f:
        fig5_json = f.read()
        fig5 = pio.from_json(fig5_json)
    st.plotly_chart(fig5, use_container_width=True)
    st.write("There is significant overlap between the Default_binary classes (e.g., no clear separation between defaulted and non-defaulted groups). This might imply that the first two PCA components do not strongly distinguish between those who defaulted and those who did not. This could be a sign that the features (bill and payment amounts) alone do not provide a strong signal for predicting default risk, at least in the two-dimensional space captured by PCA.")

elif page == "📊Macroeconomic Factors":
    st.subheader("Defaults in relation to Unemployment Rate and Inflation")
    st.write("")
    st.write("")
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
    

elif page == "🤑Sep Defaulters":
    #show raw table of those who will default
    st.subheader("Customers Predicted to Default Next Month")
    st.write("")
    st.write("")
    data1 = data[data['Default'] == 'yes']
    rows_per_page = 10
    total_pages = (len(data1) // rows_per_page) + (1 if len(data1) % rows_per_page > 0 else 0)
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    start_row = st.session_state.current_page * rows_per_page
    end_row = start_row + rows_per_page
    if start_row < len(data1):
        st.dataframe(data1.iloc[start_row:end_row])
    st.write("Page:", st.session_state.current_page + 1, "of", total_pages)
    if st.session_state.current_page > 0:
        if st.button("Previous"):
            st.session_state.current_page -= 1
    if st.session_state.current_page < total_pages - 1:
        if st.button("Next"):
            st.session_state.current_page += 1

elif page == "🖥️Decoding the Algorithm":
    st.subheader("Understanding the Model's Inner Workings")
    st.write("Machine learning models are deployed to identify risky customers and minimise lenders' losses. By using algorithms to study historical transactions and customer demographics, we apply the findings to future customers, effectively distinguishing between risky and non-risky profiles.")
    st.write("")
    st.write("")
    st.write("**T-Test for numerical columns**")
    st.write("A t-test is used to determine if a particular feature significantly contributes to the differences observed in the data. The horizontal bar chart displays the test results, with each bar representing a feature and its significance level. The red line marks the threshold of statistical significance. If a bar crosses this red line, it indicates that the feature is statistically significant, meaning it has a meaningful impact on the prediction model. Features that do not cross this line are considered less relevant.")
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
    st.write("A chi-square test is used to determine if there is a significant association between categorical features and the outcome variable. The horizontal bar chart displays the results of the chi-square test, with each bar representing a feature and its level of significance. The red line marks the threshold of statistical significance. If a bar crosses this red line, it indicates that the feature is statistically significant, meaning it has a strong association with the outcome. Features that do not cross this line are considered less relevant for the prediction model.")
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
    st.write("**Class Imbalance**")
    image3 = Image.open("class_imbalance.png")  
    col1, col2 = st.columns([1.5, 1])  
    with col1:
        st.image(image3, use_column_width=True)  
    with col2:
        st.write("")
        st.write("")
        st.markdown("""
        The plot illustrates the distribution of classes in the dataset. 
        The majority class "not default" has 77% samples, whereas, the minority class "default" has 22% samples. Data is highly imbalanced. 
        Addressing this imbalance is vital to improve model performance. The best method would be to undersample the majority class using one-class classification. As we have not 
        studied it yet, I apply SMOTE for demonstration purposes.
        """)
    st.write("")
    st.write("")
    st.write("**Dataset Transformation After SMOTE**")
    st.markdown('<p style="font-size:12px; color:gray;">To understand how SMOTE was applied, visit the GitHub link.</p>', unsafe_allow_html=True)
    image4 = Image.open("imbalance_smote.png")  
    col1, col2 = st.columns([1.5, 1])  
    with col1:
        st.image(image4, use_column_width=True)  
    with col2:
        st.write("")
        st.write("")
        st.markdown("""
        Both classes are now balanced, i.e. we have an equal count of observations for default and non-default cases.
        """)
    st.write("")
    st.write("SMOTE made an impact on the shape of the histograms since the smoothed-out line now follows an exaggerated pattern for the default class, i.e. the mean, median and mode have changed. There is an increase in the count for some values of LIMIT_BAL, but not relative to the proportion of the difference in the count for each rectangle, which has raised the height of the histograms. So the values now occur more frequently than before.")
    st.write("")
    image5 = Image.open("distribution_smote.png")
    st.image(image5, width=800)

