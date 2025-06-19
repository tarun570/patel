import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")
df=pd.read_csv("C:\\Users\\HP\\Downloads\\Bank Customer Churn Prediction.csv")
df
df.info()
df["customer_id"]
df["credit_score"]
df["gender"].unique()
df["gender"].value_counts()
df["credit_card"]
df["credit_card"].value_counts()
df["estimated_salary"]
df["balance"]
df["balance"].value_counts()
df["churn"]
df["churn"].value_counts()

continous= ['balance','estimated_salary']
discrete_categorical= ['country','gender']
discrete_count= ['customer_id ','credit_score','age','tenure', 'products_number',
                'credit_card','active_member','churn']
df[continous].describe()

df[continous].skew()

df[discrete_categorical].describe()

df.isnull().sum()

sns.histplot(df['balance'],bins=15,kde=True)
plt.show()

sns.kdeplot(df['balance'])
plt.show()

sns.boxplot(x=df["estimated_salary"])
plt.show()

sns.boxplot(y=df["balance"])
plt.show()

sns.scatterplot(x=df["balance"],y=df["estimated_salary"])
plt.show()

sns.scatterplot(x=df["balance"],y=df["estimated_salary"],hue=df["gender"])
plt.show()


sns.relplot(x=df['balance'],y=df['estimated_salary'],col=df['customer_id'],col_wrap=2,
            hue=df["gender"])
plt.show()

df['sno'] = pd.DataFrame(np.arange(1,10001))
df

sns.relplot(x = 'sno', y = 'estimated_salary',data = df,kind='line')
plt.show()

df.drop("sno",axis=1,inplace=True)

sns.jointplot(x="balance",y="estimated_salary",data=df)
plt.show

sns.violinplot(x="balance",y="estimated_salary",data=df)
plt.show()

sns.pairplot(df,vars=continous)
plt.show()

c_m = df[continous].corr()
c_m

sns.heatmap(c_m,annot=True)
plt.show()

df["country"].unique()

df["country"].value_counts()

sns.countplot(x="country",data=df)
plt.show()

sns.countplot(y=df["country"])
plt.show()

df.groupby("gender")["balance"].mean()

sns.catplot(x='gender',y='balance',data=df,kind="bar")
plt.show()

df.groupby("country")["balance"].mean()

sns.catplot(x='country',y='balance',data=df,kind="bar")
plt.show()

sns,catplot(x="products_number", y="balance",data=df,kind = 'box')
plt.show()
