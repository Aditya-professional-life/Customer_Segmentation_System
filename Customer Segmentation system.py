#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Correct the typo here: change "df-pd.read_csv" to "df = pd.read_csv"
df = pd.read_csv("Mall_Customers.csv")
df.head()


# In[ ]:





# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


df.dtypes


# In[6]:


df.isnull().sum()


# In[7]:


df.drop(["CustomerID"],axis = 1, inplace = True)
df.head()


# In[13]:


plt.figure(1, figsize=(15, 6))
n = 0
for x in ["Age", "Annual Income (k$)", "Spending Score (1-100)"]:  # Corrected the column names
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.distplot(df[x], bins=20)  # Changed from sns.displot to sns.histplot
    plt.title("Distplot of {}".format(x))
plt.show()


# In[14]:


plt.figure(figsize =(15,5) )
sns.countplot(y="Gender",data = df)
plt.show()


# In[15]:


plt.figure(1, figsize=(15, 7))  # Corrected the syntax error here

n = 0

for col in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:  # Corrected the column names
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.set(style="whitegrid")  # Corrected the syntax error here
    sns.violinplot(x=col, y='Gender', data=df)  # Changed 'Gender •' to 'Gender'
    plt.ylabel('Gender' if n == 1 else '')  # Corrected the syntax error here
    plt.title('Violin Plot of {}'.format(col))

plt.show()







# In[17]:


age_18_25 = df[df['Age'].between(18, 25)]
age_26_35 = df[df['Age'].between(26, 35)]
age_36_45 = df[df['Age'].between(36, 45)]
age_46_55 = df[df['Age'].between(46, 55)]
age_above_55 = df[df['Age'] > 55]

# Define age group labels
agex = ["18-25", "26-35", "36-45", "46-55", "55+"]

# Calculate the number of customers in each age group
agey = [
    len(age_18_25.values),
    len(age_26_35.values),
    len(age_36_45.values),
    len(age_46_55.values),
    len(age_above_55.values)
]

# Create a bar plot
plt.figure(figsize=(15, 6))
sns.barplot(x=agex, y=agey, palette="mako")  # Corrected the palette parameter
plt.title("Number of Customers in Different Age Groups")
plt.xlabel("Age Group")
plt.ylabel("Number of Customers")
plt.show()


# In[19]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=df)
plt.title("Scatter Plot of Annual Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()


# In[22]:


ss_1_20 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 1) & (df["Spending Score (1-100)"] <= 20)]
ss_21_40 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 21) & (df["Spending Score (1-100)"] <= 40)]
ss_41_60 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 41) & (df["Spending Score (1-100)"] <= 60)]
ss_61_80 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 61) & (df["Spending Score (1-100)"] <= 80)]
ss_81_100 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 81) & (df["Spending Score (1-100)"] <= 100)]

# Define score ranges
ssx = ["1-20", "21-40", "41-60", "61-80", "81-100"]

# Count the number of customers in each score range
ssy = [
    len(ss_1_20.values),
    len(ss_21_40.values),
    len(ss_41_60.values),
    len(ss_61_80.values),
    len(ss_81_100.values)
]

# Create a bar plot
plt.figure(figsize=(15, 6))
sns.barplot(x=ssx, y=ssy, palette="rocket")
plt.title("Spending Scores")
plt.xlabel("Score Range")
plt.ylabel("Number of Customers Having the Score")
plt.show()


# In[23]:


ai0_30 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 0) & (df["Annual Income (k$)"] <= 30)]
ai31_60 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 31) & (df["Annual Income (k$)"] <= 60)]
ai61_90 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 61) & (df["Annual Income (k$)"] <= 90)]
ai91_120 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 91) & (df["Annual Income (k$)"] <= 120)]
ai121_150 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 121) & (df["Annual Income (k$)"] <= 150)]

# Define income ranges
aix = ["$0 - $30,000", "$30,001 - $60,000", "$60,001 - $90,000", "$90,001 - $120,000", "$120,001 - $150,000"]

# Count the number of customers in each income range
aiy = [
    len(ai0_30.values),
    len(ai31_60.values),
    len(ai61_90.values),
    len(ai91_120.values),
    len(ai121_150.values)
]

# Create a bar plot
plt.figure(figsize=(15, 6))
sns.barplot(x=aix, y=aiy, palette="Spectral")
plt.title("Annual Income")
plt.xlabel("Income Range")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[24]:


from sklearn.cluster import KMeans

# Assuming you have a DataFrame named 'df'

X1 = df.loc[:, ["Age", "Spending Score (1-100)"]].values

wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss, linewidth=2, color="red", marker="8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.show()


# In[25]:


kmeans = KMeans(n_clusters  = 4)
label = kmeans.fit_predict(X1)
print(label)


# In[26]:


print(kmeans.cluster_centers_)


# In[30]:


plt.scatter(X1[:,0], X1[:,1],c = kmeans.labels_,cmap = "rainbow")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color = "black")
plt.title("Clusters of Customers")
plt.xlabel("Age")
plt.ylabel("Spending Score(1-100)")
plt.show()


# In[31]:


X2 = df.loc[:, ["Annual Income (k$)", "Spending Score (1-100)"]].values

wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X2)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss, linewidth=2, color="red", marker="8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.show()


# In[33]:


kmeans = KMeans(n_clusters=5)  # Specify the number of clusters
labels = kmeans.fit_predict(X2)

print(labels)


# In[34]:


print(kmeans.cluster_centers_)


# In[35]:


plt.scatter(X2[:,0], X2[:,1],c = kmeans.labels_,cmap = "rainbow")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color = "black")
plt.title("Clusters of Customers")
plt.xlabel("Annual Income(K$)")
plt.ylabel("Spending Score(1-100)")
plt.show()


# In[38]:


X3 = df.iloc[:,1:]
wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X3)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss, linewidth=2, color="red", marker="8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.show()


# In[39]:


kmeans = KMeans(n_clusters=5)  # Specify the number of clusters
labels = kmeans.fit_predict(X3)

print(labels)


# In[40]:


print(kmeans.cluster_centers_)


# In[45]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

# Assuming you have a DataFrame named 'df' and X3 is defined as mentioned before
X3 = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].values

kmeans = KMeans(n_clusters=5)  # Specify the number of clusters
clusters = kmeans.fit_predict(X3)
df["label"] = clusters

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection="3d")

# Define colors for each cluster
colors = ["blue", "red", "green", "purple", "orange"]

# Scatter points for each cluster
for i in range(5):
    ax.scatter(
        df["Age"][df["label"] == i],
        df["Annual Income (k$)"][df["label"] == i],
        df["Spending Score (1-100)"][df["label"] == i],
        c=colors[i],
        s=60,
        label=f"Cluster {i}"
    )

ax.view_init(30, 185)

ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")

plt.legend()
plt.show()


# In[ ]:




