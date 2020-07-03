import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
# This is followed by checking some dataset, its shape and statistics. As can be seen there are 4 columns and 21293 rows.
data = pd.read_csv(
    ' https://www.kaggle.com/sulmansarwar/transactions-from-a-bakery')
data.head()
# %% [code]
data.shape
# %% [code]
data.describe()
# %% [code]
data.shape
# %% [code] {"scrolled":true}
data.info()
# While exploring this dataset, I found that although there was no evident null value, some of the items (786) were labeled as 'NONE'. So, I removed these items from the data.
data.loc[data['Item'] == 'NONE', :].count()
data = data.drop(data.loc[data['Item'] == 'NONE'].index)
# Next question that comes up is- how many items is this bakery selling? And the answer is thatthis bakery menu contains 94 items and the best seller among them is Coffee.
data['Item'].nunique()
data['Item'].value_counts().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(6, 4))
data['Item'].value_counts().sort_values(
    ascending=False).head(10).plot(kind='bar')
plt.ylabel('Number of transactions')
plt.xlabel('Items')
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.title('Best sellers')
# Next interesting thing to find out is that when is the bakery doing the most business duringthe day. Here, it seems that the bakery is most busy during afternoon and morning and has littlebusiness during evening and night.
data.loc[(data['Time'] < '12:00:00'), 'Daytime'] = 'Morning'
data.loc[(data['Time'] >= '12:00:00') & (
    data['Time'] < '17:00:00'), 'Daytime'] = 'Afternoon'
data.loc[(data['Time'] >= '17:00:00') & (
    data['Time'] < '21:00:00'), 'Daytime'] = 'Evening'
data.loc[(data['Time'] >= '21:00:00') & (
    data['Time'] < '23:50:00'), 'Daytime'] = 'Night'
fig, ax = plt.subplots(figsize=(6, 4))
sns.set_style('darkgrid')
data.groupby('Daytime')['Item'].count().sort_values().plot(kind='bar')
plt.ylabel('Number of transactions')
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.title('Business during the day')
# During its nearly 6 months in business, this bakery has sold over 11569 items during afternoonand only 14 items during night hours.
data.groupby('Daytime')['Item'].count().sort_values(ascending=False)
# For further analysis, I needed to extract month and day from the dataset which is done asshown below.
data['Date_Time'] = pd.to_datetime(data['Date']+' '+data['Time'])
data['Day'] = data['Date_Time'].dt.day_name()
data['Month'] = data['Date_Time'].dt.month
data['Month_name'] = data['Date_Time'].dt.month_name()
data['Year'] = data['Date_Time'].dt.year
data['Year_Month'] = data['Year'].apply(str)+' '+data['Month_name'].apply(str)
data.drop(['Date', 'Time'], axis=1, inplace=True)
data.index = data['Date_Time']
data.index.name = 'Date'
data.drop(['Date_Time'], axis=1, inplace=True)
data.head()
# The plot shows the performance of bakery during different months of its short existence.October and April showed less business which was due to few number of operational days inthese months- 2 and 7 respectively.
# %% [code] {"scrolled":false}
data.groupby('Year_Month')['Item'].count().plot(kind='bar')
plt.ylabel('Number of transactions')
plt.title('Business during the past months')
data.loc[data['Year_Month'] == '2016 October'].nunique()
data.loc[data['Year_Month'] == '2017 April'].nunique()
# Next, I was interested in finding out monthly bestseller. This table below shows not only theitem that has maximum buyers but one can also check how many quantities of their items ofinterest were sold . As expected, coffee is the topseller in all the months.
data2 = data.pivot_table(index='Month_name', columns='Item', aggfunc={
                         'Item': 'count'}).fillna(0)
data2['Max'] = data2.idxmax(axis=1)
data2
# Here, I checked for the daytime bestseller. Coffee top the charts during morning, afternoonand evening, but for obvious reasons it is not the favourite during night. Vegan feast is the bestseller for nights.
data3 = data.pivot_table(index='Daytime', columns='Item', aggfunc={
                         'Item': 'count'}).fillna(0)
data3['Max'] = data3.idxmax(axis=1)
data3
# As expected, Coffee is the best seller from Monday to Sunday.
data4 = data.pivot_table(index='Day', columns='Item',
                         aggfunc={'Item': 'count'}).fillna(0)
data4['Max'] = data4.idxmax(axis=1)
data4
# I was curious about the business growth of this bakery. For that I have plotted some line plots.As observed above in the barplot, November showed maximum business for the bakery,followed by February and March, with a dip shown for December and January.
data['Item'].resample('M').count().plot()
plt.ylabel('Number of transactions')
plt.title('Business during the past months')
# The next plot shows weekly performance of the bakery. A big dip in business is shown aroundend of December and start of first week of January
data['Item'].resample('W').count().plot()
plt.ylabel('Number of transactions')
plt.title('Weekly business during the past months')
# I zoomed in to the daily performance of the bakery and found that there have been daysaround December end and January beginning when the bakery sold 0 item.
data['Item'].resample('D').count().plot()
plt.ylabel('Number of transactions')
plt.title('Daily business during the past months')
data['Item'].resample('D').count().min()
# During the most profitable day the bakery could sell around 292 items and this happened inFebruary (as seen in the daily graph above).
data['Item'].resample('D').count().max()
# ### Apriori Algorithm and Association Rule:
# Next, I plan to perform an association rule analysis which gives an idea about how things areassociated to each other. The common metrics to measure association are-
# 1. Support- It is the measure of frequency or abundance of an item in a dataset. It can be'antecedent support', 'consequent support', and 'support'. 'antecedent support' containsproportion of transactions done for the antecedent while 'consequent support' involves thosefor consequent. 'Support' is computed for both antecedent and consequent in question.
# 2. Confidence-This gives the probability of consequent in a transaction given the presence ofantecedent.
# 3. Lift- Given that antecedents and consequents are independent, how often do they cometogether/bought together.
# 4. Leverage- It is the difference between frequency of antecedent and consequent together intransactions to frequency of both in independent transactions.
# 5.Conviction- A higher conviction score means that consequent is highly dependent on

# Apriori algorithm is used to extract frequent itemsets that are further used for association ruleanalysis. In this algorithm, user defines a minimum support that is the minimum threshold thatdecides if an itemset is considered as 'frequent'.
# To begin with association rule analysis, I made a dataset that contains lists of items that arebought together.
lst = []
for item in data['Transaction'].unique():
    lst2 = list(set(data[data['Transaction'] == item]['Item']))
    if len(lst2) > 0:
        lst.append(lst2)
print(lst[0:3])
print(len(lst))
# For Apriori algorithm, this dataset needs to be one-hot encoded. This is done usingTransactionEncoder as shown here, followed by apriori algorithm to get the frequent itemsets.Then association rules function is used which can take any metric. Here I have used 'lift' andspecified minimum threshold as 1.
te = TransactionEncoder()
te_data = te.fit(lst).transform(lst)
data_x = pd.DataFrame(te_data, columns=te.columns_)
print(data_x.head())
frequent_items = apriori(data_x, use_colnames=True, min_support=0.03)
print(frequent_items.head())
rules = association_rules(frequent_items, metric="lift", min_threshold=1)
rules.antecedents = rules.antecedents.apply(lambda x: next(iter(x)))
rules.consequents = rules.consequents.apply(lambda x: next(iter(x)))
rules

fig, ax = plt.subplots(figsize=(10, 4))
GA = nx.from_pandas_edgelist(rules, source='antecedents', target='consequents')
nx.draw(GA, with_labels=True)
plt.show()
