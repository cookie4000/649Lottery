import pandas as pd;
import matplotlib.pyplot as plot;
import scipy.stats as stats

# Import the 6/49 Data 
# Data From https://www.kaggle.com/datasets/datascienceai/lottery-dataset
df_draws = pd.read_csv('data/649.csv')


##################
## Clean up Data
##################

# Filter to Columns of Interest
df_drawsSubset = df_draws[['DRAW DATE','NUMBER DRAWN 1', 'NUMBER DRAWN 2', 'NUMBER DRAWN 3', 'NUMBER DRAWN 4', 'NUMBER DRAWN 5', 'NUMBER DRAWN 6']]

# Change Date to Date Type
df_drawsSubset['DRAW DATE'] = pd.to_datetime(df_drawsSubset['DRAW DATE'], format='%m/%d/%Y')

######################
## Frequency Analysis
######################

# Get Frequency of Each Ball (get frequency of each ball and union togeher)
df_ballFreqBall1=df_drawsSubset.groupby('NUMBER DRAWN 1').agg(count=('NUMBER DRAWN 1', 'count'))
df_ballFreqBall2=df_drawsSubset.groupby('NUMBER DRAWN 2').agg(count=('NUMBER DRAWN 2', 'count'))
df_ballFreqBall3=df_drawsSubset.groupby('NUMBER DRAWN 3').agg(count=('NUMBER DRAWN 3', 'count'))
df_ballFreqBall4=df_drawsSubset.groupby('NUMBER DRAWN 4').agg(count=('NUMBER DRAWN 4', 'count'))
df_ballFreqBall5=df_drawsSubset.groupby('NUMBER DRAWN 5').agg(count=('NUMBER DRAWN 5', 'count'))
df_ballFreqBall6=df_drawsSubset.groupby('NUMBER DRAWN 6').agg(count=('NUMBER DRAWN 6', 'count'))


# Union all the frequencies
df_allFreq= pd.concat([df_ballFreqBall1, df_ballFreqBall2,df_ballFreqBall3,df_ballFreqBall4,df_ballFreqBall5,df_ballFreqBall6])
df_allFreq.index.names = ['ball']

# sum the values from the unioned table 
df_frequencyPerBall=df_allFreq.groupby('ball').agg(frequency=('count', 'sum'))
df_frequencyPerBall.to_csv("out.csv")

# Whats the most common ball during this time?
#print(df_frequencyPerBall.sort_values(by='frequency', ascending=False)) # Ball 31 with 499 appearences

# Frequency Graph
df_frequencyPerBall.plot.bar(rot=15, title="Frequency Per Lottery Ball)");
plot.show(block=True);

# Restructure data frame to give us the Expected/Observed table shown on the blog - use crosstab
df_frequencyPerBall.reset_index(inplace=True)
df_expected = df_frequencyPerBall[['ball']].copy()
df_expected['value']=449
df_expected['type'] = 'Expected'
df_expected = df_expected[['ball','value', 'type']]
df_observed = df_frequencyPerBall[['ball','frequency']].copy()
df_observed['type'] = 'Observed'
df_observed.rename(columns = {'frequency':'value'}, inplace = True)

# Union observed and Expected
df_toPivot= pd.concat([df_expected,df_observed])

# Crosstab
df_crosstab=pd.crosstab(index = df_toPivot.type, columns = df_toPivot.ball,values =df_toPivot.value, aggfunc = 'sum')
#print(df_crosstab)

#############################################
## Prepare Data for Chi Squared Testing
## Get 2 Arrays - Observed and Expected
##############################################

df_chi = df_frequencyPerBall
df_chi['expected']=448.7755102 
df_chi.rename(columns = {'frequency':'observed'}, inplace = True)

# Convert columns to Arrays
expected=df_chi['expected'].to_numpy()
observed=df_chi['observed'].to_numpy()
 
# chi squared (goodness of fit)
print(stats.chisquare(observed, expected))
# statistic=47.927148704392245
# pvalue= 0.47580793479137123
# As pvalue IS NOT < 0.05 (95% confidence) - We Accept H0
