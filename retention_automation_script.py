import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# from sklearn import preprocessing
# from pycaret import classification
# import matplotlib.pyplot as plt 
# plt.rc("font", size=14)
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import sklearn.metrics as metrics
# import seaborn as sns
# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)
from pycaret.classification import *
import pyodbc
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=10.27.251.83;'
                      'Database=OpenBook;'
                      'Trusted_Connection=yes;')

# Read in the retention model SQL output
retention_query = open(r'C:\Users\dmcdonnell2\OneDrive - City Colleges of Chicago\Python Projects\git\retention_model\retention_model_obdata.sql', 'r')
data_current = pd.read_sql_query(retention_query.read(),conn)
retention_query.close()

# Scale placement test scores to normalize, then create a variable for highest placement score
sc_X = StandardScaler()
sc_X = sc_X.fit_transform(data_current[['aleks', 'sat_english', 'sat_math', 'act_composite', 'cccrtw']])
sc_X = pd.DataFrame(data=sc_X, columns=['aleks_scaled', 'sat_english_scaled', 'sat_math_scaled', 'act_composite_scaled', 'cccrtw_scaled'])
data_scaled = pd.concat([data_current, sc_X], axis = 1)
data_scaled['max_placement_score'] = data_scaled[['aleks_scaled', 'sat_english_scaled', 'sat_math_scaled', 'act_composite_scaled', 'cccrtw_scaled']].max(axis=1)


final_model = load_model('fa22_retention_model_20220725')

current_set = predict_model(final_model, raw_score = True, data = data_scaled)

# Assign percentiles by home college based on retention probability
current_set['hc_percentile'] = current_set.groupby(['Home_College'])['Score_1'].rank(pct=True)
def tiers(df):
    if (df['Home_College'] == 'DA' and df['hc_percentile'] < .22)\
    or (df['Home_College'] == 'HW' and df['hc_percentile'] < .16)\
    or (df['Home_College'] == 'KK' and df['hc_percentile'] < .16)\
    or (df['Home_College'] == 'MX' and df['hc_percentile'] < .05)\
    or (df['Home_College'] == 'OH' and df['hc_percentile'] < .16)\
    or (df['Home_College'] == 'TR' and df['hc_percentile'] < .14)\
    or (df['Home_College'] == 'WR' and df['hc_percentile'] < .09)\
    or (df['Gateway/Bridge status'] == 1):
        return 'Tier 1'
    elif (df['Home_College'] == 'DA' and df['hc_percentile'] < .55)\
    or (df['Home_College'] == 'HW' and df['hc_percentile'] < .54)\
    or (df['Home_College'] == 'KK' and df['hc_percentile'] < .54)\
    or (df['Home_College'] == 'MX' and df['hc_percentile'] < .29)\
    or (df['Home_College'] == 'OH' and df['hc_percentile'] < .54)\
    or (df['Home_College'] == 'TR' and df['hc_percentile'] < .87)\
    or (df['Home_College'] == 'WR' and df['hc_percentile'] < .32):
        return 'Tier 2'
    else:
        return 'Tier 3'
current_set['tier'] = current_set.apply(tiers, axis = 1)

# insert into SQL to update report for Tableau dashboard and end users
conn_ds = pyodbc.connect('Driver={SQL Server};'
                      'Server=10.27.251.83;'
                      'Database=DecisionSupport;'
                      'Trusted_Connection=yes;')
cursor = conn_ds.cursor()

# delete old data, if it exists
cursor.execute("drop table if exists decisionsupport.dbo.[retention scores 2022FA]")
# conn_ds.commit()
# cursor.close()

# insert Dataframe into SQL Server:
# cursor = conn_ds.cursor()
cursor.execute("""create table decisionsupport.dbo.[retention scores 2022FA] 
               ([Student_ID] varchar(100), [Retention_Score] varchar(100), tier varchar(100), hc varchar(100))""")
# conn_ds.commit()
# cursor.close()

# cursor = conn_ds.cursor()
for index, row in current_set[['student id', 'Score_1', 'tier', 'Home_College']].iterrows():
    cursor.execute("""\
    insert into decisionsupport.dbo.[retention scores 2022FA] ([Student_ID], [Retention_Score], [tier], [hc]) values(?, ?, ?, ?)"""
                   , row[0], row.Score_1 * 100, row[2], row[3])
conn_ds.commit()
cursor.close()