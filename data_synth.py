import sqlalchemy
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
engine2 = sqlalchemy.create_engine('sqlite:///path\\to\\kolibri\\db\\db.sqlite3')
engine2.connect()
# content_channelmetadata
sql = """
    SELECT id,
     name, 
     last_updated
    FROM content_channelmetadata
    WHERE public = 1;
    """
channelmt = pd.read_sql(sql, engine2)
sql = """
    SELECT *
    FROM content_contentnode
    where available = 1
    """
channelcont = pd.read_sql(sql, engine2)
# a lot of metadata
sql = """
    SELECT *
    FROM exams_examassignment
    """
examsass = pd.read_sql(sql, engine2)
# super important table
sql = """
    SELECT *
    FROM kolibriauth_facilityuser
    """
facilusers = pd.read_sql(sql, engine2)
# user created lessons
sql = """
    SELECT *
    FROM lessons_lesson
    """
lessons = pd.read_sql(sql, engine2)
# match user id to correct or incorrect answers
# very important table
sql = """
    SELECT *
    FROM logger_attemptlog
    """
attempts = pd.read_sql(sql, engine2)
# matches user id to lessons viewed
sql = """
SELECT *
FROM logger_contentsessionlog
"""
lessons_log    = pd.read_sql(sql, engine2)
sql = """
    SELECT *
    FROM logger_contentsummarylog
    """
contentsummary = pd.read_sql(sql, engine2)

df_generated_data = pd.DataFrame(columns=facilusers.columns)
print(df_generated_data.columns)
df_generated_data = df_generated_data.drop(['_morango_dirty_bit', '_morango_source_id', '_morango_partition'
                                            ], axis='columns')
# todo remove this variable later

df_generated_data['paternal_status_S'] = np.random.choice(a=[0,1], size=1000,
                                                   p=[0.6, 0.4])
df_generated_data['paternal_status_T'] = [0 if x==1 else 1 for x in list(df_generated_data['paternal_status_S'])]

# add lessons completed, right answer in quizzes/total questions,
lessons_meta = pd.merge(lessons_log, facilusers, right_on='id', left_on='user_id', how='inner')
lessons_meta = pd.merge(lessons_meta, channelcont, on='content_id', how='inner', )
lessons_meta = pd.merge(lessons_meta, contentsummary, on='content_id', how='inner',)

lessons_meta = lessons_meta.groupby('username', as_index=False).sum()
lessons_meta = pd.merge(lessons_meta, facilusers.loc[:,['username','id']], right_on='username', left_on='username', how='inner')
merged_df = pd.merge(df_generated_data, lessons_meta, left_on='id',right_on='id')
merged_df = merged_df.rename({'progress_x': 'finished', 'available':'material_remaining'}, axis='columns')
merged_df['material_remaining'] = merged_df.material_remaining.round()-merged_df['finished']
df_generated_data['total_material'] = np.random.randint(4, 33, size=1000)
random_finished_material = []
for s in df_generated_data['total_material']:
    random_finished_material.append(np.random.randint(2, s))
df_generated_data['finished_material'] = random_finished_material
df_generated_data['remaining_material'] = df_generated_data['total_material']-df_generated_data['finished_material']
df_generated_data['time_spent'] = df_generated_data['finished_material']*200
df_generated_data['grade_average'] = np.random.normal(70, 6,1000)
df_generated_data = df_generated_data.dropna(axis=1)
import matplotlib.pyplot as plt
plt.hist(df_generated_data['grade_average'])
plt.xlabel('Grade average')
plt.show()
df_generated_data['at_risk_target'] = [0 if x>70 else 1 for x in df_generated_data['grade_average']]
df_generated_data['school'] = ['city_sch' if x == 0 else 'loc_sch' for x in df_generated_data['at_risk_target']]
df_generated_data['drugs'] = [int(np.random.choice(a=[0, 1], size=1,
                                               p=[0.1, 0.9])) if x == 1 else int(np.random.choice(a=[0, 1], size=1,
                                                                                             p=[0.9, 0.1])) for x in df_generated_data['at_risk_target']]
df_generated_data['alcohol'] = [int(np.random.choice(a=[0, 1], size=1,
                                                p=[0.1, 0.9])) if x==1 else int(np.random.choice(a=[0, 1], size=1,
                                                                                              p=[0.90, 0.1])) for x
                               in df_generated_data['at_risk_target']]
df_generated_data['study_time'] = [int(np.random.choice(a=[0.5, 1], size=1,
                                                p=[0.9, 0.1])) if x==1 else int(np.random.choice(a=[2, 3], size=1,
                                                                                              p=[0.30, 0.70])) for x
                               in df_generated_data['at_risk_target']]
df_generated_data['extra_activities'] = [int(np.random.choice(a=[0, 1], size=1,
                                                p=[0.9, 0.1])) if x==1 else int(np.random.choice(a=[0, 1], size=1,
                                                                                              p=[0.15, 0.85])) for x
                               in df_generated_data['at_risk_target']]

df_generated_data['free_time'] = [int(np.random.choice(a=[2, 3], size=1,
                                                p=[0.2, 0.8])) if x==1 else int(np.random.choice(a=[0, 1], size=1,
                                                                                              p=[0.2, 0.8])) for x
                               in df_generated_data['at_risk_target']]
df_generated_data['job_mother'] = [np.random.choice(a=['nurse', 'teacher', 'bsns_owner', 'stay_home', 'other'], size=1,
                                                   p=[0.2, 0, 0.1, 0.70, 0.0])[0] if x==1 else np.random.choice(a=['nurse', 'teacher', 'bsns_owner', 'stay_home', 'other'], size=1,
                                                   p=[0.05, .70, 0.2, 0.0, 0.05])[0] for x in df_generated_data['at_risk_target']]
df_generated_data['job_father'] = [np.random.choice(a=['taxi_driver', 'builder', 'blacksmith', 'stay_home', 'other'], size=1,
                                                   p=[0.7, 0.2, 0, 0.05, 0.05])[0] if x==1 else np.random.choice(a=['taxi_driver', 'builder', 'blacksmith', 'stay_home', 'other'], size=1,
                                                   p=[0.1, 0.1, 0.1, 0.7, 0.00])[0]  for x in df_generated_data['at_risk_target']]
freq_table = pd.crosstab(index=df_generated_data.at_risk_target, columns=df_generated_data.drugs)
print(freq_table)

df_generated_data = pd.get_dummies(df_generated_data)
df_generated_data.to_csv('without_normalization.csv')
scaler_normalizer = MinMaxScaler()
df_generated_data = pd.DataFrame(scaler_normalizer.fit_transform(df_generated_data), columns=df_generated_data.columns,
                                 index=df_generated_data.index)
print(df_generated_data)
# todo add variable capturing system
# add
# variable correlation
import matplotlib.pyplot as plt
plt.hist(df_generated_data['grade_average'])
plt.xlabel('Grade average')
plt.show()
df_generated_data.to_csv('normalized.csv')
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.figure(figsize=(7,5))
# correlation = df_generated_data.corr()
# sns.heatmap(correlation)
# plt.show()

