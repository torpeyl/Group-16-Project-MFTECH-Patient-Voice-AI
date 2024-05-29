import audformat
import pandas as pd

db = audformat.Database.load('./COVYT/dataset')
df = db['covid'].df
df['speaker'] = db['files'].get(index=df.index)['speaker']
df['language'] = df['speaker'].apply(lambda x: db.schemes['speaker'].labels[x]['language'])

features = pd.read_csv('features/eGeMAPSv02.csv')
features['start'] = features['start'].apply(pd.to_timedelta)
features['end'] = features['end'].apply(pd.to_timedelta)
features.set_index(['file', 'start', 'end'], inplace=True)

# get all features of German COVID-positive speakers
features.loc[df.loc[(df['language'] == 'german') & df['covid']].index]
# get all features of Boris Johnson when he was COVID-positive
features.loc[df.loc[(df['speaker'] == 'johnson') & df['covid']].index]
# get all features of Boris Johnson when he was COVID-negative
features.loc[df.loc[(df['speaker'] == 'johnson') & ~df['covid']].index]

table = db["files"].get()
print(table.iloc[0:2, 0:2])