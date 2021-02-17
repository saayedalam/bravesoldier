import re
import numpy as np
import pandas as pd
from collections import Counter

def main():
    df = pd.read_csv('rleaves_clean.csv', encoding='utf-8')
    df = get_week(df['raw'])
    df.to_csv('rleaves_week.csv', index=False)
    
def day_to_week(df):
    return df.groupby(pd.cut(df.iloc[:, 0], np.arange(0, df.iloc[:, 0].max(), 7))).sum()

def get_day(series):
    day = list(series.str.findall(r'\d+\s*day[s\s]|\s*day\s*\d+'))
    day = [int(item) for item in re.findall(r'\d+', str(day))]
    day = pd.DataFrame([[x, day.count(x)] for x in set(day)]).rename(columns={0:'day', 1:'count'})
    day_week = day.copy()
    day = day.loc[(day['day'] > 0) & (day['day'] < 31)]
    day['day'] = 'Day ' + day['day'].astype(str)
    day.sort_values(by='count', ascending=False, inplace=True) 
    return day, day_week

def get_week(series):
    #global day_week
    week = list(series.str.findall(r'\d+\s*week[s\s]|\s*week\s*\d+'))
    week = [int(item) for item in re.findall(r'\d+', str(week))]
    week = pd.DataFrame.from_dict(Counter(week), orient='index').rename(columns={0:'w_count'}).sort_index(ascending=True)
    week = week[week.index < 35]
    week.index = 'Week ' + week.index.astype(str)
    day_week = day_to_week(get_day(series)[1])
    day_week = day_week.drop(['day'], axis=1).reset_index(drop=True)
    day_week = day_week[day_week['count'] > 1]
    day_week.index = ['Week %s' %i for i in range(1, len(day_week) + 1)]
    week = pd.concat([day_week, week], axis=1).fillna(0).reset_index()
    week[['count','w_count']] = week[['count','w_count']].astype(int)
    week['count'] = week['count'] + week['w_count']
    week.sort_values(by='count', ascending=False, inplace=True)
    week.drop(columns=['w_count'], inplace=True)
    return week

if __name__ == "__main__":
    main()
