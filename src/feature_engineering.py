import pandas as pd
import numpy as np

def create_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Holiday proximity features
    df['DaysToHoliday'] = (pd.to_datetime('2015-12-25') - df['Date']).dt.days
    df['DaysAfterHoliday'] = (df['Date'] - pd.to_datetime('2015-12-25')).dt.days
    
    # Promo2 active flag
    df['Promo2Active'] = ((df['Promo2'] == 1) & 
                          (df['Date'].dt.year >= df['Promo2SinceYear']) & 
                          (df['Date'].dt.week >= df['Promo2SinceWeek'])).astype(int)
    
    # Competition active flag
    df['CompetitionActive'] = ((df['CompetitionOpenSinceYear'] <= df['Year']) & 
                               (df['CompetitionOpenSinceMonth'] <= df['Month'])).astype(int)
    
    # Seasonality flags
    seasons = {
        1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall',
        12: 'Winter'
    }
    df['Season'] = df['Month'].map(seasons)
    
    return df