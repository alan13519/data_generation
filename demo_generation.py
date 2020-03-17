import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
import datetime


class Cohort:
    def __init__(self, cohort_values):
        self.values = cohort_values
        self.prob = None

    # Compares self.values with cohort list dictionary 
    def compare_cohorts(self, cohort_list):
        for key in self.values:
            # If cohort diction
            if key not in cohort_list.keys():
                return False
            else:
                # Compare each cohort with value
                if (self.values[key] != cohort_list[key]) and (self.values[key] is not None):
                    return False
        return True


# Cohort options (Change this as needed)
cohort_dict = {
    'Channel': ["App", "Branch", "Call Center", "Web"],
    'Cross_Sell': ["Yes", "No"],
    'Gender': ["M", "F"],
    'Generation': ["Post-Millennial", "Millennial", "Gen X", "Boomer"],
    'LoanType': ["Mortgage", "Personal", "Auto"],
    'Region': ["West Coast", "East Coast", "Midwest", "South"],
}


'''
Features we can tune for random data generation are below
'''
daily_obs = 100  # Number of observations per day
mean_rev = 5000  # Mean revenue per user
n_years = 3  # Number of years
anomaly_months = 30  # How many days of anomalous data
n_cohorts = 5  # Number of cohorts
cf_range = (2,4)  # Range of features a cohort might have
# TODO: Change below
month_start = 23
month_end = month_start + 3

'''
Feature tune end
'''

daily_data = np.repeat(range(1, (365*n_years)+1), daily_obs) # daily data
n_data = len(daily_data) # Number of data points

anom_start = int(daily_obs * month_start * (365/12))
anom_end = int(daily_obs * month_end * (365/12))

cohorts = [] # List of cohorts we want to make anomalies
keys = list(cohort_dict.keys())

# Generates cohorts automatically
for i in range(n_cohorts):
    samples = random.sample(range(0, len(keys)), random.randint(*cf_range))
    cohort = Cohort({
            'Channel': None,
            'Cross_Sell': None,
            'Gender': None,
            'Generation': None,
            'LoanType': None,
            'Region': None,
        })

    # Assigns random cohort values and probability
    for sample in samples:
        cohort.values[keys[sample]] = np.random.choice(cohort_dict[keys[sample]])
        cohort.prob = np.random.normal(.3, 0.25, 1)[0]
    
    # Append to cohort list, Not sure if this works
    cohorts.append(cohort)

# data
data = dict.fromkeys(cohort_dict.keys())

for key in data.keys():
    data[key] = np.random.choice(cohort_dict[key], n_data)

data = pd.DataFrame.from_dict(data)

# Custom map for generation vs age group
generation_map ={
    'Post-Millennial':(18,24),
    'Millennial':(24,40),
    'Gen X':(40,56),
    'Boomer':(56,72)
}


# Returns a random value between two integers of the map dictionary
def mapper(x, x_map):
    min_val, max_val = x_map[x]
    return random.choices(range(min_val, max_val))[0]


# Gets Age based on 
data['Age'] = data['Generation'].apply(mapper, x_map=generation_map)

values = np.random.normal(mean_rev * 2, (math.sqrt(math.sqrt(daily_obs)))*mean_rev, n_data)
data['Values'] = values + (daily_data * (mean_rev/100))
del values


# Add anomalies to data
def add_anomalies(row):
    # Compare data with cohort to see if it matches
    for cohort in cohorts:
        # If it matches then we transform the data
        if cohort.compare_cohorts(row.to_dict()):
            return row['Values'] * (1 - cohort.prob)
    else:
        return row['Values']


data.loc[anom_start:anom_end, 'Values'] = data.iloc[anom_start:anom_end].apply(add_anomalies, axis=1)

# Format output file name
d = datetime.date.today()
name = [str(d.month), str(d.day), str(d.year)]
name[0] = name[0].zfill(2) # Pad with 0 for month if less than len 2
name[1] = name[1].zfill(2) # Pad with 0 for day if less than len 2
filename = name[0] + name[1] + name[2] # File name

# ts = pd.Timestamp(year=2018, month=1, day=1)
# data['starterts'] = ts
# data['Date'] = data.apply(lambda row: row.starterts + pd.Timedelta(days=row.Day-1), axis = 1)
# data.drop('starterts', axis=1, inplace=True)
# data.to_csv(str(filename) + '_data.csv', index=False, header=False)

data['Day'] = daily_data
faux = data.groupby('Day').sum()
y = range(1, len(set(daily_data))+1)
plt.plot(y, faux['Values']) #; plt.axhline(y=(m*p), color='red', lw=3) ; plt.axhline(y=(m*p)+2*(math.sqrt(m*p*(1-p))), color='purple', lw=3) ; plt.axhline((m*p)-2*(math.sqrt(m*p*(1-p))), color='purple', lw=3)
plt.show()
