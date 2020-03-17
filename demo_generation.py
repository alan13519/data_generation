import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
import datetime
import seaborn as sns
from itertools import repeat 


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


class CohortGenerator:
    def __init__(self, cohort_dict=None, daily_obs=100, mean_val=5000, n_years=3, n_cohorts=5, cf_range=(2, 4),
                 anomalous_months=[(1,3),(12,6)]):
        """
        Initializes cohort generator, takes in the initial parameters for the generated data
        :param cohort_dict: dict, cohort dictionary
        :param daily_obs: int, number of daily observations
        :param mean_val: int, mean value of data
        :param n_years: int, number of years we want to generate
        :param n_cohorts: int, number of cohorts we want to generate
        :param cf_range: tuple, the range of cohort features (min, max)
        :param anomalous_months: list of tuples, each tuple is the start month, and how many months it spans
        """

        # Default cohort dictionary, for testing
        if cohort_dict is None:
            # Cohort options (Change this as needed)
            self.cohort_dict = {
                'Channel': ["App", "Branch", "Call Center", "Web"],
                'Cross_Sell': ["Yes", "No"],
                'Gender': ["M", "F"],
                'Generation': ["Post-Millennial", "Millennial", "Gen X", "Boomer"],
                'LoanType': ["Mortgage", "Personal", "Auto"],
                'Region': ["West Coast", "East Coast", "Midwest", "South"],
            }
        else:
            self.cohort_dict = cohort_dict

        self.daily_obs = daily_obs  # number of daily observations
        self.mean_val = mean_val  # Mean revenue per user
        self.n_years = n_years  # Number of years
        self.n_cohorts = n_cohorts  # Number of cohorts
        self.cf_range = cf_range  # Range of features a cohort might have

        self.daily_data = np.repeat(range(1, (365 * self.n_years) + 1), self.daily_obs)  # daily data
        self.n_data = len(self.daily_data)  # Number of data points

        self.anomaly_list = []
        for anom in anomalous_months:
            anomaly_start = int(self.daily_obs *
                                    anom[0] * (365 / 12))  # Anomaly start, converted into days x daily observation
            anomaly_end = int(
                daily_obs * (anom[0] + anom[1]) * (365 / 12))  # Anomaly end, converted into days x daily observation\
            self.anomaly_list.append((anomaly_start, anomaly_end))

        self.cohorts, self.data = self.__generate_cohorts()
        self.data['Day'] = self.daily_data

    def __generate_cohorts(self, generation_map=None):
        """
        Generates the cohorts
        :param generation_map:
        :return: list of cohort class objects and dataframe of all data
        """
        # Custom map for generation vs age group if none are defined
        if generation_map is None:
            generation_map = {
                'Post-Millennial': (18, 24),
                'Millennial': (24, 40),
                'Gen X': (40, 56),
                'Boomer': (56, 72)
            }
        cohorts = []  # List of cohorts we want to make anomalies
        keys = list(self.cohort_dict.keys())  # Keys of cohort dictionary

        # Generates cohorts automatically
        for _ in range(self.n_cohorts):
            samples = random.sample(range(0, len(keys)), random.randint(*self.cf_range))

            # Initialize cohort dictionary with None values
            cohort = Cohort(
                dict(zip(keys, repeat(None)))
            )

            # Assigns random cohort values and probability
            for sample in samples:
                cohort.values[keys[sample]] = np.random.choice(self.cohort_dict[keys[sample]])
                cohort.prob = np.random.normal(0.5, 0.15, 1)[0]

            # Append to cohort list, Not sure if this works
            cohorts.append(cohort)

        # data
        data = dict.fromkeys(self.cohort_dict.keys())

        for key in data.keys():
            data[key] = np.random.choice(self.cohort_dict[key], self.n_data)

        data = pd.DataFrame.from_dict(data)

        # Returns a random value between two integers of the map dictionary
        def mapper(x, x_map):
            min_val, max_val = x_map[x]
            return random.choices(range(min_val, max_val))[0]

        # Gets Age based on
        data['Age'] = data['Generation'].apply(mapper, x_map=generation_map)

        values = np.random.normal(self.mean_val * 2, (math.sqrt(math.sqrt(self.daily_obs))) * self.mean_val,
                                  self.n_data)
        data['Values'] = values + (self.daily_data * (self.mean_val / 100))

        # Add anomalies to data
        def add_anomalies(row):
            # Compare data with cohort to see if it matches
            for c in cohorts:
                # If it matches then we transform the data
                if c.compare_cohorts(row.to_dict()):
                    return row['Values'] * (1 - c.prob)
            else:
                return row['Values']

        for start, end in self.anomaly_list:
            data.loc[start:end, 'Values'] = data.iloc[start:end].apply(add_anomalies, axis=1)

        return cohorts, data

    def output_file(self):
        # Format output file name
        d = datetime.date.today()
        name = [str(d.month), str(d.day), str(d.year)]
        name[0] = name[0].zfill(2)  # Pad with 0 for month if less than len 2
        name[1] = name[1].zfill(2)  # Pad with 0 for day if less than len 2
        filename = name[0] + name[1] + name[2]  # File name

        ts = pd.Timestamp(year=2018, month=1, day=1)
        df = self.data.copy()
        df['start'] = ts
        df['Date'] = df.apply(lambda row: row.starterts + pd.Timedelta(days=row.Day-1), axis = 1)
        df.drop('start', axis=1, inplace=True)
        df.to_csv(str(filename) + '_data.csv', index=False, header=False)

    def plot_data(self):
        df = self.data.copy()
        faux = df.groupby('Day').sum()
        y = range(1, len(set(self.daily_data)) + 1)
        plt.plot(y, faux[
            'Values'])
        plt.show()


if __name__ == '__main__':
    anomalies = [(1,3), (12, 6)]  # list of start months and how many months the anomaly spans
    cg = CohortGenerator(anomalous_months=anomalies)  # Create generator
    cg.plot_data()
