import pandas as pd
import numpy as np
import torch

def preprocessing(data):
    scheduled_matching = data[data["Date/Time"].dt.time == pd.to_datetime("14:46:00").time()]
    train_size = len(scheduled_matching)*0.8
    train_row = int(train_size)
    training_data_schedule = scheduled_matching.iloc[:train_row]
    testing_data_schedule = scheduled_matching.iloc[train_row:]
    #timeline = testing_data_schedule['Date\Time'].values
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(2, len(training_data_schedule)):
        close_day = training_data_schedule.iloc[[i]]['Date/Time'].values[0]
        last_day  = data[data['Date/Time'] < close_day]
        # Getting price from 9:15-> 11:29 and 13:00->14:29 = 223 rows
        last_day_223_m = last_day.iloc[len(last_day)-223: ]
        close_price_last_1day = training_data_schedule.iloc[[i-1]]
        close_price_last_2day = training_data_schedule.iloc[[i-2]]
        close_price_combined = pd.concat([last_day_223_m, close_price_last_1day, close_price_last_2day], axis=0).drop(['Date/Time'], axis=1).to_numpy()
        x_train.append(close_price_combined)
        y_train.append(training_data_schedule.iloc[[i]]['Close'].values[0])

    for i in range(2, len(testing_data_schedule)):
        close_day = testing_data_schedule.iloc[[i]]['Date/Time'].values[0]
        last_day  = data[data['Date/Time'] < close_day]
        # Getting price from 9:15-> 11:29 and 13:00->14:29 = 223 rows
        last_day_223_m = last_day.iloc[len(last_day)-223: ]
        close_price_last_1day = testing_data_schedule.iloc[[i-1]]
        close_price_last_2day = testing_data_schedule.iloc[[i-2]]
        close_price_combined = pd.concat([last_day_223_m, close_price_last_1day, close_price_last_2day], axis=0).drop(['Date/Time'], axis=1).to_numpy()
        x_test.append(close_price_combined)
        y_test.append(testing_data_schedule.iloc[[i]]['Close'].values[0])

    # Reshaping the arrays into 3D arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    #return torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
    return x_train, y_train, x_test, y_test