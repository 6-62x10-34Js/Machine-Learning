import pyspedas
import datetime as dt
import pandas as pd
from pytplot import get_data
from pyspedas import time_string
import matplotlib.pyplot as plt
import os

# TODO: read data from file if already downloaded


def check_timespan(timespan: list):
    start_time = dt.datetime.strptime(timespan[0], '%Y-%m-%d/%H:%M:%S')
    end_time = dt.datetime.strptime(timespan[1], '%Y-%m-%d/%H:%M:%S')

    if start_time > end_time:
        raise ValueError('Start time is later than end time')
    else:
        return True


def check_if_already_downloaded(timespan: list, mission: str, instrument: str):
    dir_path = '../data/'

    if check_timespan(timespan):
        matching_files = []
        # to datetime
        to_be_checked_start = dt.datetime.strptime(timespan[0], '%Y-%m-%d/%H:%M:%S')
        to_be_checked__end = dt.datetime.strptime(timespan[1], '%Y-%m-%d/%H:%M:%S')


        for file in os.scandir(os.path.join(dir_path, mission + "_data", instrument)):
            filename = file.name
            spans = filename.split('_')
            #remove .csv
            spans[-1] = spans[-1][:-4]
            # recombine date and time
            start_time = spans[2] + spans[3]
            end_time = spans[4] + spans[5]
            # from unix to datetime
            existing_start_times = dt.datetime.strptime(start_time, '%y%m%d%H%M%S')
            existing_end_times = dt.datetime.strptime(end_time, '%y%m%d%H%M%S')

            if to_be_checked_start >= existing_start_times and to_be_checked__end <= existing_end_times:
                matching_files.append(file.name)
                print(f'Found matching file for {mission} {instrument}')

        if matching_files:
            return matching_files
        else:
            return []

    return False


def encode_dates(timespan: list):
    savePath_datestring = '%y%m%d_%H%M%S'
    start_time = dt.datetime.strptime(timespan[0], '%Y-%m-%d/%H:%M:%S').strftime(savePath_datestring)
    end_time = dt.datetime.strptime(timespan[1], '%Y-%m-%d/%H:%M:%S').strftime(savePath_datestring)

    print(start_time, end_time)
    return [start_time, end_time]


# 14-44-22   :31:60.765
def save_data(data: pd.DataFrame, mission: str, instrument: str, start_time: str, end_time: str):
    path = f'../data/{mission}_data/{instrument}/{mission}_{instrument}_{start_time}_{end_time}.csv'
    data.to_csv(path, index=False)


def replace_time_with_datetime(data: pd.DataFrame):
    # replace the time column with a datetime column
    data_time = data['time']
    data_time = data_time.apply(lambda x: time_string(x))
    data['time'] = data_time
    return data


def download_ace_mfi(timespan: list):
    mission = 'ace'
    instrument = 'mfi'
    if check_timespan(timespan):
        pyspedas.ace.mfi(trange=timespan, time_clip=True, varnames=['BGSEc', 'Magnitude'])
        ace_xyz = get_data('BGSEc')
        ace_mag = get_data('Magnitude')
        dataframe = pd.DataFrame(
            {'time': ace_xyz[0], 'bx': ace_xyz[1][:, 0], 'by': ace_xyz[1][:, 1], 'bz': ace_xyz[1][:, 2],
             'b': ace_mag[1]})
        replace_time_with_datetime(dataframe)
        x, y = encode_dates(timespan)
        save_data(dataframe, mission, instrument, x, y)

        return ace_xyz, ace_mag
    else:
        return False


def download_wind_mfi(timespan: list):
    mission = 'wind'
    instrument = 'mfi'
    if check_timespan(timespan):
        pyspedas.wind.mfi(trange=timespan, time_clip=True, varnames=['BGSE'])
        wind_xyz = get_data('BGSE')
        dataframe = pd.DataFrame(
            {'time': wind_xyz[0], 'bx': wind_xyz[1][:, 0], 'by': wind_xyz[1][:, 1], 'bz': wind_xyz[1][:, 2]})
        replace_time_with_datetime(dataframe)
        x, y = encode_dates(timespan)
        save_data(dataframe, mission, instrument, x, y)

        return wind_xyz
    else:
        return False


if __name__ == '__main__':
    timespan = ['2016-10-08/13:00:00', '2016-10-9/13:07:00']
    missions = ['ace', 'wind']
    instruments = ['mfi']
    for mission in missions:
        for instrument in instruments:
            matching_files = check_if_already_downloaded(timespan, mission, instrument)
            if len(matching_files) > 0:
                print('Already downloaded')
            else:
                print('Downloading')
                if mission == 'ace':
                    ace_xyz, ace_mag = download_ace_mfi(timespan)
                elif mission == 'wind':
                    wind_xyz = download_wind_mfi(timespan)
                else:
                    print('Mission not found')




