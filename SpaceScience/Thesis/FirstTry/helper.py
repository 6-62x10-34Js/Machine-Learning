import pyspedas
import datetime as dt
import pandas as pd
from pytplot import get_data


def check_timespan(timespan: list):
    start_time = dt.datetime.strptime(timespan[0], '%Y-%m-%d/%H:%M:%S')
    end_time = dt.datetime.strptime(timespan[1], '%Y-%m-%d/%H:%M:%S')

    if start_time > end_time:
        raise ValueError('Start time is later than end time')
    else:
        return True

def check_if_already_downloaded(timespan: list):

    return False


#14-44-22   :31:60.765
def save_data(data: pd.DataFrame, filename: str):
    data.to_csv(filename)

def construct_date_from_timespan(data: list, timespan: list):
    # find out how many data points there are and how many seconds are between each data point
    # then construct a list of datetime objects
    # then return that list
    start_time = dt.datetime.strptime(timespan[0], '%Y-%m-%d/%H:%M:%S')
    end_time = dt.datetime.strptime(timespan[1], '%Y-%m-%d/%H:%M:%S')
    time_delta = end_time - start_time
    seconds_between_data_points = time_delta.total_seconds() / len(data)
    print(seconds_between_data_points)
    print(len(data))
    print(time_delta.total_seconds())
    #construct dataframe of len data of datetime objects


    return
def download_ace_mfi(timespan: list):
    if check_timespan(timespan):
        mfi_ace = pyspedas.ace.mfi(trange=timespan, time_clip=True, varnames=['BGSEc', 'Magnitude'])
        ace_xyz = get_data('BGSEc')
        ace_mag = get_data('Magnitude')
        time_xyz = ace_xyz.times
        time_mag = ace_mag.times
        print(time_xyz)
        print(time_mag)
        #[1.44422316e+09 1.44422316e+09 1.44422316e+09 ... 1.44448242e+09
        dataframe = pd.DataFrame({'time': ace_xyz[0], 'bx': ace_xyz[1][:, 0], 'by': ace_xyz[1][:, 1], 'bz': ace_xyz[1][:, 2], 'b': ace_mag[1]})
        save_data(dataframe, f'ace_mfi_{timespan[0]}_{timespan[1]}')

        return ace_xyz, ace_mag
    else:
        return False

def download_wind_mfi(timespan: list):
    if check_timespan(timespan):
        mfi_wind = pyspedas.wind.mfi(trange=timespan, time_clip=True, varnames=['BGSE'])
        wind_xyz = get_data('BGSE')
        print(wind_xyz)

    else:
        return False

if __name__ == '__main__':
    timespan = ['2016-11-07/13:00:00', '2016-11-10/13:07:00']
    ace_xyz, ace_mag = download_wind_mfi(timespan)
