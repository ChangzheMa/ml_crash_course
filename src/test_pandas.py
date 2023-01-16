import numpy as np
import pandas as pd


def test_pd():
    # data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    # column = ['temp', 'active']
    # data_frame = pd.DataFrame(data=data, columns=column)
    # print(data_frame)
    #
    # data_frame['adj_active'] = data_frame['active'] + 3
    # print(data_frame)

    data = np.random.randint(0, 101, (3, 4))
    column = ['E', 'C', 'T', 'J']
    data_frame = pd.DataFrame(data=data, columns=column)
    print(data_frame)
    print(data_frame[1:2]['E'])

    print("============================")

    data_frame['S'] = data_frame['T'] + data_frame['J']
    print(data_frame)
