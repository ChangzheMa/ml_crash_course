import numpy as np


def test_np():
    # one_dim_arr = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
    # print(one_dim_arr)

    # two_dim_arr = np.array([[6, 5], [11, 7], [4, 8]])
    # print(two_dim_arr)

    # sequence_arr = np.arange(5, 12)
    # print(sequence_arr)

    # random_int_50_to_100 = np.random.randint(50, 100, (6, 5, 3))
    # print(random_int_50_to_100)

    # random_float_0_1 = np.random.random([6])
    # print(random_float_0_1)

    # random_float_0_3 = np.random.random([3]) * 3.0
    # print(random_float_0_3)

    feature = np.arange(6, 21)
    print(feature)

    label = feature * 3 + 4
    print(label)

    noise = np.random.random(feature.size) * 4 - 2
    print(noise)

    label = label + noise
    print(label)
