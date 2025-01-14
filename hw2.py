import numpy as np
from code_for_hw02_downloadable import gen_flipped_lin_separable

data1, labels1, data2, labels2 = \
    (np.array([[-2.97797707,  2.84547604,  3.60537239, -1.72914799, -2.51139524,
                3.10363716,  2.13434789,  1.61328413,  2.10491257, -3.87099125,
                3.69972003, -0.23572183, -4.19729119, -3.51229538, -1.75975746,
                -4.93242615,  2.16880073, -4.34923279, -0.76154262,  3.04879591,
                -4.70503877,  0.25768309,  2.87336016,  3.11875861, -1.58542576,
                -1.00326657,  3.62331703, -4.97864369, -3.31037331, -1.16371314],
               [0.99951218, -3.69531043, -4.65329654,  2.01907382,  0.31689211,
                2.4843758, -3.47935105, -4.31857472, -0.11863976,  0.34441625,
                0.77851176,  1.6403079, -0.57558913, -3.62293005, -2.9638734,
                -2.80071438,  2.82523704,  2.07860509,  0.23992709,  4.790368,
                -2.33037832,  2.28365246, -1.27955206, -0.16325247,  2.75740801,
                4.48727808,  1.6663558,  2.34395397,  1.45874837, -4.80999977]]),
     np.array([[-1., -1., -1., -1., -1., -1.,  1.,  1.,  1., -1., -1., -1., -1.,
                -1.,  1., -1.,  1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1., -1.,
                -1., -1., -1., -1.]]), np.array([[0.6894022, -4.34035772,  3.8811067,  4.29658177,  1.79692041,
                                                  0.44275816, -3.12150658,  1.18263462, -1.25872232,  4.33582168,
                                                  1.48141202,  1.71791177,  4.31573568,  1.69988085, -2.67875489,
                                                  -2.44165649, -2.75008176, -4.19299345, -3.15999758,  2.24949368,
                                                  4.98930636, -3.56829885, -2.79278501, -2.21547048,  2.4705776,
                                                  4.80481986,  2.77995092,  1.95142828,  4.48454942, -4.22151738],
                                                 [-2.89934727,  1.65478851,  2.99375325,  1.38341854, -4.66701003,
                                                  -2.14807131, -4.14811829,  3.75270334,  4.54721208,  2.28412663,
                                                  -4.74733482,  2.55610647,  3.91806508, -2.3478982,  4.31366925,
                                                  -0.92428271, -0.84831235, -3.02079092,  4.85660032, -1.86705397,
                                                  -3.20974025, -4.88505017,  3.01645974,  0.03879148, -0.31871427,
                                                  2.79448951, -2.16504256, -3.91635569,  3.81750006,  4.40719702]]),
     np.array([[-1., -1.,  1.,  1., -1., -1., -1.,  1.,  1.,  1., -1.,  1.,  1.,
                -1.,  1.,  1.,  1., -1., -1., -1.,  1., -1.,  1., -1.,  1., -1.,
                -1.,  1.,  1.,  1.]]))


def is_mistake(th, th0, xi, yi):
    return (yi * (np.dot(th, xi) + th0)) <= 0


def update_thetas(th, th0, xi, yi):
    th = th + yi*xi
    th0 = th0 + yi
    return th, th0


def perceptron(data, labels, params={}, hook=None):
    n = data.shape[0]  # number of features
    m = data.shape[1]  # number of data
    th = np.zeros(n)
    th0 = 0
    T = params.get('T', 100)

    for _ in range(T):
        for i in range(m):
            xi = data[:, i]
            yi = labels[0, i]
            if is_mistake(th, th0, xi, yi):
                th, th0 = update_thetas(th, th0, xi, yi)

    return th.reshape(-1, 1), np.array([[th0]])


def averaged_perceptron(data, labels, params={}, hook=None):
    m = data.shape[0]  # number of features
    n = data.shape[1]  # number of data
    th = np.zeros(m)
    th0 = 0

    ths = np.zeros(m)
    th0s = 0
    T = params.get('T', 100)
    for _ in range(T):
        for i in range(n):
            xi = data[:, i]
            yi = labels[0, i]
            if is_mistake(th, th0, xi, yi):
                th, th0 = update_thetas(th, th0, xi, yi)
            ths += th
            th0s += th0

    return (ths/(n*T)).reshape(-1, 1), np.array([[th0s/(n*T)]])


def score(data, labels, th, th0):
    return 5
    pass


def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    scores = 0
    for _ in range(it):
        data_train, labels_train = data_gen(n_train)
        data_test, labels_test = data_gen(n_test)
        th, th0 = learner(data_train, labels_train)
        scores += score(data_test, labels_test, th, th0) / len(labels_test[0])

    return scores/it


def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th, th0 = learner(data_train, labels_train)

    return score(data_test, labels_test, th, th0) / len(labels_train[0])


def xval_learning_alg(learner, data, labels, k):
    scores = 0
    k_data = np.array_split(data, k, axis=1)
    l_data = np.array_split(labels, k, axis=1)

    for i in range(k):
        d_minus_j = np.concatenate([k_data[di]
                                   for di in range(k) if di != i], axis=1)
        l_minus_j = np.concatenate(
            [l_data[li] for li in range(k) if li != i], axis=1)
        th, th0 = learner(d_minus_j, l_minus_j)
        j_score = score(k_data[i], l_data[i], th, th0) / len(l_data[i][0])
        scores += j_score
    return scores / k


def test():
    # data, labels = data_generator()
    print(eval_learning_alg(averaged_perceptron,
          gen_flipped_lin_separable(pflip=0.01), 20, 20, 5))

    # res = eval_learning_alg(perceptron, data_generator, 10, 10, 5)


def main():
    # format_data(data1)
    # print(perceptron(data1, labels1, {'T': 100}))
    # print(eval_classifier(perceptron, data1, labels1, data2, labels2))
    # print(xval_learning_alg(perceptron, data1, labels1, 3))
    test()


main()
