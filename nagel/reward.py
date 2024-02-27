import numpy as np


def generate_reward_matrix(activity):
    xx = np.arange(0, 12)
    yy = np.arange(0, 24)
    X, Y = np.meshgrid(xx, yy)
    Z = np.zeros((len(xx), len(yy)))
    if activity == 'shopping':
        Z[0:2, 8:19] = 10
    elif activity == 'home':
        for k in (20, 19, 18, 17, 16, 15):
            for i in range(0, 12):
                for j in range(0, 24):
                    if i + j == k:
                        Z[i, j] = k - 14

        for k in (9, 10, 11, 12):
            for i in range(0, 12):
                for j in range(0, 24):
                    if i + j == k:
                        Z[i, j] = 26 - 2 * k

        for i in range(0, 12):
            for j in range(0, 24):
                if i + j <= 8:
                    Z[i, j] = 10
                if i + j <= 1:
                    Z[i, j] = 9
                if i + j >= 20:
                    Z[i, j] = 9
                if i + j >= 27:
                    Z[i, j] = 10
                if i + j >= 30:
                    Z[i, j] = 9
                if i + j >= 33:
                    Z[i, j] = 0
    elif activity == 'leisure':
        for i in range(0, 12):
            for j in range(0, 24):
                Z[i, j] = 1

        for k in range(13, 19):
            for i in range(0, 12):
                for j in range(0, 24):
                    if i + j == k:
                        Z[i, j] = k - 9

        for i in range(0, 12):
            for j in range(0, 24):
                if i + j >= 19:
                    Z[i, j] = 10
                if i + j >= 24:
                    Z[i, j] = 0
    elif activity == 'work':
        for i in range(0, 10):
            for j in range(0, 24):
                Z[i, j] = 10 - 10 * i / 10

        for i in range(9, 11):
            for j in range(0, 24):
                Z[i, j] = 400

        for i in range(0, 12):
            for j in range(0, 24):
                if i + j <= 8 or i + j >= 20:
                    Z[i, j] = 0
    return Z




