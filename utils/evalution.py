import numpy as np
from scipy.spatial.distance import euclidean


def angle(u,v):
    c = np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v)
    return np.arccos(np.clip(c, -1, 1))


def classify_path(test, pred_y):
    first_points = [t[0] for t in test]
    last_points = [t[-1] for t in test]

    target = []
    pred = []
    straight_line = np.array([0.0, 1.0])
    for i in range(len(pred_y)):
        tar_line = np.array(last_points[i]) - np.array(first_points[i])
        angle_t = angle(tar_line, straight_line)
        pred_line = np.array(pred_y[i]) - np.array(first_points[i])
        angle_p = angle(pred_line, straight_line)

        target.append(angle_t < np.pi / 4 or angle_t > 3 * np.pi / 4)
        pred.append(angle_p < np.pi / 4 or angle_p > 3 * np.pi / 4)

    acc = sum([1 if target[i] == pred[i] else 0 for i in range(len(pred))]) / len(pred)

    return target, pred, acc


def full_dist(path):
    dist = 0
    for i in range(len(path) - 1):
        dist += euclidean(path[i], path[i+1])
    return dist


def error_rate(test, pred_y, test_y):

    """

    :param test:
    :param pred_y:
    :param test_y:
    :return: avg error (in pixel), pct error, list of path lengths, list of errors in pixels
    """
    pred_deviation = [euclidean(pred_y[i], test_y[i]) for i in range(len(pred_y))]
    path_dist = [full_dist(t) for t in test]
    percent_error = [0 if path_dist[i] == 0 else pred_deviation[i]/path_dist[i] for i in range(len(test))]
    return np.sum(pred_deviation)/len(test), np.sum(percent_error)/len(test), path_dist, pred_deviation
