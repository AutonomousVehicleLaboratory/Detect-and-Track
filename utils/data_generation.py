import numpy as np
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d


def find_closest_euclidean(centroids, points, trajectories, tracked, max_dist=15,):
    """
    Helper function for get_tracked used to generate trajectory data by appending detected object
     in the current frame to the closest existing trajectory by euclidean distance.
     If the distance exceeds max_dist threshold then create a new trajectory.
     :param centroids: list of centroids of pedestrian bounding box
     :param points:
     :param trajectories: existing trajectories data
     :param max_dist: threshold for euclidean distance
     :param tracked: when a trajectory is complete (no more preceding centroids), add it to tracked

    """
    if tracked is None:
        tracked = []
    if len(centroids) == 0:
        return points, [[p] for p in points], []
    new_centroids = []
    new_tj = []
    for p in points:
        new_centroids.append(p)
        if len(centroids) == 0:
            new_tj.append([p])
            pass
        else:
            dist = [euclidean(p, c) for c in centroids]
            min_idx = np.argmin(dist)
            if dist[min_idx] < max_dist:
                # add the centroid to existing trajectory
                current = trajectories[min_idx] + [p]
                new_tj.append(current)
                del centroids[min_idx]
                del trajectories[min_idx]
            else:
                new_tj.append([p])

    if len(trajectories) > 0:
        tracked += trajectories
    return new_centroids, new_tj, tracked


def get_tracked(ped_seq):
    """

    :param ped_seq: 2d array containing UnitObject (each row is a list of pedestrians detected in one img frame)
    :return: list of pedestrian trajectories
    """
    centroids = []
    tracked = []
    tj = []
    for ps in ped_seq:
        points = [i.box for i in ps]
        points = [((p[0] + p[2]) / 2, (p[1] + p[3]) / 2) for p in points]
        points = [p for p in points if p[0] >= 0]
        centroids, tj, tracked = find_closest_euclidean(centroids, points, tj, tracked)

    tracked += tj
    return tracked


def normalized_length(t, n=5):
    """
    Normalize trajectories to the same length
    :param t: original trajectory
    :param n: length of new trajectory
    :return:
    """
    step = 1/len(t)
    x = np.arange(0.0, 1.0, step)
    f = interp1d(x, t, axis=0)
    xnew = np.arange(0.0, x[-1], x[-1]/n)
    return f(xnew)


def transform_coord(peds_tj):
    """

    :param preds_tj: pedestrian trajectories in their original image coord
    :return: transformed coord centered at when the trajectories begin (when pedestrian enters the frame)
    """
    new_points = [[0,0]]
    p0 = peds_tj[0]
    for p in peds_tj[1:]:
        new_points.append([p[0] - p0[0], p[1] - p0[1]])
    return new_points


