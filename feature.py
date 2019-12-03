import cv2
import numpy as np



def get_features(left_img=None, right_img=None):
    """
    get SIFT features using orb model.

    :param left_img:
    :param right_img:
    :return:
    """
    left_grey = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_grey = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    left_keypoints, left_descriptors = orb.detectAndCompute(left_grey, None)
    right_keypoints, right_descriptors = orb.detectAndCompute(right_grey, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(left_descriptors, right_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)

    # choose top 20% for RANSAC
    nmatch = int(len(matches) * 0.2)
    if nmatch < 4:
        raise RuntimeError("More points are need for calculating transform matrix! ")

    filtered_matches = matches[:nmatch]

    return left_keypoints, right_keypoints, filtered_matches


def least_squares_fit(f1, f2, matches, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

    '''

    M = np.eye(3)
    # For spherically warped images, the transformation is a
    # translation and only has two degrees of freedom.
    # Therefore, we simply compute the average translation vector
    # between the feature in f1 and its match in f2 for all inliers.
    u = 0.0
    v = 0.0

    for i in range(len(inlier_indices)):
        # Use this loop to compute the average translation vector
        # over all inliers.

        trainIdx = matches[inlier_indices[i]].trainIdx
        queryIdx = matches[inlier_indices[i]].queryIdx
        u += f2[trainIdx].pt[0] - f1[queryIdx].pt[0]
        v += f2[trainIdx].pt[1] - f1[queryIdx].pt[1]

    u /= len(inlier_indices)
    v /= len(inlier_indices)

    M[0, 2] = u
    M[1, 2] = v

    return M


def get_inliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        # BEGIN TODO 5
        # Determine if the ith matched feature f1[id1], when transformed
        # by M, is within RANSACthresh of its match in f2.
        # If so, append i to inliers
        # TODO-BLOCK-BEGIN
        m = matches[i]
        a_xy = np.array(
            [f1[m.queryIdx].pt[0], f1[m.queryIdx].pt[1], 1])  # (a_x, a_y) pixel coordinate in the first image
        b_xy = np.array(
            [f2[m.trainIdx].pt[0], f2[m.trainIdx].pt[1], 1])  # (b_x, b_y) pixel coordinate in the second image
        a_xy = a_xy.reshape((3, 1))
        proj_a = M.dot(a_xy)
        from scipy.spatial import distance
        proj_a /= proj_a[2, 0]
        dst = distance.euclidean(proj_a, b_xy)
        if dst <= RANSACthresh:
            inlier_indices.append(i)

    return inlier_indices


def align_pair(f1, f2, matches, nRANSAC=500, RANSACthresh=5.0):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
    '''

    # This function should also call get_inliers and, at the end,
    # least_squares_fit.
    M = np.eye(3)
    M[2,] = np.array([0, 0, 1])
    most = (M, [])
    for i in range(nRANSAC):
        idx = np.random.choice(len(matches), 1)
        selected_matches = [matches[j] for j in idx]
        M[0, 2] = f2[selected_matches[0].trainIdx].pt[0] - f1[selected_matches[0].queryIdx].pt[0]
        M[1, 2] = f2[selected_matches[0].trainIdx].pt[1] - f1[selected_matches[0].queryIdx].pt[1]
        inliners = get_inliers(f1, f2, matches, M, RANSACthresh)
        if len(inliners) > len(most[1]):
            most = (M, inliners)

    # inliner_matches = [matches[j] for j in most[1]]
    M = least_squares_fit(f1, f2, matches, most[1])

    return M

def get_mapping(left_img, right_img):
    """
    get mapping: left -> right
    """
    k1, k2, m = get_features(left_img, right_img)
    M = align_pair(k1, k2, m)

    return M
