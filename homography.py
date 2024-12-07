def computeH(x1, x2):
    """
    Compute the homography between two sets of points

    Input
    -----
    x1, x2: Sets of points

    Returns
    -------
    H2to1: 3x3 homography matrix that best transforms x2 to x1
    """

    if x1.shape != x2.shape:
        raise RuntimeError('number of points do not match')

    # ===== your code here! =====
    # TODO: Compute the homography between two sets of points
    A = []
    for i in range(len(x1)):

      A.append([x2[i][0], x2[i][1], 1, 0, 0, 0, -x2[i][0]*x1[i][0], -x2[i][1]*x1[i][0], -x1[i][0]])
      A.append([0, 0, 0, x2[i][0], x2[i][1], 1, -x2[i][0]*x1[i][1], -x2[i][1]*x1[i][1], -x1[i][1]])
    A = np.matrix(np.array(A))

    U, Sigma, VT = np.linalg.svd(A)
    V = VT.T
    h = V[:, -1]
    H2to1 = np.reshape(h, (3,3))
    # ==== end of code ====

    return H2to1

def computeH_norm(x1, x2):
    """
    Compute the homography between two sets of points using normalization

    Input
    -----
    x1, x2: Sets of points

    Returns
    -------
    H2to1: 3x3 homography matrix that best transforms x2 to x1
    """



    c1 = np.mean(x1, axis = 0)
    c2 = np.mean(x2, axis = 0)

    # TODO: Shift the origin of the points to the centroid

    x1_shifted = x1 - c1
    x2_shifted = x2 - c2

    # TODO: Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x1_shifted_norm = np.linalg.norm(x1_shifted, axis = 1)
    x2_shifted_norm = np.linalg.norm(x2_shifted, axis = 1)
    max_norm1 = np.max(x1_shifted_norm)
    max_norm2 = np.max(x2_shifted_norm)
    scale1 = np.sqrt(2) / max_norm1
    scale2 = np.sqrt(2) / max_norm2
    x1_normalized = x1_shifted * scale1
    x2_normalized = x2_shifted * scale2

    # TODO: Similarity transform 1

    T1 = np.array([
      [scale1, 0, -scale1*c1[0]],
      [0, scale1, -scale1*c1[1]],
      [0, 0, 1]
    ])

    # TODO: Similarity transform 2
    T2 = np.array([
      [scale2, 0, -scale2*c2[0]],
      [0, scale2, -scale2*c2[1]],
      [0, 0, 1]
    ])

    # TODO: Compute homography
    H = computeH(x1_normalized, x2_normalized)

    # TODO: Denormalization
    #H2to1 = np.matmul(np.linalg.inv(T1), np.matmul(H, T2))
    intermediate = np.matmul(H, T2)
    H2to1 = np.matmul(np.linalg.inv(T1), intermediate)



    return H2to1

import numpy as np
from scipy.spatial.distance import cdist

def computeH_ransac(locs1, locs2, max_iters, inlier_tol):
    """
    Estimate the homography between two sets of points using ransac

    Input
    -----
    locs1, locs2: Lists of points
    max_iters: the number of iterations to run RANSAC for
    inlier_tol: the tolerance value for considering a point to be an inlier

    Returns
    -------
    bestH2to1: 3x3 homography matrix that best transforms locs2 to locs1
    inliers: indices of RANSAC inliers

    """

    # ===== your code here! =====

    # TODO:
    # Compute the best fitting homography using RANSAC
    # given a list of matching points locs1 and loc2
    def compute_inliers(H, locs1, locs2, tol):
        """
        Compute inliers by applying homography to locs2 and comparing with locs1.
        Returns the indices of inliers.
        """

        locs2_hom = np.hstack((locs2, np.ones((locs2.shape[0], 1))))


        locs2_proj = np.dot(H, locs2_hom.T).T
        locs2_proj = locs2_proj[:, :2] / locs2_proj[:, 2].reshape(-1, 1)


        errors = np.linalg.norm(locs1 - locs2_proj, axis=1)


        inliers = np.where(errors < tol)[0]
        return inliers

    bestH2to1 = None
    max_inliers = 0
    best_inliers = []

    for _ in range(max_iters):

        indices = np.random.choice(len(locs1), 4, replace=False)
        x1, x2 = locs1[indices], locs2[indices]


        H = computeH_norm(x1, x2)


        inliers = compute_inliers(H, locs1, locs2, inlier_tol)


        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            bestH2to1 = H
            best_inliers = inliers

    print(best_inliers)

    # ==== end of code ====

    return bestH2to1, inliers

def compositeH(H2to1, template, img):
    """
    Returns the composite image.

    Input
    -----
    H2to1: Homography from image to template
    template: template image to be warped
    img: background image

    Returns
    -------
    composite_img: Composite image

    """

    # ===== your code here! =====
    # TODO: Create a composite image after warping the template image on top
    # of the image using the homography


    mask_ones = np.ones(template.shape)
    mask_ones = cv2.transpose(mask_ones)
    warp_mask = cv2.warpPerspective(
        mask_ones, H2to1, (img.shape[0], img.shape[1]))
    template = cv2.transpose(template)

    warp_mask = cv2.transpose(warp_mask)
    non_zero_ind = np.nonzero(warp_mask)

    warp_template = cv2.warpPerspective(template, H2to1, (img.shape[0], img.shape[1]))
    warp_template = cv2.transpose(warp_template)
    img[non_zero_ind] = warp_template[non_zero_ind]
    # composite_img = img.astype('uint8')
    # composite_img = cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)
    composite_img = img


    # ==== end of code ====

    return composite_img

def warpImage(ratio, sigma, max_iters, inlier_tol):
    """
    Warps hp_cover.jpg onto the book cover in cv_desk.png.

    Input
    -----
    ratio: ratio for BRIEF feature descriptor
    sigma: threshold for corner detection using FAST feature detector
    max_iters: the number of iterations to run RANSAC for
    inlier_tol: the tolerance value for considering a point to be an inlier

    """

    hp_cover = skimage.io.imread(os.path.join(DATA_DIR, 'hp_cover.jpg'))
    cv_cover = skimage.io.imread(os.path.join(DATA_DIR, 'cv_cover.jpg'))
    cv_desk = skimage.io.imread(os.path.join(DATA_DIR, 'cv_desk.png'))
    cv_desk = cv_desk[:, :, :3]

    # ===== your code here! =====

    print("hp_cover shape:", hp_cover.shape)
    print("cv_cover shape:", cv_cover.shape)
    print("cv_desk shape:", cv_desk.shape)




    matches, locs1, locs2 = matchPics(cv_desk, cv_cover, ratio, sigma)


    if cv_cover.ndim == 2:
        cv_cover = cv2.cvtColor(cv_cover, cv2.COLOR_GRAY2BGR)
    if hp_cover.ndim == 2:
        hp_cover = cv2.cvtColor(hp_cover, cv2.COLOR_GRAY2BGR)

    # TODO: Get matched features
    matched_locs1 = locs1[matches[:, 0]]
    matched_locs2 = locs2[matches[:, 1]]

    print("Matched Locations 1 shape:", matched_locs1.shape)
    print("Matched Locations 2 shape:", matched_locs2.shape)

    if len(matched_locs1) < 4 or len(matched_locs2) < 4:
        raise ValueError("Not enough matched points to compute homography.")

    # TODO: Scale matched pixels in cv_cover to size of hp_cover
    scale_x = hp_cover.shape[1] / cv_cover.shape[1]
    scale_y = hp_cover.shape[0] / cv_cover.shape[0]

    matched_locs2 = matched_locs2.astype(np.float64)

    matched_locs2[:, 0] *= scale_y
    matched_locs2[:, 1] *= scale_x

    # TODO: Get homography by RANSAC using computeH_ransac

    H2to1, inliers = computeH_ransac(matched_locs1, matched_locs2, max_iters, inlier_tol)
    print("Computed Homography Matrix:\n", H2to1)


    # TODO: Overlay using compositeH to return composite_img
    composite_img = compositeH(H2to1, hp_cover, cv_desk)

    # ==== end of code ====

    plt.imshow(composite_img)
    plt.show()

