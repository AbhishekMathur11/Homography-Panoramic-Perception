def createPanorama(left_im, right_im, ratio, sigma, max_iters, inlier_tol):
    """
    Create a panorama augmented reality application by computing a homography
    and stitching together a left and right image.

    Input
    -----
    left_im: left image
    right_im: right image
    ratio: ratio for BRIEF feature descriptor
    sigma: threshold for corner detection using FAST feature detector
    max_iters: the number of iterations to run RANSAC for
    inlier_tol: the tolerance value for considering a point to be an inlier

    Returns
    ------
    panorama_im: Stitched together panorama
    """


    left_gray = cv2.cvtColor(left_im, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_im, cv2.COLOR_BGR2GRAY)


    matches, locs1, locs2 = matchPics(left_im, right_im, ratio, sigma)


    matched1 = np.array([locs1[m[0]][::-1] for m in matches])
    matched2 = np.array([locs2[m[1]][::-1] for m in matches])

    H2to1, inliers = computeH_ransac(matched2, matched1, max_iters, inlier_tol)
    print("Computed Homography Matrix:\n", H2to1)



    left_corners = np.array([[0, 0, 1], [left_im.shape[1] - 1, 0, 1],
                              [0, left_im.shape[0] - 1, 1],
                              [left_im.shape[1] - 1, left_im.shape[0] - 1, 1]])
    right_corners = np.array([[0, 0, 1], [right_im.shape[1] - 1, 0, 1],
                               [0, right_im.shape[0] - 1, 1],
                               [right_im.shape[1] - 1, right_im.shape[0] - 1, 1]])


    warped_left_corners = H2to1 @ left_corners.T
    warped_left_corners /= warped_left_corners[2, :]


    min_x = int(min(0, warped_left_corners[0, :].min()))
    max_x = int(max(right_im.shape[1], warped_left_corners[0, :].max()))
    min_y = int(min(0, warped_left_corners[1, :].min()))
    max_y = int(max(left_im.shape[0], warped_left_corners[1, :].max()))


    size = (max_x - min_x, max_y - min_y)


    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)


    translation = np.array([[1, 0, -min_x],
                            [0, 1, -min_y],
                            [0, 0, 1]])
    H2to1_translated = translation @ H2to1


    left_im_warped = cv2.warpPerspective(left_im, H2to1_translated, size)


    canvas[-min_y:right_im.shape[0] - min_y, -min_x:right_im.shape[1] - min_x] = right_im


    mask_left = (left_im_warped > 0).astype(np.uint8)
    mask_right = (canvas > 0).astype(np.uint8)


    stitched_image = left_im_warped * mask_left + canvas * mask_right * (1 - mask_left)


    stitched_gray = cv2.cvtColor(stitched_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(stitched_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if contours:
        hull = cv2.convexHull(contours[0])
        x, y, w, h = cv2.boundingRect(hull)
        output = stitched_image[y:y + h, x:x + w]
    else:
        output = stitched_image


    cv2.imwrite("cropped_panorama.jpg", output)
    return output.astype(np.uint8)
