import os
import cv2
import numpy as np
import scipy.ndimage

def briefRot(min_deg, max_deg, deg_inc, ratio, sigma, filename):
    """
    Tests Brief with rotations.

    Input
    -----
    min_deg: minimum degree to rotate image
    max_deg: maximum degree to rotate image
    deg_inc: number of degrees to increment when iterating
    ratio: ratio for BRIEF feature descriptor
    sigma: threshold for corner detection using FAST feature detector
    filename: filename of image to rotate

    """

    print("Starting briefRot function...")

    if not os.path.exists(RES_DIR):
        raise RuntimeError('RES_DIR does not exist. Did you run all cells?')


    image_path = os.path.join(DATA_DIR, filename)
    print(f"Reading image from: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not read image from {image_path}.")

    print(f"Image shape: {image.shape}")

    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Converted image from BGR to RGB.")
    else:
        raise RuntimeError("Image is not in expected BGR format.")

    match_degrees = []
    match_counts = []


    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Converted image to grayscale.")

    for i in range(min_deg, max_deg, deg_inc):
        print(f"Processing rotation: {i} degrees")


        try:
            rotated_img = scipy.ndimage.rotate(image_gray, i, reshape=False)
            print(f"Rotated image by {i} degrees. Rotated image shape: {rotated_img.shape}")
        except Exception as e:
            print(f"Error rotating image at {i} degrees: {e}")
            continue


        try:
            interest_points1 = corner_detection(image_gray, sigma)
            print(f"Detected interest points in original image: {len(interest_points1)} points")

            interest_points2 = corner_detection(rotated_img, sigma)
            print(f"Detected interest points in rotated image: {len(interest_points2)} points")

            desc1, locs1 = computeBrief(image_gray, interest_points1)
            desc2, locs2 = computeBrief(rotated_img, interest_points2)
            print(f"Computed descriptors: {desc1.shape} and {desc2.shape}")

            matches = briefMatch(desc1, desc2, ratio)
            print(f"Number of matches found: {len(matches)}")
        except Exception as e:
            print(f"Error during feature matching: {e}")
            continue

        match_counts.append(len(matches))
        match_degrees.append(i)


        if i in [min_deg, (min_deg + max_deg) // 2, max_deg - deg_inc]:
            print("Plotting matches...")
            try:
                rotated_img_rgb = cv2.cvtColor(rotated_img, cv2.COLOR_GRAY2RGB)
                plotMatches(image_rgb, rotated_img_rgb, matches, locs1, locs2)
                print("Matches plotted successfully.")
            except Exception as e:
                print(f"Error plotting matches: {e}")


    try:
        matches_to_save = [match_counts, match_degrees, deg_inc]
        write_pickle(ROT_MATCHES_PATH, matches_to_save)
        print("Match counts and degrees saved to pickle file successfully.")
    except Exception as e:
        print(f"Error saving matches to pickle file: {e}")

    print("Completed briefRot function.")


def dispBriefRotHist(matches_path=ROT_MATCHES_PATH):

    if not os.path.exists(matches_path):
      raise RuntimeError('matches_path does not exist. did you call briefRot?')


    match_counts, match_degrees, deg_inc = read_pickle(matches_path)


    plt.figure()
    bins = [x - deg_inc/2 for x in match_degrees]
    bins.append(bins[-1] + deg_inc)
    plt.hist(match_degrees, bins=bins, weights=match_counts, log=True)

    plt.title("Histogram of BRIEF matches")
    plt.ylabel("# of matches")
    plt.xlabel("Rotation (deg)")
    plt.tight_layout()

    output_path = os.path.join(RES_DIR, 'histogram.png')
    plt.savefig(output_path)
