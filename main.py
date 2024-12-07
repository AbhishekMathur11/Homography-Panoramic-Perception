from homography import computeH, computeH_norm, computeH_ransac, compute_inliers, compositeH, warpImage
from matching import matchPics, displayMatched
from panorama import createPanorama
from rotations import briefRot, dispBriefRotHist, compute_orientations
import utils

#Data setup

DATA_PARENT_DIR = '/content/'

if not os.path.exists(DATA_PARENT_DIR):
  raise RuntimeError('DATA_PARENT_DIR does not exist: ', DATA_PARENT_DIR)

RES_DIR = os.path.join(DATA_PARENT_DIR, 'results')
if not os.path.exists(RES_DIR):
  os.mkdir(RES_DIR)
  print('made directory: ', RES_DIR)


#paths different files are saved to
# OPTIONAL:
# feel free to change if funning locally
ROT_MATCHES_PATH = os.path.join(RES_DIR, 'brief_rot_test.pkl')
ROT_INV_MATCHES_PATH = os.path.join(RES_DIR, 'ec_brief_rot_inv_test.pkl')
AR_VID_FRAMES_PATH = os.path.join(RES_DIR, 'q_3_1_frames.npy')
AR_VID_FRAMES_EC_PATH = os.path.join(RES_DIR, 'q_3_2_frames.npy')

HW3_SUBDIR = 'hw3_data'
DATA_DIR = os.path.join(DATA_PARENT_DIR, HW3_SUBDIR)
ZIP_PATH = DATA_DIR + '.zip'
if not os.path.exists(DATA_DIR):
  !wget 'https://www.andrew.cmu.edu/user/hfreeman/data/16720_spring/hw3_data.zip' -O $ZIP_PATH
  !unzip -qq $ZIP_PATH -d $DATA_PARENT_DIR





#Basic Matching
image1_name = "pano_left.jpg"
image2_name = "pano_right.jpg"
ratio = 0.7
sigma = 0.15

image1_path = os.path.join(DATA_DIR, image1_name)
image2_path = os.path.join(DATA_DIR, image2_name)

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

#bgr to rgb
if len(image1.shape) == 3 and image1.shape[2] == 3:
  image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

if len(image2.shape) == 3 and image2.shape[2] == 3:
  image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

displayMatched(image1, image2, ratio, sigma)

#Ratio ablation

image1_name = "hp_cover.jpg"
image2_name = "hp_desk.png"

image1_path = os.path.join(DATA_DIR, image1_name)
image2_path = os.path.join(DATA_DIR, image2_name)

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

#bgr to rgb
if len(image1.shape) == 3 and image1.shape[2] == 3:
  image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

if len(image2.shape) == 3 and image2.shape[2] == 3:
  image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

if image1.shape[:2] != image2.shape[:2]:  # Compare height and width
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
# ===== your code here! =====
# Experiment with different sigma and ratio values.
# Use displayMatches to visualize.
# Include the matched feature figures in the write-up.

ratios = [0.6, 0.7, 0.8]
sigma = 0.15  # Fixed sigma for this experiment

for ratio in ratios:
    displayMatched(image1, image2, ratio, sigma)





# Displaying the matches 

image1_name = "hp_cover.jpg"
image2_name = "hp_desk.png"

image1_path = os.path.join(DATA_DIR, image1_name)
image2_path = os.path.join(DATA_DIR, image2_name)

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

#bgr to rgb
if len(image1.shape) == 3 and image1.shape[2] == 3:
  image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

if len(image2.shape) == 3 and image2.shape[2] == 3:
  image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


sigmas = [0.1, 0.3, 0.5]
ratio = 0.8

for sigma in sigmas:
    displayMatched(image1, image2, ratio, sigma)




# Rotation variance checking
min_deg = 0
max_deg = 360
deg_inc = 10

# Brief feature descriptor and Fast feature detector paremeters
# (change these if you want to use different values)
ratio = 0.7
sigma = 0.15

# image to rotate and match
# (no need to change this but can if you want to experiment)
filename = 'hp_cover.jpg'

# Call briefRot
briefRot(min_deg, max_deg, deg_inc, ratio, sigma, filename)
briefRotInvEc(min_deg, max_deg, deg_inc, 0.8, 0.03, filename)

#Ablation for max_iters and inlier_tols

max_iters_list = [100, 500, 700, 1000]
inlier_tol_list = [1.0, 5.0, 10.0, 15.0]

def ablation_study(ratio, sigma, max_iters, inlier_tol):


    composite_img = warpImage(ratio, sigma, max_iters, inlier_tol)
    print(f'max_iters={max_iters}, inlier_tol={inlier_tol}')


    if composite_img is None:
        print("Error: warpImage returned None.")
        return


    if len(composite_img.shape) == 2:
        composite_img = np.expand_dims(composite_img, axis=-1)


    print(f'composite_img dtype: {composite_img.dtype}, shape: {composite_img.shape}')


    if composite_img.dtype != np.uint8:
        composite_img = (composite_img - np.min(composite_img)) / (np.max(composite_img) - np.min(composite_img))
        composite_img = (composite_img * 255).astype(np.uint8)


    plt.figure(figsize=(8, 6))
    plt.imshow(composite_img)
    print(f'max_iters={max_iters}, inlier_tol={inlier_tol}')
    plt.title(f'Composite Image\nmax_iters={max_iters}, inlier_tol={inlier_tol}')
    plt.axis('off')
    plt.show()


ratio = 0.7
sigma = 0.15

for max_iters in max_iters_list:
    for inlier_tol in inlier_tol_list:
        ablation_study(ratio, sigma, max_iters, inlier_tol)
#Panorama

cliff_left = io.imread(os.path.join(DATA_DIR, 'cliff_left.jpg'))
cliff_right = io.imread(os.path.join(DATA_DIR,'cliff_right.jpg'))


print("Original Images are:")
print("cliff_left.jpg")
plt.imshow(cliff_left)
plt.show()
print("cliff_right.jpg")
plt.imshow(cliff_right)
plt.show()
left_im_path = os.path.join(DATA_DIR, 'cliff_left.jpg')
left_im = skimage.io.imread(left_im_path)
right_im_path = os.path.join(DATA_DIR, 'cliff_right.jpg')
right_im = skimage.io.imread(right_im_path)

# Feel free to adjust as needed
ratio = 0.7
sigma = 0.15
max_iters = 600
inlier_tol = 10.0

panorama_im = createPanorama(left_im, right_im, ratio, sigma, max_iters, inlier_tol)

plt.imshow(panorama_im)
plt.axis('off')
plt.show()








left_im_path = os.path.join(DATA_DIR, 'pano_left.jpg')
left_im = skimage.io.imread(left_im_path)
right_im_path = os.path.join(DATA_DIR, 'pano_right.jpg')
right_im = skimage.io.imread(right_im_path)

# Feel free to adjust as needed
ratio = 0.7
sigma = 0.15
max_iters = 600
inlier_tol = 10.0

panorama_im = createPanorama(left_im, right_im, ratio, sigma, max_iters, inlier_tol)

plt.imshow(panorama_im)
plt.axis('off')
plt.show()

left_im_path = os.path.join(DATA_DIR, 'cmu_left.jpg')
left_im = skimage.io.imread(left_im_path)
right_im_path = os.path.join(DATA_DIR, 'cmu_right.jpg')
right_im = skimage.io.imread(right_im_path)

# Feel free to adjust as needed
ratio = 0.7
sigma = 0.15
max_iters = 600
inlier_tol = 1.0

panorama_im = createPanorama(left_im, right_im, ratio, sigma, max_iters, inlier_tol)

plt.imshow(panorama_im)
plt.axis('off')
plt.show()
