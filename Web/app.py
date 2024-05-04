from flask import Flask, render_template, request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.pyplot.switch_backend('Agg')
import imageio
import os
import warnings
from PIL import Image
import torchvision.transforms.functional as F_pil


# Disable OpenCL to prevent crashes with certain Intel GPUs
cv2.ocl.setUseOpenCL(False)

# Suppress warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Get the uploaded files
        train_image = request.files['train_image']
        query_image = request.files['query_image']

        # Save the uploaded images to the server
        train_path = './uploads/train.jpg'
        query_path = './uploads/query.jpg'
        train_image.save(train_path)
        query_image.save(query_path)

        # Your image processing code here...
        # Define feature extraction and matching parameters
        feature_extraction_algo = request.form['feature_extraction_algo']
        feature_to_match = 'knn'

        # Read images
        train_photo = cv2.imread(train_path)
        train_photo = cv2.cvtColor(train_photo, cv2.COLOR_BGR2RGB)
        train_photo_gray = cv2.cvtColor(train_photo, cv2.COLOR_RGB2GRAY)
        
        query_photo = cv2.imread(query_path)
        query_photo = cv2.cvtColor(query_photo, cv2.COLOR_BGR2RGB)
        query_photo_gray = cv2.cvtColor(query_photo, cv2.COLOR_RGB2GRAY)

        # Resize images to have the same dimensions
        min_height = min(train_photo.shape[0], query_photo.shape[0])
        min_width = min(train_photo.shape[1], query_photo.shape[1])
        train_photo = cv2.resize(train_photo, (min_width, min_height))
        query_photo = cv2.resize(query_photo, (min_width, min_height))

        # Feature extraction and matching
        if feature_extraction_algo == 'lucas-kanade':
            warped_image = image_alignment_with_optical_flow(train_photo, query_photo)
        else:
            keypoints_train_img, features_train_img = select_descriptor_methods(train_photo_gray, method=feature_extraction_algo)
            keypoints_query_img, features_query_img = select_descriptor_methods(query_photo_gray, method=feature_extraction_algo)


            # Feature matching
            if feature_to_match == 'bf':
                matches = key_points_matching(features_train_img, features_query_img, method=feature_extraction_algo)
            elif feature_to_match == 'knn':
                matches = key_points_matching_KNN(features_train_img, features_query_img, ratio=0.75, method=feature_extraction_algo)

            # Homography estimation and stitching
            M = homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh=4)
            (matches, Homography_Matrix, status) = M

            # Warp images and blend
            warped_image = warp_and_blend(train_photo, query_photo, Homography_Matrix)

            

        # Save the result
        result_path = "./static/output/horizontal_panorama_img_cropped.jpeg"
        imageio.imwrite(result_path, warped_image)

        # Display the result page
        return render_template('result.html', result=result_path)
    
    except Exception as e:
        # Handle any errors
        error_message = str(e)
        return render_template('error.html', error_message=error_message)



def image_alignment_with_optical_flow(image1, image2):
    # Resize images to a manageable size for feature detection and optical flow
    resize_factor = 0.5  # Adjust this factor based on your image size and performance
    resized_image1 = cv2.resize(image1, None, fx=resize_factor, fy=resize_factor)
    resized_image2 = cv2.resize(image2, None, fx=resize_factor, fy=resize_factor)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(resized_image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(resized_image2, cv2.COLOR_RGB2GRAY)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(30, 30),  # Adjust winSize for better tracking of keypoints
                     maxLevel=3,        # Increase maxLevel for more pyramid levels
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Detect keypoints in the first image (left image)
    initial_keypoints = cv2.goodFeaturesToTrack(gray1, maxCorners=200, qualityLevel=0.01, minDistance=10)

    # Calculate optical flow
    keypoints2, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, initial_keypoints, None, **lk_params)

    # Filter out keypoints with a low status
    good_keypoints1 = initial_keypoints[status == 1]
    good_keypoints2 = keypoints2[status == 1]

    # Compute the transformation matrix using RANSAC
    M, _ = cv2.findHomography(good_keypoints1, good_keypoints2, cv2.RANSAC)

    # Warp the second image (right image) onto the first image (left image)
    warped_image = cv2.warpPerspective(image2, M, (int(image1.shape[1] * resize_factor), int(image1.shape[0] * resize_factor)))

    # Resize the warped image back to the original size
    final_warped_image = cv2.resize(warped_image, (image1.shape[1], image1.shape[0]))

    return final_warped_image





def select_descriptor_methods(image, method=None):
    assert method is not None, "Please define a feature descriptor method. Accepted values are: 'sift', 'surf', 'brisk', 'orb'"
    if method == 'sift':
        descriptor = cv2.SIFT_create()
        keypoints, features = descriptor.detectAndCompute(image, None)
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
        keypoints, features = descriptor.detectAndCompute(image, None)
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
        keypoints, features = descriptor.detectAndCompute(image, None)
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        keypoints, features = descriptor.detectAndCompute(image, None)
    elif method == 'akaze':
        descriptor = cv2.AKAZE_create()
        keypoints, features = descriptor.detectAndCompute(image, None)
    else:
        raise ValueError(f"Unsupported feature extraction method: {method}")
    return keypoints, features

def key_points_matching(features_train_img, features_query_img, method):
    bf = create_matching_object(method, crossCheck=True)
    best_matches = bf.match(features_train_img, features_query_img)
    rawMatches = sorted(best_matches, key=lambda x: x.distance)
    return rawMatches

def key_points_matching_KNN(features_train_img, features_query_img, ratio, method):
    bf = create_matching_object(method, crossCheck=False)
    if bf is None:
        raise ValueError(f"Unsupported feature extraction method: {method}")
    
    rawMatches = bf.knnMatch(features_train_img, features_query_img, k=2)
    matches = []
    for m, n in rawMatches:
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

def create_matching_object(method, crossCheck):
    if method in ['sift', 'surf']:
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method in ['orb', 'brisk']:
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    elif method == 'akaze':
        # Create matcher object for AKAZE
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)  # Adjust norm type if needed
    else:
        raise ValueError(f"Unsupported feature extraction method: {method}")

def homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh):   
    keypoints_train_img = np.float32([keypoint.pt for keypoint in keypoints_train_img])
    keypoints_query_img = np.float32([keypoint.pt for keypoint in keypoints_query_img])
    
    if len(matches) > 4:
        points_train = np.float32([keypoints_train_img[m.queryIdx] for m in matches])
        points_query = np.float32([keypoints_query_img[m.trainIdx] for m in matches])
        
        (H, status) = cv2.findHomography(points_train, points_query, cv2.RANSAC, reprojThresh)
        return matches, H, status
    else:
        return None

def warp_and_blend(train_photo, query_photo, Homography_Matrix):
    width = query_photo.shape[1] + train_photo.shape[1]
    height = max(query_photo.shape[0], train_photo.shape[0])
    result = cv2.warpPerspective(train_photo, Homography_Matrix,  (width, height))
    mask = (result[0:query_photo.shape[0], 0:query_photo.shape[1]] == 0)
    result[0:query_photo.shape[0], 0:query_photo.shape[1]] = mask * query_photo + (1 - mask) * result[0:query_photo.shape[0], 0:query_photo.shape[1]]
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    result = result[y:y+h, x:x+w]
    return result

if __name__ == '__main__':
    app.run(debug=True)
