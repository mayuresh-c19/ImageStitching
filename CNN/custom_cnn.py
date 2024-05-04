import numpy as np
import cv2
import os
import tensorflow as tf
import tensorflow_hub as hub

def load_image(image_path):
    """Load and preprocess image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def extract_features(image_paths):
    """Extract features using DELF model."""
    # Link to model
    delf = hub.load('https://tfhub.dev/google/delf/1')
    features_list = []
    for image_path in image_paths:
        image = load_image(image_path)
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        input_dict = {
            'image': image_tensor,
            'image_scales': tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
            'max_feature_num': tf.constant(1000),
            'score_threshold': tf.constant(100.0)
        }
        features = delf.signatures['default'](**input_dict)
        locations = features['locations'].numpy()
        descriptors = features['descriptors'].numpy()
        features_list.append((locations, descriptors))
    return features_list


def match_features(query_features, train_features):
    """Match features between query and train images."""
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(query_features[1], train_features[1])
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def stitch_images(image_paths):
    """Stitch images into a panoramic image."""
    # Extract features from all images
    features = extract_features(image_paths)

    # Match features between adjacent images
    matches = []
    for i in range(len(features) - 1):
        matches.append(match_features(features[i], features[i + 1]))

    # Compute homography for each pair of matching images
    homography_matrices = []
    for i, match in enumerate(matches):
        src_pts = np.float32([features[i][0][m.queryIdx] for m in match]).reshape(-1, 1, 2)
        dst_pts = np.float32([features[i + 1][0][m.trainIdx] for m in match]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)
        homography_matrices.append(H)

    # Stitch images using homography
    stitched_image = cv2.imread(image_paths[0])
    for i, image_path in enumerate(image_paths[1:], start=1):
        image = cv2.imread(image_path)
        stitched_image = cv2.warpPerspective(stitched_image, homography_matrices[i - 1], (image.shape[1] + stitched_image.shape[1], image.shape[0]))
        stitched_image[0:image.shape[0], 0:image.shape[1]] = image

    return stitched_image


def main():
    # Paths to images
    image_paths = ["CNN/query.jpg", "CNN/train.jpg"]  # Add more image paths as needed

    # Stitch images
    panoramic_image = stitch_images(image_paths)

    # Display and save stitched image
    cv2.imshow("Panoramic Image", panoramic_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the stitched image
    output_path = "panoramic_image.jpg"
    cv2.imwrite(output_path, panoramic_image)
    print("Panoramic image saved at:", output_path)

if __name__ == "__main__":
    main()
