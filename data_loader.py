import os
import cv2
import numpy as np

def load_and_process_data(dataset_path):
    """Loads and processes image dataset into feature vectors and labels."""
    labels_map = {
        "thumbs_up": 0, "thumbs_down": 1,
        "thumbs_left": 2, "thumbs_right": 3, "fist_closed": 4
    }
    
    features, labels = [], []
    feature_dim = None
    
    for label_name, label in labels_map.items():
        folder = os.path.join(dataset_path, label_name)
        if not os.path.exists(folder):
            raise IOError("Directory not found: "+folder)
            
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            try:
                img = cv2.imread(path)
                if img is None:
                    raise IOError("Failed to read "+path)
                    
                # Convert to grayscale and process
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                pooled = cv2.resize(gray, (gray.shape[1]//4, gray.shape[0]//4), interpolation=cv2.INTER_AREA)
                sobelx = cv2.Sobel(pooled, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(pooled, cv2.CV_64F, 0, 1, ksize=3)
                edges = np.sqrt(sobelx**2 + sobely**2)
                
                # Feature extraction
                fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(edges))) + 1)
                spatial = edges.flatten()
                combined = np.hstack((fft.flatten(), spatial))
                
                # Validate feature dimension consistency
                if feature_dim is None:
                    feature_dim = len(combined)
                elif len(combined) != feature_dim:
                    raise ValueError("Inconsistent feature dimension at "+path)
                    
                features.append(combined)
                labels.append(label)
                
            except Exception as e:
                print("Error processing "+path+": "+str(e))
                continue
                
    if not features:
        raise ValueError("No valid images processed")
        
    return np.array(features), np.array(labels), feature_dim

