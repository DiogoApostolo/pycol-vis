

import cv2
import numpy as np


from scipy.linalg import eigh
from .embedding_models import EfficientNetLite0EmbeddingModel, MobileNetV3EmbeddingModel, CNNEmbeddingModel
from ..datasets.dataset_class import ImageDataset


import torch
from torch.utils.data import DataLoader

from ..datasets.dataset_class import ImageDataset
from ..utils.utils import load_image
from tqdm.auto import tqdm
from skimage.feature import graycomatrix, graycoprops

from sklearn.decomposition import PCA


def efficientnet_preprocess(img):
    '''
    Auxiliary function that preprocesses the images to fit efficient-nets expected format
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))

    return torch.tensor(img, dtype=torch.float32)

def mobilenet_preprocess(img):
    '''
    Auxiliary function that preprocesses the images to fit mobilenet expected format
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32)





def setup_cnn(image_shape,num_classes,images=None,depth=2,epochs=10,batch_size=32,train_model=True):
    '''
    Create and optionally train a CNNEmbeddingModel.
    '''

    if(depth < 1):
        raise ValueError("depth must be at least 1.")
    if(epochs < 1):
        raise ValueError("epochs must be at least 1.")
   

    model = CNNEmbeddingModel(image_shape=image_shape,num_classes=num_classes,depth=depth)

    if(train_model):
        if(images is None):
            raise ValueError("images must be provided when train_model=True")

        model.train_model(images, epochs=epochs, batch_size=batch_size)

    return model

EMBEDDING_MODELS = {

    "efficient_net": (
        EfficientNetLite0EmbeddingModel,
        efficientnet_preprocess
    ),

    "mobile_net": (
        MobileNetV3EmbeddingModel,
        mobilenet_preprocess
    )
}

embdedding_types = list(EMBEDDING_MODELS.keys()) + ["current", "raw", "CNN", "histogram_texture", "patch_distribution", "gradient", "hog_color"]


def extract_torch_embeddings(image_paths,model,preprocess_fn,batch_size=4,num_workers=0,device=None,output_path="embeddings.dat"):
    
    
    if hasattr(model, 'to'):
        model = model.to("cpu")
        if hasattr(model, 'encoder') and model.encoder is not None:
            model.encoder = model.encoder.to("cpu")

    if(device is None):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    if hasattr(model, 'encoder') and model.encoder is not None:
        model.encoder = model.encoder.to(device)
        
    model.eval()

    dataset = ImageDataset(image_paths, preprocess_fn)

    pin_memory = torch.cuda.is_available()

    loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory)

    num_images = len(image_paths)

    with torch.inference_mode():
        for batch in loader:
            
            batch = batch.to(device)
            sample_emb = model(batch[:1])
            break

    embedding_dim = sample_emb.shape[1]

    embeddings = np.memmap(output_path,dtype=np.float32,mode='w+',shape=(num_images, embedding_dim))
    start = 0

    pbar = tqdm(total=num_images,desc="extracting embeddings")

    with torch.inference_mode():
        for batch in loader:
            
            batch = batch.to(device)
            batch_embeddings = model(batch).cpu().numpy()

            end = start + len(batch_embeddings)
            embeddings[start:end] = batch_embeddings
            start = end

            pbar.update(len(batch_embeddings))

    pbar.close()
    embeddings.flush()

    clean_numpy_array = np.array(embeddings, dtype=np.float32)
    
    if hasattr(embeddings, '_mmap'):
        embeddings._mmap.close()
        
    try:
        import os
        if os.path.exists(output_path):
            os.remove(output_path)
    except Exception:
        pass

    return clean_numpy_array


def generate_embeddings(image_paths,emb_type="efficient_net",batch_size=32,num_workers=0,device=None):
    '''
    Generate embeddings for a list of image paths using a specified embedding model.
    '''
    if(emb_type not in EMBEDDING_MODELS):

        raise ValueError(f"Unknown embedding type: {emb_type}")

    model_cls, preprocess_fn = EMBEDDING_MODELS[emb_type]
    model = model_cls()
    embeddings = extract_torch_embeddings(image_paths=image_paths,model=model,preprocess_fn=preprocess_fn,batch_size=batch_size,num_workers=num_workers,device=device)

    return embeddings


def get_roi(img):
    edges = cv2.Canny(img, 100, 200)

    ys, xs = np.where(edges > 0)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    roi = img[y1:y2, x1:x2]

    return roi


def extract_patch_distribution_features(image_paths,patch_size=5,n_patches=200, roi=True):
    features = []

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if (img is None):
            continue

        if(roi):
            img = get_roi(img)

        h, w = img.shape
        patches = []

        for _ in range(n_patches):
            r = np.random.randint(0, h - patch_size)
            c = np.random.randint(0, w - patch_size)

            patch = img[r:r + patch_size, c:c + patch_size]

            patches.append(patch.flatten().astype(np.float32))

        patches = np.asarray(patches)

        patch_means = patches.mean(axis=1)
        patch_vars = patches.var(axis=1)

        pca = PCA(n_components=3)
        pca.fit(patches)

        features.append([
            patch_means.mean(),
            patch_means.std(),
            patch_vars.mean(),
            patch_vars.std(),
            *pca.explained_variance_ratio_
        ])

    return np.array(features)


def extract_hog_color(image_paths, roi=True):

    features = []

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        hog_features = hog_texture_features(gray)

        color_features = extract_color_features(img)

        feat = np.concatenate(
            [
                hog_features,
                color_features
            ]
        )

        features.append(feat)

    return np.array(features)

def extract_color_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mean_h = np.mean(hsv[:, :, 0])
    mean_s = np.mean(hsv[:, :, 1])
    mean_v = np.mean(hsv[:, :, 2])

    std_h = np.std(hsv[:, :, 0])
    std_s = np.std(hsv[:, :, 1])
    std_v = np.std(hsv[:, :, 2])

    return np.array(
        [
            mean_h,
            mean_s,
            mean_v,
            std_h,
            std_s,
            std_v
        ],
        dtype=np.float32
    )


def hog_texture_features(gray):
    gray = cv2.resize(gray, (64, 64))

    features = []

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)

    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx**2 + gy**2)

    ang = np.rad2deg(np.arctan2(gy, gx))

    ang = (ang + 180) % 180

    h, w = gray.shape

    for r in range(2):
        for c in range(2):
            y0 = r * h // 2
            y1 = (r + 1) * h // 2

            x0 = c * w // 2
            x1 = (c + 1) * w // 2

            hist, _ = np.histogram(
                ang[y0:y1, x0:x1],
                bins=8,
                range=(0, 180),
                weights=mag[y0:y1, x0:x1]
            )

            hist = hist.astype(np.float32)

            hist /= (hist.sum() + 1e-10)

            features.extend(hist)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    features.append(graycoprops(glcm, "contrast")[0, 0])
    features.append(graycoprops(glcm, "homogeneity")[0, 0])
    features.append(graycoprops(glcm, "energy")[0, 0])
    features.append(graycoprops(glcm, "correlation")[0, 0])

    p = glcm[:, :, 0, 0]

    features.append(-np.sum(p * np.log2(p + 1e-10)))

    features.append(np.mean(mag))
    features.append(np.std(mag))
    features.append(np.mean(mag > np.percentile(mag, 75)))

    return np.asarray(features, dtype=np.float32)



def extract_histogram_texture_features(image_paths, roi=True):
    features = []

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if (img is None):
            continue

        if(roi):
            img = get_roi(img)

        pixels = img.flatten().astype(np.float64)

        mean = np.mean(pixels)
        std = np.std(pixels)

        centered = pixels - mean

        skewness = np.mean(centered ** 3) / (std ** 3 + 1e-10)
        kurtosis = np.mean(centered ** 4) / (std ** 4 + 1e-10)

        hist = np.bincount(img.ravel(), minlength=256)
        p = hist / hist.sum()

        entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))

        edges = cv2.Canny(img, 100, 200)
        edge_density = np.mean(edges > 0)

        glcm = graycomatrix(img, distances=[1], angles=[0], levels=256,
                            symmetric=True, normed=True)

        contrast = graycoprops(glcm, "contrast")[0, 0]
        homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
        energy = graycoprops(glcm, "energy")[0, 0]
        correlation = graycoprops(glcm, "correlation")[0, 0]

        features.append([
            mean,
            std,
            skewness,
            kurtosis,
            entropy,
            edge_density,
            contrast,
            homogeneity,
            energy,
            correlation
        ])

    return np.array(features)


def extract_gradient_features(image_paths,n_bins=8, roi=True):
    features = []

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if (img is None):
            continue

        if(roi):
            img = get_roi(img)

        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        orientation = np.arctan2(gy, gx)

        mean_mag = magnitude.mean()
        std_mag = magnitude.std()
        max_mag = magnitude.max()

        hist_mag, _ = np.histogram(magnitude, bins=32)

        p = hist_mag / (hist_mag.sum() + 1e-10)

        grad_entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))

        orient_hist, _ = np.histogram(orientation,bins=n_bins,range=(-np.pi, np.pi),density=True)

        features.append([
            mean_mag,
            std_mag,
            max_mag,
            grad_entropy,
            *orient_hist
        ])

    return np.array(features)


def embed_images(image_paths, feature_embeddings=None, model=None, emb_type='efficient_net', layer_index=-1, num_workers=0, device=None):


    if(emb_type not in embdedding_types):
        raise ValueError(f"Unknown embedding type: {emb_type}")
    #validate inputs
    if(layer_index < -1):
        raise ValueError("layer_index must be -1 or a non-negative integer.")
    if(num_workers < 0):
        raise ValueError("num_workers must be a non-negative integer.")
    if(image_paths is None or len(image_paths) == 0):
        raise ValueError("image_paths must be a non-empty list of image paths.")
    


    if(emb_type == "current"):

        if(feature_embeddings is None):
            print("No current embeddings found.")
            return None

        return feature_embeddings

    elif(emb_type == "raw"):

        feature_embeddings = []

        for image_path in image_paths:

            img = load_image(image_path)
            feature_embeddings.append(img.flatten())

        embeddings = np.array(feature_embeddings)

    elif(emb_type == "CNN"):
        if(model is None):
            raise ValueError("model must be provided when emb_type='CNN'")
        
        print("Extracting CNN embeddings...")
        embeddings = model.get_feature_embeddings_all(image_paths=image_paths, layer_index=layer_index)
    elif(emb_type == "histogram_texture"):
        print("Extracting histogram and texture features...")
        embeddings = extract_histogram_texture_features(image_paths=image_paths)
    elif(emb_type == "patch_distribution"):
        print("Extracting patch distribution features...")
        embeddings = extract_patch_distribution_features(image_paths=image_paths)
    elif(emb_type == "gradient"):
        print("Extracting gradient features...")
        embeddings = extract_gradient_features(image_paths=image_paths)
    elif(emb_type == 'hog_color'):
        print("Extracting hog_color features...")
        embeddings = extract_hog_color(image_paths=image_paths)
    elif(emb_type in EMBEDDING_MODELS):
        print(f"Extracting {emb_type} embeddings...")
        embeddings = generate_embeddings(image_paths=image_paths, emb_type=emb_type, num_workers=num_workers, device=device)
    else:
        raise ValueError(f"Unknown embedding type: {emb_type}")

    return embeddings