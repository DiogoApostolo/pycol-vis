

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





def setup_cnn(image_shape,num_classes,images=None,depth=2,epochs=10,train_model=True):
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

        model.train_model(images,epochs=epochs)

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


def embed_images(image_paths, feature_embeddings=None, model=None, emb_type='efficient_net', layer_index=-1, num_workers=0, device=None):


    if(emb_type not in EMBEDDING_MODELS and emb_type != "current" and emb_type != "raw" and emb_type != "CNN"):
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

    elif(emb_type in EMBEDDING_MODELS):
        print(f"Extracting {emb_type} embeddings...")
        embeddings = generate_embeddings(image_paths=image_paths, emb_type=emb_type, num_workers=num_workers, device=device)
    else:
        raise ValueError(f"Unknown embedding type: {emb_type}")

    return embeddings