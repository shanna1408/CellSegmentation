import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def image_to_embedding(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  
    image = np.array(image)
    

    if len(image.shape) == 3:
        image = image.flatten() 
    elif len(image.shape) == 2: 
        image = image.flatten()
    
    return image

image_file = r"C:\Users\Shanna\Desktop\Vision\Final Project\Dataset\MoNuSeg 2018 Training Data\Tissue Images\TCGA-G9-6348-01Z-00-DX1.tif"
embeddings = []


image_path = os.path.join(image_file)
embedding = image_to_embedding(image_path)
# embeddings.append(embedding)


# embeddings = np.array(embeddings)

# embeddings = np.load('embeddings.npy')  


scaler = StandardScaler()
embeddings_standardized = scaler.fit_transform(embedding)


pca = PCA(n_components=0.95)  
reduced_embeddings = pca.fit_transform(embeddings_standardized)

print(f"Original dimensionality: {embeddings.shape[1]}")
print(f"Reduced dimensionality: {reduced_embeddings.shape[1]}")


np.save('reduced_embeddings.npy', reduced_embeddings)