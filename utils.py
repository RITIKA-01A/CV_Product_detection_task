import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Setup CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def get_embedding(image_path: str):
    """Extract CLIP embedding from image."""
    img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img).detach()
    return emb.cpu().numpy()


def average_product_embeddings(product_dir="products", num_products=100):
    """
    Compute average embedding for each product (3 images each).
    Returns dict: {product_id: embedding}
    """
    product_embeddings = {}
    for pid in range(1, num_products + 1):
        embs = []
        for k in range(3):
            idx = (pid - 1) * 3 + k + 1  # qr1..qr300
            path = f"{product_dir}/qr{idx}.jpg"
            embs.append(get_embedding(path))
        product_embeddings[pid] = np.mean(embs, axis=0)
    return product_embeddings


def match_crop_with_products(crop_emb, product_embeddings, threshold=0.8):
    """
    Compare crop embedding with all product embeddings.
    Return list of matching product IDs with similarity above threshold.
    """
    matches = []
    for pid, p_emb in product_embeddings.items():
        sim = cosine_similarity(crop_emb, p_emb)[0][0]
        if sim >= threshold:
            matches.append((pid, sim))
    return matches