# Grocery Shelf Product Detection – Deep Learning Challenge

## 1. Overview
This project addresses **Problem 1** of the Deep Learning Challenge: detecting and locating products from a set of close-up images within grocery shelf images. The task is to identify all shelf images containing each product and record **bounding boxes** for every occurrence.

- **Shelf images:** 3,153 images (`db1.jpg` … `db3153.jpg`) with multiple products.
- **Product images:** 300 images of 100 unique products (`qr1.jpg` … `qr300.jpg`), 3 angles per product.

The solution leverages **YOLOv8** for region proposals and **CLIP (ViT-B/32)** embeddings for visual similarity matching.

---

## 2. Dataset Overview

### Shelf Images
- Captured in real grocery stores with proper lighting.  
- Contain multiple products, sometimes with multiple instances of the same product.  
- Filenames: `db1.jpg` … `db3153.jpg`.  

### Product Images
- 300 images corresponding to 100 unique products.  
- Each product has three images captured from different angles to account for appearance variation.  
- Filenames: `qr1.jpg` … `qr300.jpg`.  

---

## 3. Approach

The solution follows a **two-stage pipeline** combining detection and embedding-based matching:

### Stage 1 – Product Embeddings
- Each product image is encoded using **CLIP (ViT-B/32)**.  
- Embeddings of the three images per product are **averaged** to form a single representative embedding.  
- This ensures **angle-invariant matching** with shelf crops.

### Stage 2 – Shelf Detection & Matching
- **YOLOv8 (pre-trained on COCO)** generates candidate bounding boxes in shelf images.  
- Each detected crop is encoded using CLIP.  
- **Cosine similarity** is computed between crop embeddings and product embeddings.  
- A match is considered valid if similarity ≥ **0.8**.  
- Each valid match is saved in `solution_1.txt` in the format:
- product_id, shelf_id, x_min, y_min, x_max, y_max



### Key Functions
- `get_embedding(image_path)` – Extracts CLIP embedding for a single image.  
- `average_product_embeddings(product_dir)` – Averages embeddings for the three images of each product.  
- `match_crop_with_products(crop_emb, product_embeddings, threshold)` – Computes cosine similarity to find matches.  

### Dependencies
- Python 3.8+, PyTorch, CLIP, Ultralytics YOLOv8, OpenCV, PIL, scikit-learn, numpy  

**Run the pipeline**
```bash
python main.py
```


### 5. Results

- Generated `solution_1.txt` containing all detected product instances.
- Optional visualization confirmed accurate detection for sample shelves.

**Example output:**

52,1731,1884,954,2190,1419
