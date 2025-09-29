
import os
import cv2
import time
from PIL import Image
from ultralytics import YOLO
import torch
from utils import get_embedding, average_product_embeddings, match_crop_with_products, model, preprocess, device


def run_problem1(product_dir="products", shelf_dir="shelves",
                 output_file="solution_1.txt", threshold=0.8,
                 visualize=False, vis_out="debug_vis"):
    """
    Solve Problem 1:
    For each product, find occurrences in shelf images and record bounding boxes.
    """

    # Step 1: Compute product embeddings
    print("📦 Computing product embeddings...")
    product_embeddings = average_product_embeddings(product_dir)

    # Step 2: Load YOLO for region proposals
    print("🔍 Loading YOLOv8 model for object detection...")
    yolo_model = YOLO("yolov8m.pt")  # COCO-pretrained

    results = []

    # Make visualization folder
    if visualize:
        os.makedirs(vis_out, exist_ok=True)

    # Step 3: Iterate over shelf images
    shelf_files = sorted([f for f in os.listdir(shelf_dir) if f.endswith(".jpg")])
    total_shelves = len(shelf_files)

    # Start timer
    start_time = time.time()

    for sid, fname in enumerate(shelf_files, start=1):
        shelf_path = os.path.join(shelf_dir, fname)
        detections = yolo_model.predict(shelf_path, verbose=False)

        img = cv2.imread(shelf_path)

        for det in detections[0].boxes.xyxy.cpu().numpy():  # [x_min, y_min, x_max, y_max]
            x1, y1, x2, y2 = map(int, det)
            crop = Image.open(shelf_path).crop((x1, y1, x2, y2))

            with torch.no_grad():
                crop_emb = model.encode_image(
                    preprocess(crop).unsqueeze(0).to(device)
                ).detach().cpu().numpy()

            matches = match_crop_with_products(crop_emb, product_embeddings, threshold=threshold)

            for pid, sim in matches:
                results.append(f"{pid},{sid},{x1},{y1},{x2},{y2}")

                # Visualization (optional)
                if visualize and sid <= 5:  # only first 5 shelves for debugging
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"PID {pid} ({sim:.2f})", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if visualize and sid <= 5:
            cv2.imwrite(os.path.join(vis_out, f"vis_db{sid}.jpg"), img)

        # --- Progress Logging ---
        num_dets = len(detections[0].boxes.xyxy)
        elapsed = time.time() - start_time
        avg_time = elapsed / sid
        eta = avg_time * (total_shelves - sid)

        print(
            f"[{sid}/{total_shelves}] {fname} "
            f"| Detections: {num_dets} "
            f"| Elapsed: {elapsed/60:.1f} min "
            f"| ETA: {eta/60:.1f} min"
        )

    # Step 4: Save results
    with open(output_file, "w") as f:
        f.write("\n".join(results))

    print(f"\n🎯 Done! Results saved to {output_file}")
    if visualize:
        print(f"🖼️ Visualization samples saved to {vis_out}/")


if __name__ == "__main__":
    # Run with default paths
    run_problem1(product_dir="products",
                 shelf_dir="shelves",
                 output_file="solution_1.txt",
                 threshold=0.8,
                 visualize=True)