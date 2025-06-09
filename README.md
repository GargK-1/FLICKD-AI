# FLICKD-AI

# Flickd AI Hackathon: Fashion Video Intelligence

This project processes fashion-related videos to extract and analyze visual content using machine learning. The pipeline consists of three core modules:

1. **Object Detection** – Detects fashion items in videos using YOLOv8.
2. **Product Matching** – Matches detected items to a product catalog using CLIP and FAISS.
3. **Vibe Classification** – Classifies the overall "vibe" of the video using rule-based or ML-based NLP methods.

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/GargK-1/FLICKD-AI.git
cd FLICKD-AI
```

🧩 Step 1: Object Detection
Run YOLOv8 to extract fashion item detections from all videos:

```bash
python OBJECT DETECTION.py --input_dir Video Frames --output_dir detections.json
```

Outputs a CSV per video with:

Frame number

Detected class label

Confidence score

Bounding box (x1, y1, x2, y2)

🖼️ Step 2: Product Matching
Run CLIP and FAISS to match cropped detected items to catalog products:

```bash
python PRODUCT_MATCHING.py --detections_dir detections.json --catalog_path images.csv --output_dir matched
```

Uses OpenAI CLIP to compute image embeddings.

Outputs a CSV per video with best match per detected object.

🎭 Step 3: Vibe Classification
Classify the overall vibe of each video using keyword or ML-based NLP:

```bash
python VIBE_CLASSIFICATION.py --input_dir videos --vibe_definitions vibeslist.json --output_dir vibes
```

Returns top 1 or top-3 predicted vibes for each video.
