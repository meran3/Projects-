# Smart Crowd Management Using Machine Learning

A crowd-counting system that estimates how many people appear in a surveillance frame and raises a safety alert when density passes a threshold — a proposed ML module for proactive crowd management in the Holy Mosque and Holy Sites during Hajj and Umrah.

Machine Learning course project, Umm Al-Qura University. Supervised by Dr. Afaf Almehmadi.

📄 **Full report:** [ML_Project_Report.pdf](ML_Project_Report.pdf) — includes a literature review on low-latency and interpretable crowd-management systems.

**Team:** Aram Alouni · Mariam Alzahrani · Mashael Al-Yazidi · Meran Al-Daadi · Jana Alsamdani

## Pipeline

1. **Data:** the public [Mall Dataset](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html) — 2,000 surveillance frames with ground-truth head counts. It is far less dense than Hajj scenes, but it is a standard benchmark small enough to train within project constraints.
2. **Preprocessing:** frames resized to 128×128 and normalised; 80/20 train/test split (1,600 / 400).
3. **Model:** a custom CNN (3 Conv + MaxPooling blocks → Dense) that regresses the crowd count directly.
4. **Training:** 20 epochs, then continued training with EarlyStopping and ModelCheckpoint.
5. **Safety layer:** frames whose predicted count exceeds a configurable threshold (default 40) trigger a high-density warning.
6. **Visualisation:** heatmap overlays and a ground-truth vs prediction table.

## Results (test set, 400 frames)

| Metric | Value |
|---|---|
| MAE | 1.96 people |
| RMSE | 2.52 |
| MAPE | 6.26% |

> The heatmaps illustrate overall density intensity based on the predicted count; they are not true per-person localisation, since the model predicts a single count per frame.

## Running the notebook

Open `ML_Project.ipynb` in Google Colab and run all cells. The dataset is downloaded automatically.

**Stack:** Python, TensorFlow / Keras, OpenCV, scikit-learn, pandas, Matplotlib.
