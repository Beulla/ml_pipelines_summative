# Early Dyslexia Screening from Children's Handwriting — ML Pipeline

**Author:** Beulla Jeanne Francoise Rugero
**Module:** Machine Learning Pipeline (BSE), African Leadership University
**Builds on:** [Early Identification of At-Risk Learners](https://github.com/Beulla/model_training_and_evaluation) (Introduction to ML, tabular OULAD data)

## Video demo and live links

| | Link |
|---|---|
| YouTube demo | _add after recording_ |
| Live UI (Streamlit) | https://dyslexia-screening-ui.onrender.com/ |
| Live API (FastAPI docs) | https://dyslexia-screening-api.onrender.com (`/docs`, health check `/health`) |

## Project description

My mission is to help create safe havens for children from unstable homes, and the core idea behind it is to **find vulnerable children early, so help reaches them before they fall behind**. My Introduction to ML project applied that idea to university learners, using tabular data (OULAD) to predict who would fail or withdraw.

This project applies the same idea to **younger children and non-tabular data**. Children with dyslexia often go unnoticed until they are already behind in reading and writing. Teachers already collect handwriting every day, so a photo of a single handwritten character is a cheap first screen.

The pipeline takes **an image of one handwritten character** (an English letter, an Urdu letter or a digit) and predicts whether the writing pattern matches children **with dyslexia** or **without**. A positive result means *refer this child for a proper assessment*, never *this child has dyslexia*.

> Educational screening prototype. It is not a diagnostic tool.

### Dataset

| Property | Value |
|---|---|
| Name | *Multilingual AI-Based Dyslexia Detection* (bilingual English–Urdu handwriting) |
| Source | Mendeley Data, DOI [10.17632/b37s8b94nv.1](https://doi.org/10.17632/b37s8b94nv.1), CC BY 4.0 |
| Content | 852 character images from 100 children aged 6–10 (schools in Narowal, Punjab, Pakistan) |
| Classes | Dyslexic (426) / Non-dyslexic (426) |
| Split | stratified 80/20 → 681 train / 171 test (`scripts/prepare_data.py`, seed 42) |
| Paper | Kashif et al., *Optimizing Deep Neural Models for Early Dyslexia Detection Using Novel Bilingual Handwritten Dataset*, Engineering Reports 8 (2026) e70867 |

### Results (held-out test set, 171 images)

| Model | Accuracy | Precision | Recall | Macro F1 | Sensitivity (at-risk caught) | Specificity | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Hand-crafted features (best single feature) | – | – | – | – | – | – | 0.54 |
| Baseline CNN trained from scratch | 0.596 | 0.596 | 0.596 | 0.596 | 0.588 | 0.605 | 0.683 |
| MobileNetV3-Small, transfer learning | 0.807 | 0.809 | 0.807 | 0.807 | 0.765 | 0.849 | 0.877 |
| **Production model** (above + one warm-start retrain round) | **0.813** | **0.814** | **0.813** | **0.813** | **0.788** | **0.837** | **0.875** |

Production confusion matrix: 72 / 86 non-dyslexic correctly cleared, 67 / 85 dyslexic correctly flagged. Lowering the decision threshold to 0.3 catches 85 % of at-risk children (specificity 0.73). That trade-off is a sensible choice for a screener, where a missed child costs more than an extra teacher check.

The full evaluation is in the notebook: confusion matrices, ROC curves, the classification report, a screening-threshold table, and a baseline comparison.

**Limitations.** The dataset doesn't record which child wrote each image, so one child's letters can land in both train and test. That makes the test score optimistic for brand-new children. The mix of English letters, Urdu letters and digits also differs a little between the two groups. And one character is weak evidence: a real screen would combine many characters per child.

## Repository structure

```
dyslexia-screening-pipeline/
├── README.md
├── notebook/
│   └── dyslexia_screening_pipeline.ipynb   # preprocessing, training, full evaluation, retraining demo
├── src/
│   ├── config.py           # paths, classes, thresholds
│   ├── preprocessing.py    # image preprocessing + 3 handwriting features
│   ├── model.py            # MobileNetV3 transfer model, baseline CNN, training, evaluation
│   ├── prediction.py       # single-image prediction (auto hot-reload of new models)
│   ├── train.py            # train / retrain with quality gate
│   ├── retrain.py          # upload ingestion, retrain trigger, background job
│   └── database.py         # SQLite store for uploaded images and retraining runs
├── data/
│   ├── train/{non_dyslexic,dyslexic}/   # preprocessed 128x128 PNGs
│   └── test/{non_dyslexic,dyslexic}/
├── models/
│   ├── dyslexia_cnn.pt     # PyTorch model
│   └── dyslexia_cnn.pkl    # same model, pickle (joblib)
├── api/main.py             # FastAPI service
├── ui/app.py               # Streamlit dashboard
├── locust/                 # flood test + results
├── scripts/                # data download + Locust runner
├── Dockerfile, Dockerfile.ui, docker-compose.yml, nginx.conf, render.yaml
└── requirements.txt (full), requirements-api.txt, requirements-ui.txt
```

## Setup (local, Windows PowerShell)

Requires Python 3.11+.

```powershell
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

The repo already contains the preprocessed images and the trained model, so you can start the app right away:

```powershell
# terminal 1: API on http://127.0.0.1:8000  (interactive docs at /docs)
uvicorn api.main:app --host 127.0.0.1 --port 8000

# terminal 2: UI on http://localhost:8501
streamlit run ui/app.py
```

Or double-click `RUN.bat`.

To rebuild everything from scratch:

```powershell
python scripts/prepare_data.py       # download (56 MB), verify checksum, preprocess, split
python -m src.train                  # train MobileNetV3 and save models/dyslexia_cnn.pt + .pkl
jupyter nbconvert --to notebook --execute --inplace notebook/dyslexia_screening_pipeline.ipynb
```

## How the pipeline works

| Stage | Where | What happens |
|---|---|---|
| Data acquisition | `scripts/prepare_data.py` | Download from Mendeley, SHA-256 check, stratified split |
| Data processing | `src/preprocessing.py` | EXIF fix → grayscale → auto-contrast → pad to square → 128×128. Augmentation (rotate, shift, scale, jitter; **no flips**, because mirrored letters are the very reversal errors dyslexia causes) |
| Model creation | `src/model.py` | MobileNetV3-Small pre-trained on ImageNet + new 2-class head. Baseline CNN from scratch for comparison |
| Training | `src/train.py` | Adam + weight decay, full fine-tuning (lr 5e-4, chosen by experiment), LR scheduling, early stopping on validation F1, class-weighted loss |
| Testing | notebook + `evaluate_model` | accuracy, precision, recall, F1, sensitivity, specificity, ROC-AUC, confusion matrix |
| Prediction | `src/prediction.py`, `POST /predict` | one image → class, probability, handwriting features |
| Retraining | `src/retrain.py`, `POST /retrain` | warm-start fine-tuning on old + uploaded data, then a quality gate |

### Retraining and its trigger

1. **Upload bulk data**, either in the UI tab *Upload & retrain* or with `POST /upload-data`:
   * a **zip** whose folders name the class: `dyslexic/` and `non_dyslexic/` (or `Yes/` and `No/`, as in the original dataset), or
   * **many images** plus a `class_id` (0 = non-dyslexic, 1 = dyslexic).

   Each file is validated and preprocessed (128×128 grayscale), then **saved to the SQLite database** `data/pipeline.db` with status `pending`, and added to `data/train/`.
2. **Trigger:**
   * **manual:** the *Trigger retraining* button (`POST /retrain?epochs=6`);
   * **automatic:** once ≥ 20 new images are pending (`RETRAIN_MIN_NEW_SAMPLES`), the API flags that retraining is recommended. With `AUTO_RETRAIN=true` it starts retraining on its own.
3. **Retrain:** the pending images are **read from the database** (any missing from `data/train/` are restored from the stored image). The current model is the starting point (warm start / pre-trained), and it is fine-tuned on all training data in a background thread. The run is recorded in the database.
4. **Quality gate:** the new model goes live only if its test macro-F1 doesn't drop by more than 0.02. Otherwise the old model is kept, and the UI shows before/after metrics.
5. **Hot reload:** every API replica reloads the new model file without a restart, and the uploads it used are marked `used` (with the run id) in the database.

### Database (SQLite)

`src/database.py` keeps the retraining data in `data/pipeline.db` (Python's built-in `sqlite3`, so there is nothing extra to install):

| Table | One row per | Main columns |
|---|---|---|
| `uploads` | uploaded image | original name, class, source (zip / images), preprocessed PNG (BLOB), uploaded at, status `pending` → `used`, retrain run id |
| `retrain_runs` | retraining job | reason (button / auto), epochs, new images, before/after accuracy and F1, promoted? |

You can see it in the UI (*Upload & retrain* tab), through `GET /data/uploads` and `GET /data/retrain-runs`, or by opening the file in [DB Browser for SQLite](https://sqlitebrowser.org/). On Render's free tier the disk is not persistent, so the database (like uploads and retrained models) resets when the service restarts.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, uptime, model presence |
| GET | `/model/info` | test metrics, confusion matrix, data counts, retrain queue |
| GET | `/model/history` | training curves and metadata |
| POST | `/predict` | one image (`file`) → prediction |
| POST | `/upload-data` | bulk zip (`archive`) or images (`files` + `class_id`) |
| POST | `/retrain?epochs=6` | start retraining (background) |
| GET | `/data/uploads` | uploaded images stored in the SQLite database (status, class, run id) |
| GET | `/data/retrain-runs` | retraining history from the database |
| GET | `/retrain/status` | job state, before/after metrics, promoted? |
| POST | `/evaluate` | **evaluate the live model in production** on the held-out test set |
| GET | `/visualizations/features` | per-image features for the dashboard |
| GET | `/production/latency` | mean / p95 latency of recent predictions |

Example: `curl -F "file=@data/test/dyslexic/dyslexic_0000.png" http://127.0.0.1:8000/predict`

## UI (Streamlit)

* **Sidebar:** model **up-time**, API status, current accuracy/F1, retraining-queue progress bar
* **Predict:** upload one character image → class, at-risk probability gauge, features
* **Visualisations:** class balance, the 3 handwriting features with interpretation, feature-space scatter
* **Upload & retrain:** bulk upload, the uploads table from the database, *Trigger retraining* button, live job status, before/after metrics
* **Model evaluation:** training metrics and curves, plus an *Evaluate the live model now* button (evaluation in production)
* **Monitoring:** uptime and production latency

### The three features and the story they tell

| Feature | How it is computed | Median: dyslexic vs non-dyslexic | What it means |
|---|---|---|---|
| **Ink coverage** | pencil pixels ÷ all pixels | 0.030 vs 0.027 | slightly larger, heavier, over-written letters |
| **Stroke darkness** | (paper − ink brightness) ÷ 255, a proxy for pen pressure | 0.646 vs 0.700 | lighter, less even pressure |
| **Stroke thinness** | stroke-edge pixels ÷ ink pixels | 0.617 vs 0.631 (p ≈ 0.08) | thicker or retraced strokes, going back over uncertain letters |

Every difference points the same way, toward **effortful, less automatic writing**. But the two groups overlap almost completely: each feature alone is barely better than chance. Dyslexia shows up in the **shape** of letters (reversals, malformed strokes), not in simple ink statistics. That's why the CNN, which learns shape, does much better than these features alone.

## Deployment

### Docker (local, one or more API containers)

```powershell
docker compose up --build --scale api=1   # UI http://localhost:8501, API via nginx http://localhost:8000, Locust http://localhost:8089
docker compose up --build --scale api=3   # three API replicas behind nginx round-robin
```

The API replicas share `models/` and `data/` volumes, so a model retrained in one replica is hot-reloaded by all of them.

### Cloud (Render)

1. Push this repo to GitHub.
2. On [Render](https://render.com): **New → Blueprint**, then pick the repo. `render.yaml` creates two services:
   * `dyslexia-screening-api` (Docker, `Dockerfile`, health check `/health`)
   * `dyslexia-screening-ui` (Docker, `Dockerfile.ui`)
3. Set the UI service's `API_URL` env var to the API's URL (e.g. `https://dyslexia-screening-api.onrender.com`).
4. Put both URLs at the top of this README.

The free tier sleeps after 15 minutes idle, so the first request takes about a minute.

## Flood-request simulation (Locust)

`locust/locustfile.py` simulates users who each send a handwriting image to `/predict` (10:1 ratio with `/health`), waiting 0.5–1.5 s between requests.

```powershell
# interactive UI on http://localhost:8089
locust -f locust/locustfile.py --host http://127.0.0.1:8000

# headless runs saved to locust/results/ (CSV + HTML report)
python scripts/run_locust.py --host http://127.0.0.1:8000 --containers 1 --users 10,50,100 --time 60s
```

### Results

Run on 24 November 2026 on one Windows laptop (12 logical CPU cores), with Locust on the same machine. Each run lasted 45 s. Each **uvicorn worker is a separate process with its own copy of the model**, which is the same scaling idea as separate Docker containers behind a load balancer. Docker wasn't available on the machine used for these runs. To repeat them with real containers, run `docker compose up --build --scale api=N`, then `python scripts/run_locust.py --containers N --setup docker`.

| API workers | Users | Requests | Failures | Median (ms) | p95 (ms) | p99 (ms) | Avg (ms) | Requests/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 336 | 0 | 70 | 230 | 340 | 95 | 8.0 |
| 1 | 50 | 831 | 0 | 1,100 | 1,600 | 2,000 | 1,077 | 19.0 |
| 1 | 100 | 799 | 0 | 1,900 | 8,100 | 31,000 | 2,938 | 19.6 |
| 3 | 10 | 256 | 0 | 70 | 140 | 230 | 79 | 5.9 |
| 3 | 50 | 1,587 | 0 | 150 | 440 | 690 | 190 | 36.7 |
| 3 | 100 | 1,814 | 0 | 830 | 2,000 | 2,500 | 968 | 41.8 |

**What it shows**
* **Light load (10 users):** one worker is enough. Median latency is about 70 ms either way, because a prediction (preprocessing + MobileNetV3 on CPU) takes only tens of milliseconds.
* **One worker saturates at about 19 requests/s.** Beyond that, requests queue: at 100 users the median is 1.9 s and p95 is 8 s.
* **Three workers about double throughput** (19 → 42 requests/s). At 50 users the median falls from 1,100 ms to 150 ms (7× faster), and at 100 users p95 falls from 8.1 s to 2.0 s.
* **No request failed** in any run. Under overload the service slows down but does not drop requests.
* Scaling isn't a perfect 3× because the three workers and Locust share the same 12 cores. On separate cloud containers, the gain per container would be closer to linear.

Raw CSVs and Locust HTML reports are in [`locust/results/`](locust/results/).

## References

- Kashif, M., Haider, Z. M., Muneer, I., Kumar, D., Tahir, T., & Shafi, J. (2026). Optimizing Deep Neural Models for Early Dyslexia Detection Using Novel Bilingual Handwritten Dataset. *Engineering Reports*, 8, e70867. doi:10.1002/eng2.70867
- Multilingual AI-Based Dyslexia Detection dataset. Mendeley Data, v1. doi:10.17632/b37s8b94nv.1 (CC BY 4.0)
- Howard, A. et al. (2019). Searching for MobileNetV3. *ICCV*.
- Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171 (previous module).
