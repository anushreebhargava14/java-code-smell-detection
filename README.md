# SmellScope AI

**Machine Learning-Based Java Code Smell Detection**

SmellScope AI is a machine-learning-powered platform that detects code smells in Java source code, explains why code was flagged, and provides actionable refactoring suggestions.

Unlike traditional rule-based detectors, SmellScope AI uses models trained on human-reviewed code from the MLCQ dataset to identify patterns associated with poor code quality.

It detects four types of code smells:

- **Blob (God Class):** Classes that accumulate excessive responsibilities.
- **Data Class:** Classes that primarily hold data with little behaviour.
- **Feature Envy:** Methods that rely more heavily on another class's data than their own.
- **Long Method:** Methods that have grown excessively large and complex.

---

## Key Features

### 1. Java Code Analysis
- Analyse individual Java files, multiple files, or ZIP archives.
- Extract class-level and method-level metrics using the CK metrics tool.
- Detect four types of code smells using trained machine learning models.

### 2. Interactive Analysis Workspace
- IDE-style code viewer with flagged lines highlighted.
- File explorer for navigating uploaded projects.
- Hotspot treemap to identify files with potential code quality issues.
- Detailed results for individual classes and methods.

### 3. Explainable Predictions
- Understand why a particular class or method was flagged.
- Examine the contribution of important code metrics.
- Compare model scores when individual metrics are changed.
- Receive refactoring suggestions such as Extract Method, Extract Class and Move Method.

### 4. Re-scan and Compare
- Modify source code and analyse it again.
- Compare previous and current results.
- Identify resolved and newly introduced code smells.

### 5. Model Lab
- Experiment with metric sliders and observe changes in model scores.
- Explore actual held-out test cases.
- Examine precision-recall curves, confusion matrices and feature importance.
- Compare candidate machine learning models.

### 6. Scan History and Export
- Save and reopen previous scans.
- Export findings as CSV or JSON.
- Generate printable analysis reports.

---

## Model Performance

The models were evaluated on 96 repositories that were completely excluded from training.

The dataset was split by repository to reduce data leakage and provide a more realistic evaluation.

| Code Smell | Selected Model | Average Precision | F1 Score |
|---|---|---:|---:|
| Long Method | Random Forest | 0.86 | 0.74 |
| Blob | Random Forest | 0.64 | 0.58 |
| Data Class | Logistic Regression | 0.60 | 0.59 |
| Feature Envy | Logistic Regression | 0.23 | 0.17 |

**Important:** Feature Envy is the weakest-performing model because the available method-level metrics do not directly capture how much a method accesses another object's data. Its predictions should therefore be interpreted with particular caution.

The model outputs are ranking scores, not calibrated probabilities. Predictions are intended to assist human code review rather than replace it.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python, JavaScript, Java |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Code Metrics | CK (Java Code Metrics Calculator) |
| Model Storage | Joblib |
| Visualisation | Matplotlib |
| Dataset | MLCQ |

---

## How It Works

The project follows an end-to-end machine learning pipeline:

1. **Dataset Collection:** Retrieve the MLCQ dataset containing 14,739 human reviews of Java code.
2. **Source Retrieval:** Download the corresponding Java files at their original reviewed commits.
3. **Metric Extraction:** Use CK to calculate class-level and method-level software metrics.
4. **Code Mapping:** Match human-reviewed code locations with CK-generated metrics.
5. **Data Cleaning:** Perform mapping validation, quality checks and evidence-based repairs.
6. **Dataset Preparation:** Merge reviews, generate labels and create repository-aware training and testing splits.
7. **Model Training:** Compare Logistic Regression, Random Forest and Gradient Boosting against baseline approaches.
8. **Evaluation:** Evaluate models using average precision, precision, recall, F1 score and MCC.
9. **Deployment:** Integrate the selected models into a FastAPI application with an interactive frontend.

---

## Project Structure

```text
java-code-smell-detection/
│
├── api/
│   ├── main.py
│   ├── engine.py
│   ├── insights.py
│   └── workspace.py
│
├── frontend/
│   ├── css/
│   ├── fonts/
│   ├── js/
│   └── index.html
│
├── src/
│   ├── data/
│   ├── features/
│   ├── inference/
│   └── models/
│
├── models/
│   ├── blob.joblib
│   ├── data_class.joblib
│   ├── feature_envy.joblib
│   └── long_method.joblib
│
├── reports/
│   ├── figures/
│   └── evaluation results
│
├── data/
│   ├── raw/
│   └── processed/
│
├── tools/
│
├── requirements.txt
├── check_environment.py
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10 or newer
- Java 11 or newer
- Apache Maven
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/anushreebhargava14/java-code-smell-detection.git

cd java-code-smell-detection
```

### 2. Create a Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up the CK Metrics Tool

Clone the CK repository:

```bash
git clone https://github.com/mauricioaniche/ck.git tools/ck
```

Build it using Maven:

```bash
cd tools/ck
mvn clean package -DskipTests
cd ../..
```

The expected JAR file is:

```text
tools/ck/target/ck-0.7.1-SNAPSHOT-jar-with-dependencies.jar
```

### 5. Check the Environment

```bash
python check_environment.py
```

### 6. Run the Application

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open the application in your browser:

**http://127.0.0.1:8000/**

Interactive API documentation is also available at:

**http://127.0.0.1:8000/docs**

The application requires the trained models and supporting processed pipeline artifacts. To reproduce the complete dataset and generate missing artifacts, follow the pipeline instructions below.

---

## Reproducing the Machine Learning Pipeline

The pipeline can be reproduced using the scripts in `src/`.

| Stage | Script | Purpose |
|---|---|---|
| 1 | `download_mlcq.py` | Download the MLCQ dataset |
| 2 | `fetch_source_files.py` | Retrieve reviewed Java source files |
| 3 | `join_and_extract_snippets.py` | Match reviews with source files |
| 4 | `run_ck_batched.py` | Extract CK metrics in batches |
| 5 | `combine_ck_outputs.py` | Merge CK outputs |
| 6 | `map_ck_to_mlcq_full.py` | Map reviews to classes and methods |
| 7 | `repair_mapping.py` | Apply verified mapping repairs |
| 8 | `build_datasets.py` | Build labelled datasets and repository splits |
| 9 | `train_baselines.py` | Train and compare candidate models |
| 10 | `final_evaluation.py` | Generate evaluation results and visualisations |
| 11 | `parity_check.py` | Validate inference and training-time score consistency |

For example:

```bash
python src/data/download_mlcq.py
python src/data/fetch_source_files.py
python src/data/join_and_extract_snippets.py
python src/features/run_ck_batched.py
python src/features/combine_ck_outputs.py
python src/features/map_ck_to_mlcq_full.py
python src/features/repair_mapping.py
python src/features/build_datasets.py
python src/models/train_baselines.py
python src/models/final_evaluation.py
python src/inference/parity_check.py
```

Some stages involve substantial downloads and processing time. The CK extraction stage supports batch-based processing.

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Check Java, CK and model availability |
| `POST /analyze` | Analyse Java source code |
| `POST /analyze-file` | Analyse an uploaded Java file |
| `POST /api/scan` | Scan a project |
| `POST /api/scan-upload` | Scan uploaded files or ZIP archives |
| `GET /api/runs` | Retrieve saved scans |
| `GET /api/cases/{smell}` | Retrieve held-out test cases |
| `GET /api/examples` | Browse model examples |
| `GET /api/lab` | Retrieve model lab information |
| `POST /api/lab/score` | Generate model scores |
| `GET /api/results` | Retrieve evaluation results |

Additional endpoints support scan history, model comparisons, dataset information and detailed analysis.

---

## Validation and Limitations

Several checks were incorporated into the development process:

- Repository-aware training and testing splits to reduce data leakage.
- Grouped cross-validation for model selection.
- Validation of dataset mapping and repaired records.
- Checks for duplicated line endings in downloaded source files.
- Training-to-inference score consistency checks.
- Evaluation using multiple performance metrics.

### Current Limitations

- Supports Java source code only.
- Feature Envy detection has comparatively low performance.
- Model outputs are not calibrated probabilities.
- Human-reviewed labels can contain disagreements.
- Predictions should be treated as suggestions for further code review, not definitive judgements.

---

## Dataset and References

This project uses the **MLCQ (Industry-Relevant Code Smell Dataset)**.

**Research paper:**

Madeyski, L. and Lewowski, T. (2020). *MLCQ: Industry-relevant code smell data set.* Proceedings of the 24th International Conference on Evaluation and Assessment in Software Engineering (EASE).

- Paper: https://doi.org/10.1145/3383219.3383264
- Dataset: https://doi.org/10.5281/zenodo.3666840

The dataset is licensed under CC BY 4.0.

**Code Metrics Tool:**

Aniche, M. — CK: Java Code Metrics Calculator  
https://github.com/mauricioaniche/ck



---

*SmellScope AI — Making Java code quality easier to understand, analyse and improve.*
