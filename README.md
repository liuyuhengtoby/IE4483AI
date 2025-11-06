IE4483Project/
│
├── datasets/
│   ├── train/
│   │   ├── cat/
│   │   └── dog/
│   ├── val/
│   │   ├── cat/
│   │   └── dog/
│   └── test/
│
├── traditional_ml/
│   ├── hog_svm.ipynb          ← feature extraction + SVM training + submission
│   ├── utils_data.py          ← helper funcs (load, resize, HOG)
│   ├── cached_features/       ← saved .npz features (optional)
│
├── deep_learning/
│   ├── cnn_train.ipynb        ← CNN model training, plots
│   ├── cnn_predict.ipynb      ← load model, predict test, make submission
│   ├── models/                ← saved .h5 / .pth weights
│   ├── logs/                  ← training history, accuracy plots
│
├── report/
│   ├── IE4483_Report.pdf      ← final report
│   ├── figures/               ← plots, misclassified examples
│
├── submission/
│   ├── submission.csv         ← final predictions
│
└── README.md                  ← how to run, dependencies

1️⃣ Create and activate a virtual environment

python -m venv .venv
.venv\Scripts\activate


2️⃣ Upgrade pip (recommended)

python -m pip install --upgrade pip


3️⃣ Install all required packages

pip install -r requirements.txt

🍎 For macOS Users (especially M1/M2 Apple Silicon)

1️⃣ Create and activate a virtual environment

python3 -m venv .venv
source .venv/bin/activate


2️⃣ Upgrade pip

pip install --upgrade pip


3️⃣ Install packages (Mac-specific TensorFlow build)
go to requirements.txt and edit the tensorflow package name accordingly


(On Intel Macs, you can also just use pip install -r requirements.txt.)

🧪 Verify Installation

After installation, run this quick test to ensure all libraries work correctly:

macos:

python - << 'PY'
import numpy, pandas, PIL, skimage, sklearn, tqdm, joblib, tensorflow
print("✅ Environment OK — all libraries loaded successfully.")
PY


windows:

python -c "import numpy, pandas, PIL, skimage, sklearn, tqdm, joblib, tensorflow; print('✅ Environment OK — all libraries loaded successfully')"

If you see ✅ Environment OK, you’re ready to start running the notebooks.

Always activate your environment before running notebooks or scripts.

The requirements.txt file includes all necessary libraries for both:

traditional_ml/hog_svm.ipynb (SVM + HOG)
deep_learning/cnn_train.ipynb and cnn_predict.ipynb (CNN)