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
