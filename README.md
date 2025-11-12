IE4483Project/
│
├── datasets/                  ← datasets folder is ignored in .gitignore, please add it yourself if you want to reproduce training
│   ├── train/
│   │   ├── cat/
│   │   └── dog/
│   ├── val/
│   │   ├── cat/
│   │   └── dog/
│   └── test/
│
├── traditional_ml/
│   ├── hog_svm.ipynb          ← feature extraction + SVM training
│   ├── utils_data.py          ← helper funcs (load, resize, HOG)
│   ├── cached_features/       ← saved .npz features (optional)
│
├── deeplearning/
│   ├── cnn_....ipynb                ← deeplearning models notebook
├── report/
│   ├── IE4483_Report.pdf      ← final report
│   ├── figures/               ← plots, misclassified examples
│
├── submission/
│   ├── submission.csv         ← final predictions
│
└── README.md                  ← how to run, dependencies

The requirements.txt file includes all necessary libraries for both:

traditional_ml/hog_svm.ipynb (SVM + HOG)

For the deeplearning, we used different environment as we run on individual devices.
For cnn_error_analysis, cnn_gpu_checks, cnn_smallest, cnn_improved, cnn_fullset:
tensorflow 2.15 [andCuda], training done with local GPU, python 3.10.12

For cat_dog_cnn:
tensorflow 2.20 python 3.10.3

For vgg16_train:
tensorflow 2.16.2 python 3.11.7

This project include multiple branches.
Main branch include our simple cnn
VGG branch include 3 notebook files. vgg16_train.ipynb is the vgg model before fine tuning. vgg16_train finetune.ipynb is the vgg model after fine tuning. 
cnn_train_vgg16_cifar10.ipynb is the modified model for training cifar10. 

For the trained weights (not included in github but should be found in zipfolder and googleDrive)
cnn_final_15k.keras for the final simple cnn model

to enable gpu in training:
1. use wsl and Ubuntu, i used 22.04 Lts
2. use vscode with wsl extension, open project folder with linux
3. setup venv accordingly in linux, i used tensorflow 2.15 andCuda, to be compatible with my gpu, rtx5070Ti.



