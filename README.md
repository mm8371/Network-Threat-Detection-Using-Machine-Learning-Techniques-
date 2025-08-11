Overview
This is a real-time detection and classification system for cyberattacks that utilizes multiple machine learning algorithms for network traffic analysis to identify potential threats. It is built using a modular microservices architecture with visual analytics added for enhanced security monitoring.
Features
Multiple ML Models:

HistGradientBoosting (selected for production — 94% accuracy)

Random Forest (99% accuracy, but overfitting seen)

AdaBoost

Bernoulli Naive Bayes

Datasets:

CICIDS2017

UNSW-NB15

Preprocessing & Balancing:

Data cleaning, feature selection, and normalization

Random oversampling for handling class imbalance

Real-Time Monitoring:

Intake of live or emulated network logs

Threat classification with notification

Microservices Architecture:

Log intake

Threat classification

Alerting service

User authentication

Threat intelligence feed updates

Model Export: Saved pre-trained model as CYBER.pkl for simple deployment
 Project Structure

└── network_threat_detection_ml.py   # Main ML training script
├── CYBER.pkl                         # Trained model (optional, not used in repo)
└── README.md                         # Project documentation
⚙️ Installation
git clone https://github.com/mm8371/Network-Threat-Detection-Using-Machine-Learning-Techniques-.git
cd Network-Threat-Detection-Using-Machine-Learning-Techniques-
pip install -r requirements.txt
▶Usage

python network_threat_detection_ml.py
Adjust the script to reference your dataset (cyber1.csv or processed CICIDS/UNSW data).
 Results
Model	Accuracy	Notes
Random Forest	~99%	Overfitting noted
HistGradientBoosting	~94%	Best trade-off, selected
AdaBoost	Lower
BernoulliNB	Low
Future Work
Add to real-time SIEM

Optimize for IoT devices (low resource footprint)

Include explainable AI (SHAP, feature importance)

Increase dataset diversity with live network captures

License
This project is licensed under the MIT License.

