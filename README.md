🌱 AI-Powered Plant Disease Detection

📌 Project Overview
This project was developed as part of my **Edunet Foundation Internship (AICTE)** under the theme **Sustainable Agriculture**.
It aims to build an **AI-powered system that detects plant diseases from leaf images** using deep learning and computer vision.
The system helps farmers identify diseases early, reduce crop loss, minimize pesticide use, and promote **eco-friendly farming** practices.

🎯 Objectives
- Apply Artificial Intelligence (AI) in agriculture for disease identification
- Train a deep learning model using plant leaf images
- Provide a user-friendly tool to upload an image and receive predictions
- Promote sustainability by reducing waste, cost, and chemical overuse

📂 Dataset
- Source: [Kaggle – New Plant Diseases Dataset]https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- Size: ~3 GB (compressed to a mini dataset <25 MB for GitHub)
- Classes: Multiple crop species (healthy + diseased leaves)
- Preprocessing: Resizing, normalization, augmentation for robustness

⚙️ Tech Stack
- Programming Language: Python
- Frameworks & Libraries: TensorFlow, Keras, NumPy, OpenCV, scikit-learn, Matplotlib
- Platform: Google Colab for training & experimentation
- Deployment (optional): Streamlit or Flask for web app demo

🛠️ Workflow
# Week 1 – Data Preprocessing
- Downloaded and organized Kaggle dataset into structured class folders
- Cleaned and removed corrupted images
- Resized and normalized images to reduce memory usage
- Sampled and compressed dataset (3 GB → <25 MB) for easier sharing and GitHub upload

# Week 2 – Model Development
- Selected deep learning architecture (CNN with Transfer Learning: MobileNetV2 / ResNet50)
- Created training and validation splits
- Tuned hyperparameters (learning rate, batch size, epochs)
- Trained the model on the mini dataset and validated accuracy
- Saved the trained model (.h5) for deployment

# Week 3 – Evaluation & Deployment
- Evaluated performance using accuracy, precision, recall, F1-score
- Generated confusion matrix and loss/accuracy plots
- Built a simple Streamlit web interface for testing predictions
- Packaged the final model and mini dataset for GitHub sharing

📊 Results
- Training Accuracy: ~85–90% (depending on model architecture)
- Validation Accuracy: ~80–88%
- Stable loss curves showing minimal overfitting

🚀 How to Run
1. Clone the repository
   git clone https://github.com/your-username/plant-disease-detection.git
   cd plant-disease-detection
2. Install dependencies
   pip install -r requirements.txt
3. Open and run the Jupyter Notebook in Google Colab or locally
   - Upload the mini dataset or link to the Kaggle dataset
   - Train the model or load the pre-trained model
4. (Optional) Run the web app
   streamlit run app.py

🌍 Sustainable Agriculture Impact
- Early detection reduces major crop losses
- Minimizes pesticide use, supporting eco-friendly farming
- Improves farmers’ yield and income
- Encourages AI-driven innovation for long-term agricultural sustainability

📌 Future Enhancements
- Develop a mobile app for offline use by farmers
- Add real-time camera detection for field deployment
- Expand dataset with more crops and disease classes

👨‍💻 Author
- Rubesh S
- Internship: "Edunet Foundation Internship (AICTE)"
- Theme: Sustainable Agriculture
