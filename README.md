# Diabetes Prediction using Quantum Machine Learning

A full-stack web application that leverages Quantum Machine Learning (QML) to predict diabetes risk with high accuracy. This project demonstrates the practical application of quantum computing principles in healthcare diagnostics.

##  Key Features

- **Quantum-Powered Predictions**: Utilizes Quantum Machine Learning algorithms for enhanced pattern recognition
- **High Performance**: Achieves 71.4% accuracy with 83.3% sensitivity in diabetes detection
- **Full-Stack Architecture**: Complete web application with React frontend and Python backend
- **Real-Time Analysis**: Instant diabetes risk assessment based on patient data
- **User-Friendly Interface**: Intuitive design for seamless user experience

##  Model Performance

- **Accuracy**: 71.4%
- **Sensitivity**: 83.3%
- **Dataset**: PIMA Indian Diabetes Dataset
- **Technology**: Quantum Machine Learning (QML)

##  Tech Stack

### Frontend
- React.js
- JavaScript
- Tailwind CSS
- PostCSS
- HTML5 & CSS3

### Backend
- Python
- Quantum Machine Learning libraries (PennyLane)
- Flask (API server)
- NumPy & Scikit-learn
- Pickle (model serialization)

### Machine Learning
- PennyLane (Quantum Computing framework)
- Scikit-learn (preprocessing and metrics)
- Data preprocessing and feature engineering
- Quantum circuit-based classification
- Model persistence with Pickle

##  Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:
- Node.js (v14 or higher)
- Python (v3.8 or higher)
- npm or yarn
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Disha23112004/diabetes-qml-prediction.git
   cd diabetes-qml-prediction
   ```

2. **Set up the Backend**
   ```bash
   cd backend
   # Install the necessary Python packages for your QML implementation
   pip install pennylane numpy scikit-learn flask
   ```

3. **Set up the Frontend**
   ```bash
   cd ../diabetes-frontend
   npm install
   ```

### Running the Application

1. **Start the Backend Server**
   ```bash
   cd backend
   python qml_api_server.py
   ```

2. **Start the Frontend Development Server**
   ```bash
   cd diabetes-frontend
   npm start
   ```

3. Open your browser and navigate to `http://localhost:3000`

## Project Structure

```
qml/
├── __pycache__/            # Python cache files
├── .history/               # Version history
│
├── backend/                # Python backend with QML models
│   ├── data/              # Dataset files
│   ├── models/            # QML model files
│   │   ├── decision_threshold...
│   │   ├── qml_diabetes_wei...
│   │   ├── qml_diabetes_wei...
│   │   └── scaler.pkl
│   ├── outputs/           # Model outputs and results
│   │   └── qml_improved_res...
│   ├── qml_api_server.py  # Flask API server
│   └── train_qml_model.py # Model training script
│
├── diabetes-frontend/      # React frontend
│   ├── node_modules/      # Node dependencies
│   ├── public/            # Static files
│   ├── src/               # React source code
│   ├── .gitignore         # Git ignore file
│   ├── package-lock.json  # Dependency lock file
│   ├── package.json       # Node dependencies
│   ├── postcss.config.js  # PostCSS configuration
│   ├── README.md          # Frontend documentation
│   └── tailwind.config.js # Tailwind CSS configuration
│
└── .gitignore
```

##  How It Works

This application uses Quantum Machine Learning to analyze patient data and predict diabetes risk. The QML approach offers advantages over classical machine learning:

1. **Data Input**: Users enter health parameters (glucose levels, BMI, age, etc.)
2. **Quantum Processing**: Data is encoded into quantum states for processing
3. **QML Classification**: Quantum circuits perform classification
4. **Risk Assessment**: Results are decoded and presented to the user

### Input Features

The model analyzes the following health parameters:
- Glucose concentration
- Blood pressure
- Skin thickness
- Insulin levels
- Body Mass Index (BMI)
- Diabetes pedigree function
- Age

##  Use Cases

- **Medical Screening**: Early diabetes risk detection
- **Healthcare Research**: Exploring QML applications in medicine
- **Educational Tool**: Learning quantum computing in healthcare
- **Clinical Support**: Assisting healthcare professionals in diagnosis

##  Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Disha Parasu**
- GitHub: [@Disha23112004](https://github.com/Disha23112004)

##  Acknowledgments

- PIMA Indian Diabetes Dataset
- Quantum Computing community
- Healthcare AI research contributors

##  References

This project builds upon research in quantum machine learning for healthcare applications, particularly diabetes classification using quantum algorithms.

##  Future Enhancements

- [ ] Add more QML algorithms for comparison
- [ ] Implement model explainability features
- [ ] Deploy to cloud platform
- [ ] Add mobile responsive design
- [ ] Integrate with healthcare systems
- [ ] Expand dataset for improved accuracy

##  Contact

For questions or suggestions, please open an issue in the GitHub repository.

---

**Note**: This is a research and educational project. Always consult healthcare professionals for medical diagnosis and treatment.
