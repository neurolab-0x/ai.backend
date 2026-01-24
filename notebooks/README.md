# NeuroLab Jupyter Notebooks

This directory contains comprehensive Jupyter notebooks that demonstrate all aspects of the NeuroLab EEG & Voice Analysis Platform.

## 📚 Notebook Overview

### 00_NeuroLab_Complete_Overview.ipynb
**Complete system architecture and feature overview**
- System architecture visualization
- Mental state classification system
- Feature engineering pipeline (930+ features)
- Technology stack overview

### 01_EEG_Data_Generation_and_Exploration.ipynb
**EEG data generation and exploratory analysis**
- Synthetic EEG data generation for different mental states
- Statistical analysis and visualization
- Feature engineering basics
- Data preprocessing techniques

### 02_Model_Training_and_Evaluation.ipynb
**Complete model training pipeline**
- Multiple model architectures (CNN-LSTM, ResNet-LSTM, Transformer)
- Advanced training techniques and callbacks
- Comprehensive evaluation metrics
- Model comparison and selection
- Confidence calibration

### 03_Voice_Processing_Analysis.ipynb
**Voice emotion detection and analysis**
- Audio preprocessing and feature extraction
- Emotion detection using TensorFlow models
- Mental state mapping from emotions
- Multimodal analysis integration

### 04_Feature_Engineering_Deep_Dive.ipynb
**Advanced feature extraction techniques**
- 930+ comprehensive EEG features
- Time-domain, frequency-domain, and nonlinear features
- Wavelet analysis and harmonic features
- Cross-channel connectivity analysis
- Feature selection and dimensionality reduction

## 🚀 Getting Started

### Prerequisites
```bash
# Install required packages
pip install -r ../requirements.txt

# Additional packages for notebooks
pip install jupyter ipykernel matplotlib seaborn
```

### Running the Notebooks

1. **Start Jupyter Lab/Notebook:**
```bash
# From the project root directory
jupyter lab notebooks/
# or
jupyter notebook notebooks/
```

2. **Run notebooks in order:**
   - Start with `00_NeuroLab_Complete_Overview.ipynb` for system overview
   - Follow with `01_EEG_Data_Generation_and_Exploration.ipynb` for data basics
   - Continue with other notebooks based on your interests

3. **Ensure the API is running (for API-related notebooks):**
```bash
# From project root
python main.py
```

## 📊 What You'll Learn

### EEG Analysis
- Mental state classification (Relaxed, Focused, Stressed)
- Advanced signal processing techniques
- Feature extraction from brain signals
- Model training and evaluation

### Voice Processing
- Emotion detection from audio
- Audio feature extraction
- Mental state mapping from emotions
- Multimodal data fusion

### Machine Learning
- Deep learning architectures for EEG
- Model comparison and selection
- Confidence calibration
- Performance optimization

### API Integration
- RESTful API usage
- Authentication and security
- Real-time processing
- Error handling and best practices

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors:**
   - Ensure you're running from the correct directory
   - Check that all dependencies are installed
   - Verify the virtual environment is activated

2. **Model Loading Issues:**
   - Run notebook 02 first to train and save models
   - Check that model files exist in the `../model/` directory

3. **API Connection Issues:**
   - Ensure the NeuroLab API server is running
   - Check the API URL in notebook configurations
   - Verify authentication settings

4. **Memory Issues:**
   - Reduce batch sizes in training notebooks
   - Use smaller datasets for testing
   - Close unused notebooks to free memory

### Dependencies

**Core Requirements:**
- Python 3.8+
- TensorFlow 2.15+
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn
- Jupyter Lab/Notebook

**Optional (for full functionality):**
- librosa (audio processing)
- antropy (nonlinear features)
- pywt (wavelet analysis)
- SHAP, LIME (interpretability)

## 📈 Performance Notes

- **Training notebooks** may take 10-30 minutes depending on hardware
- **Feature extraction** can be computationally intensive
- **GPU acceleration** is recommended for model training
- **Memory usage** can be high with large datasets

## 🤝 Contributing

To add new notebooks or improve existing ones:

1. Follow the existing naming convention
2. Include comprehensive documentation
3. Add error handling and user guidance
4. Test with different data scenarios
5. Update this README with new notebook descriptions

## 📝 License

These notebooks are part of the NeuroLab project and are licensed under the MIT License.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the main project documentation
3. Open an issue in the project repository

---

**Happy Learning with NeuroLab! 🧠🎵**