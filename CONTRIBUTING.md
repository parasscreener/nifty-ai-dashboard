# Contributing to Nifty AI Dashboard

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/nifty-ai-dashboard.git
   cd nifty-ai-dashboard
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Code Standards

### Python Style Guide
- Follow PEP 8 style guide
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Maximum line length: 88 characters (Black formatter)

### AI Model Guidelines
- Document model architecture and training process
- Include model performance metrics
- Add proper validation and testing
- Use consistent feature engineering approaches

### Testing Requirements
- Write unit tests for all new functions
- Include integration tests for AI models
- Test data pipeline with edge cases
- Maintain minimum 80% code coverage

## Contribution Process

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**
   - Write code following style guidelines
   - Add appropriate tests
   - Update documentation

3. **Test locally**
   ```bash
   pytest tests/
   streamlit run app.py
   ```

4. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Areas for Contribution

### High Priority
- [ ] Additional AI models (LSTM, Transformer)
- [ ] Real-time data streaming
- [ ] Advanced pattern recognition
- [ ] Backtesting framework
- [ ] Mobile responsive design

### Medium Priority
- [ ] Additional technical indicators
- [ ] News sentiment analysis
- [ ] Portfolio optimization
- [ ] Risk management tools
- [ ] User authentication

### Low Priority
- [ ] Additional chart types
- [ ] Export functionality
- [ ] Email alerts
- [ ] Multi-timeframe analysis
- [ ] Custom indicator builder

## AI Model Contributions

### Model Development Guidelines
1. **Data Requirements**
   - Use consistent data preprocessing
   - Handle missing data appropriately
   - Implement proper train/validation/test splits

2. **Model Architecture**
   - Document model complexity and requirements
   - Include model interpretability features
   - Provide performance benchmarks

3. **Integration**
   - Follow existing model interface patterns
   - Add proper error handling
   - Include model monitoring capabilities

### Supported Model Types
- **Time Series**: LSTM, GRU, Transformer
- **Classification**: Random Forest, SVM, Neural Networks
- **Ensemble**: Voting, Stacking, Boosting
- **Deep Learning**: CNN, RNN, Attention models

## Bug Reports

When reporting bugs, please include:
- Python version and OS
- Full error traceback
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable

## Feature Requests

For new features, please provide:
- Clear use case description
- Expected behavior
- Alternative solutions considered
- Willingness to implement

## Code Review Process

All contributions require code review:
1. **Automated checks** must pass (linting, tests)
2. **Manual review** by maintainer
3. **Testing** in development environment
4. **Documentation** review
5. **Final approval** and merge

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for major contributions
- Special recognition for AI model improvements

## Questions?

- 💬 **GitHub Discussions** for general questions
- 🐛 **GitHub Issues** for bugs and features
- 📧 **Email** for private inquiries

Thank you for contributing to making financial AI more accessible!
