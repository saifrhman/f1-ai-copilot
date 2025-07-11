# Formula 1 AI Copilot

A comprehensive AI-powered system that supports Formula 1 race engineers, strategists, and performance analysts with real-time race insights, strategy optimization, and compliance monitoring.

## 🏁 Features

### 🧠 Core AI Modules
- **Strategy Optimizer**: Multi-factor race strategy optimization using telemetry, weather, tire degradation, and driver behavior
- **FIA RAG Agent**: Intelligent querying of FIA regulations using LangChain + OpenAI
- **Penalty Predictor**: ML-based penalty prediction using historical precedents and FIA rules
- **Natural Query Processor**: Natural language queries for performance and regulatory analysis
- **Driver Emotion Classifier**: Real-time emotion analysis from radio communications
- **Ghost Car Visualizer**: Lap comparison visualizations with telemetry overlays
- **Setup Recommender**: Bayesian optimization for optimal car setup recommendations

### 📊 Real-time Analysis
- Telemetry processing and analysis
- Tire degradation modeling
- Weather condition adaptation
- Competition intelligence
- Driver behavior profiling
- Rule violation detection

### 🎯 Strategic Insights
- Pit stop strategy optimization
- Tire compound selection
- Undercut/overcut opportunity identification
- Risk assessment and management
- Performance trend analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL (for metadata storage)
- Redis (optional, for caching)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/f1-ai-copilot.git
cd f1-ai-copilot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and database credentials
```

5. **Run the application**
```bash
python app/main.py
```

The API will be available at `http://localhost:8000`

## 📦 Project Structure

```
f1-ai-copilot/
├── app/                          # FastAPI application
│   ├── main.py                   # Main application entry point
│   └── api/
│       └── schemas.py            # Pydantic schemas
├── core_modules/                 # Core AI modules
│   ├── strategy_optimizer/       # Race strategy optimization
│   ├── rule_checker/            # FIA rules and penalty prediction
│   ├── llm_query/               # Natural language processing
│   ├── driver_emotion/          # Emotion classification
│   ├── ghost_car/               # Lap comparison visualization
│   ├── setup_optimizer/         # Car setup optimization
│   └── driver_modeling/         # Driver behavior modeling
├── data/                        # Data storage
│   ├── telemetry/               # Telemetry data
│   ├── video/                   # Video feeds
│   ├── audio/                   # Radio communications
│   └── fia_docs/               # FIA regulations
├── notebooks/                   # Jupyter notebooks for analysis
└── tests/                      # Test suite
```

## 🧠 Core Modules

### Strategy Optimizer
```python
from core_modules.strategy_optimizer.strategy_engine import generate_strategy

strategies = generate_strategy(
    telemetry=telemetry_data,
    car_status=car_status,
    driver_profile=driver_profile,
    tire_data=tire_data,
    race_state=race_state,
    competition=competition
)
```

### FIA RAG Agent
```python
from core_modules.rule_checker.fia_rag_agent import query_fia_regulations

answer = query_fia_regulations("What are the penalties for track limits violations?")
```

### Penalty Predictor
```python
from core_modules.rule_checker.penalty_predictor import predict_penalty

penalty = predict_penalty(
    incident_type="track_limits",
    track_condition="dry",
    intent="accidental"
)
```

### Natural Query Processor
```python
from core_modules.llm_query.natural_query import process_natural_query

result = process_natural_query("Why did we lose 0.7s in Sector 2 on Lap 14?")
```

## 📊 API Endpoints

### Strategy Generation
```bash
POST /api/strategy/generate
```
Generates optimal race strategies based on current conditions.

### FIA Regulations Query
```bash
POST /api/fia/query
```
Queries FIA regulations using RAG system.

### Penalty Prediction
```bash
POST /api/penalty/predict
```
Predicts penalties for incidents based on FIA rules and precedent.

### Natural Language Queries
```bash
POST /api/query/natural
```
Processes natural language queries about race performance or regulations.

### Emotion Classification
```bash
POST /api/emotion/classify
```
Classifies driver emotions from radio communications.

### Ghost Car Visualization
```bash
POST /api/ghost/generate
```
Generates ghost car visualizations for lap comparisons.

### Setup Recommendations
```bash
POST /api/setup/recommend
```
Recommends optimal car setup based on conditions.

## 🧪 Testing

### Run all tests
```bash
python -m pytest tests/
```

### Test individual modules
```bash
# Strategy engine
python core_modules/strategy_optimizer/test_strategy_engine.py

# FIA RAG agent
python core_modules/rule_checker/fia_rag_agent.py

# Setup recommender
python core_modules/setup_optimizer/setup_recommender.py
```

### Interactive testing
```bash
# Start Jupyter notebook
jupyter notebook notebooks/
```

## 🔧 Configuration

### Environment Variables
```bash
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Database
DATABASE_URL=postgresql://user:password@localhost/f1_copilot

# Redis (optional)
REDIS_URL=redis://localhost:6379

# Qdrant Vector Database
QDRANT_URL=http://localhost:6333
```

### API Configuration
The main application supports the following configuration options:

- **Host**: `0.0.0.0` (default)
- **Port**: `8000` (default)
- **CORS**: Enabled for all origins
- **Logging**: Structured logging with configurable levels

## 🏗️ Architecture

### Data Flow
1. **Telemetry Ingestion**: Real-time telemetry data from F1 cars
2. **Data Processing**: Analysis and feature extraction
3. **AI Analysis**: Multi-module AI processing
4. **Strategy Generation**: Optimal strategy recommendations
5. **Visualization**: Real-time dashboards and insights

### AI Pipeline
1. **Input Processing**: Validate and structure input data
2. **Feature Extraction**: Extract relevant features from telemetry
3. **Model Inference**: Run AI models for analysis
4. **Strategy Optimization**: Generate optimal strategies
5. **Output Generation**: Provide actionable insights

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -t f1-ai-copilot .

# Run container
docker run -p 8000:8000 f1-ai-copilot
```

### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/
```

### Cloud Deployment
The application is designed to run on Google Cloud Platform with:
- **Compute Engine**: For application hosting
- **Cloud SQL**: For PostgreSQL database
- **Cloud Storage**: For data storage
- **Cloud Run**: For serverless deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 core_modules/ app/ tests/

# Run type checking
mypy core_modules/ app/

# Run tests with coverage
pytest --cov=core_modules --cov=app tests/
```

## 📈 Performance

### Benchmarks
- **Strategy Generation**: < 2 seconds for 5 strategy options
- **FIA Query**: < 1 second for regulatory questions
- **Emotion Classification**: < 0.5 seconds per audio clip
- **Setup Optimization**: < 3 seconds for complete setup recommendation

### Scalability
- **Concurrent Users**: Supports 100+ simultaneous users
- **Data Processing**: Real-time processing of 1000+ telemetry points/second
- **Storage**: Efficient vector storage for FIA regulations
- **Caching**: Redis-based caching for frequently accessed data

## 🔒 Security

- **API Authentication**: JWT-based authentication
- **Data Encryption**: AES-256 encryption for sensitive data
- **Input Validation**: Comprehensive input validation using Pydantic
- **Rate Limiting**: Configurable rate limiting for API endpoints
- **Audit Logging**: Complete audit trail for all operations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FIA**: For regulatory framework and rules
- **Formula 1**: For the sport that inspires this project
- **OpenAI**: For GPT models powering natural language processing
- **FastAPI**: For the excellent web framework
- **LangChain**: For RAG implementation

## 📞 Support

For support and questions:
- **Email**: support@f1-ai-copilot.com
- **Discord**: [F1 AI Copilot Community](https://discord.gg/f1-ai-copilot)
- **Documentation**: [docs.f1-ai-copilot.com](https://docs.f1-ai-copilot.com)

---

**Built with ❤️ for the F1 community**
