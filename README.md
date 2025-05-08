# DataStorm

# Agent Performance Analytics Dashboard

## Overview

The Agent Performance Analytics Dashboard is a Flask-based web application designed to visualize and analyze agent performance data, risk predictions, and personalized recommendations. This tool helps managers identify at-risk agents, understand key performance factors, and implement targeted interventions to improve agent retention and performance.

![image](https://github.com/user-attachments/assets/40fce0ee-528d-41b9-b222-e511d6bb4690)

## Features

- **Performance Overview**: Visualize agent risk distribution and performance classifications
- **Agent Management**: View and filter detailed agent information
- **Risk Prediction**: Predict NILL (Not In Labor List) risk for agents based on key metrics
- **Key Performance Factors**: Analyze the most important factors affecting agent performance
- **Personalized Recommendations**: Generate tailored action plans for each agent based on their risk profile

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/agent-dashboard.git](https://github.com/Gangadari11/DataStorm.git
   cd agent-dashboard
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```
3. Run the application:

  ``` bash
    python app.py
  ```

4. Open your browser and navigate to:
  ```
   http://127.0.0.1:5000/
  ```

## Usage Guide

### Dashboard Overview

The dashboard is divided into five main sections:

1. **Overview**: Displays summary statistics and key visualizations of agent risk and performance distribution.
2. **Agents**: Lists all agents with their key metrics and risk levels.
3. **Predictions**: Allows you to predict NILL risk for agents based on input parameters.
4. **Key Factors**: Visualizes the most important factors affecting agent performance.
5. **Recommendations**: Provides personalized action plans for individual agents.

### Making Predictions

To predict the NILL risk for an agent:

1. Navigate to the "Predictions" tab
2. Fill in the agent metrics in the prediction form
3. Click "Predict"
4. View the prediction results, including risk level and recommended actions

### Viewing Agent Details

To view detailed information for a specific agent:

1. Navigate to the "Agents" tab
2. Click the "View" button next to the agent of interest
3. A modal will appear with detailed agent information, key metrics, and personalized recommendations

### Generating Recommendations

To generate personalized recommendations:

1. Navigate to the "Recommendations" tab
2. Select an agent from the dropdown menu
3. View the agent's profile, risk level, and personalized action plan

## File Structure

```
agent-dashboard/
├── app.py                  # Main Flask application
├── prepare_features.py     # Feature preparation utilities
├── static/
│   ├── css/
│   │   └── style.css       # Dashboard styling
│   └── js/
│       └── dashboard.js    # Dashboard JavaScript functionality
├── templates/
│   └── index.html          # Main dashboard HTML template
├── model/                  # Directory for model artifacts (created on first run)
├── data/                   # Directory for data files (created on first run)
└── requirements.txt        # Python dependencies
```

## Dependencies

- **Flask**: Web framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **scikit-learn**: Machine learning utilities
- **Bootstrap**: Frontend styling
- **ECharts**: Interactive visualizations

## Configuration

The application creates sample data and model files on first run if they don't exist. To use your own data:

1. Place your agent data CSV in the `data/` directory as `agents_data.csv`
2. Place your trained model and related artifacts in the `model/` directory:
   - `xgboost_model.pkl`: Trained XGBoost model
   - `scaler.pkl`: Feature scaler
   - `feature_selector.pkl`: Feature selector
   - `feature_names.json`: List of feature names
   - `feature_importances.pkl`: Feature importance values

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   - Ensure all required packages are installed: `pip install -r requirements.txt`

2. **Port Already in Use**:
   - Change the port in `app.py` if port 5000 is already in use:
     ```python
     app.run(debug=True, port=5001)
    ```

3. **Model Prediction Errors**:
   - The application uses a simplified prediction approach if model files are missing or incompatible
   - Check that your model artifacts match the expected format and feature names

4. **Visualization Issues**:
   - Ensure JavaScript is enabled in your browser
   - Try clearing your browser cache if charts don't render properly

## Future Enhancements

- User authentication and role-based access control
- Real-time data integration
- Advanced filtering and search capabilities
- Export functionality for reports and recommendations
- Comparative analysis between agents
- Historical trend analysis
- Mobile-responsive design improvements
- Dark mode toggle
- Integration with notification systems

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

![image](https://github.com/user-attachments/assets/547d570a-577f-4f42-b90d-f5648e20de14)
![image](https://github.com/user-attachments/assets/236925a6-fe01-442b-8a49-4c2ee0e02287)
![image](https://github.com/user-attachments/assets/bff452ee-1444-41a9-8612-2309289301c6)
![image](https://github.com/user-attachments/assets/ef86f4f8-807a-449f-bdfb-88d86c25de91)
![image](https://github.com/user-attachments/assets/ed1c4481-5474-4990-8965-c81b613feae9)




