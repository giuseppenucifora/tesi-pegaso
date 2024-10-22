import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Simulazione dei dati
def simulate_data(weather_data):
    np.random.seed(42)
    
    # Assumiamo che weather_data abbia colonne come 'temperature', 'rainfall', 'humidity'
    n_samples = len(weather_data)
    
    # Simulazione produzione olive (kg/ettaro)
    base_production = 5000  # produzione media di base
    weather_effect = 0.1 * weather_data['temperature'] + 0.05 * weather_data['rainfall'] - 0.02 * weather_data['humidity']
    olive_production = base_production + weather_effect + np.random.normal(0, 500, n_samples)
    
    # Simulazione qualità olio (acidità %)
    base_acidity = 0.5
    acidity = base_acidity - 0.01 * weather_data['temperature'] + 0.005 * weather_data['humidity'] + np.random.normal(0, 0.1, n_samples)
    
    # Simulazione prezzi di mercato (€/litro)
    base_price = 10
    price = base_price + 0.1 * olive_production/1000 - 1 * acidity + np.random.normal(0, 1, n_samples)
    
    return pd.DataFrame({
        'olive_production': olive_production,
        'oil_acidity': acidity,
        'market_price': price
    })

# Caricamento dati meteo (da implementare con i dati reali)
weather_data = pd.DataFrame({
    'temperature': np.random.normal(20, 5, 1000),
    'rainfall': np.random.normal(50, 20, 1000),
    'humidity': np.random.normal(60, 10, 1000)
})

# Simulazione dei dati di produzione
production_data = simulate_data(weather_data)
full_data = pd.concat([weather_data, production_data], axis=1)

# Preparazione del modello di machine learning
X = full_data[['temperature', 'rainfall', 'humidity']]
y = full_data['olive_production']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Creazione dell'app Dash
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard Produzione Olio d'Oliva"),
    
    dcc.Graph(id='production-weather-scatter'),
    
    dcc.Dropdown(
        id='weather-feature-dropdown',
        options=[
            {'label': 'Temperatura', 'value': 'temperature'},
            {'label': 'Precipitazioni', 'value': 'rainfall'},
            {'label': 'Umidità', 'value': 'humidity'}
        ],
        value='temperature',
        style={'width': '50%'}
    ),
    
    html.Div(id='prediction-output')
])

@app.callback(
    Output('production-weather-scatter', 'figure'),
    Input('weather-feature-dropdown', 'value')
)
def update_graph(weather_feature):
    fig = px.scatter(full_data, x=weather_feature, y='olive_production', 
                     trendline="ols", title=f"Produzione Olive vs {weather_feature}")
    return fig

@app.callback(
    Output('prediction-output', 'children'),
    [Input('weather-feature-dropdown', 'value')]
)
def update_prediction(weather_feature):
    # Esempio di previsione usando valori medi
    sample_weather = full_data[['temperature', 'rainfall', 'humidity']].mean().to_frame().T
    prediction = model.predict(sample_weather)[0]
    return f"Produzione prevista con condizioni medie: {prediction:.2f} kg/ettaro"

if __name__ == '__main__':
    app.run_server(debug=True)
