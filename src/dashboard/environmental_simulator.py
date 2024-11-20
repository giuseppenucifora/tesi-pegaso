import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class EnvironmentalSimulator:
    def __init__(self):
        # Parametri base per la crescita delle olive
        self.optimal_temp_range = (15, 25)  # °C
        self.optimal_humidity = 60  # %
        self.optimal_rainfall = 50  # mm/mese
        self.optimal_radiation = 250  # W/m²

        # Fasi fenologiche dell'olivo
        self.growth_phases = {
            'Dormienza': {'duration': 60, 'sensitivity': 0.3},
            'Germogliamento': {'duration': 30, 'sensitivity': 0.7},
            'Fioritura': {'duration': 30, 'sensitivity': 1.0},
            'Allegagione': {'duration': 45, 'sensitivity': 0.8},
            'Sviluppo Frutti': {'duration': 90, 'sensitivity': 0.6},
            'Maturazione': {'duration': 60, 'sensitivity': 0.5}
        }

    def calculate_stress_index(self, temp, humidity, rainfall, radiation):
        """Calcola l'indice di stress ambientale"""
        # Stress temperatura
        temp_avg = np.mean(temp)
        temp_stress = abs(temp_avg - np.mean(self.optimal_temp_range)) / 10

        # Stress idrico
        humidity_stress = abs(humidity - self.optimal_humidity) / 100
        rainfall_stress = abs(rainfall - self.optimal_rainfall) / self.optimal_rainfall

        # Stress radiazione
        radiation_stress = abs(radiation - self.optimal_radiation) / self.optimal_radiation

        # Indice di stress composito
        stress_index = (temp_stress * 0.4 +
                        humidity_stress * 0.2 +
                        rainfall_stress * 0.2 +
                        radiation_stress * 0.2)

        return min(1.0, stress_index)

    def simulate_growth(self, temp_range, humidity, rainfall, radiation, days=365):
        """Simula la crescita dell'olivo nel tempo"""
        results = []
        current_date = datetime.now()

        for day in range(days):
            # Calcola la fase corrente
            day_of_year = (current_date + timedelta(days=day)).timetuple().tm_yday
            phase = self.get_growth_phase(day_of_year)

            # Simula temperatura giornaliera
            temp = np.random.uniform(temp_range[0], temp_range[1])

            # Calcola stress giornaliero
            stress = self.calculate_stress_index(temp_range, humidity, rainfall, radiation)

            # Calcola crescita giornaliera (0-100%)
            growth_rate = self.calculate_growth_rate(phase, stress)

            results.append({
                'date': current_date + timedelta(days=day),
                'phase': phase,
                'temperature': temp,
                'stress_index': stress,
                'growth_rate': growth_rate
            })

        return pd.DataFrame(results)

    def get_growth_phase(self, day_of_year):
        """Determina la fase di crescita in base al giorno dell'anno"""
        total_days = 0
        for phase, details in self.growth_phases.items():
            total_days += details['duration']
            if day_of_year % 365 <= total_days:
                return phase
        return list(self.growth_phases.keys())[0]

    def calculate_growth_rate(self, phase, stress):
        """Calcola il tasso di crescita giornaliero"""
        base_rate = self.growth_phases[phase]['sensitivity']
        return base_rate * (1 - stress) * 100

    def calculate_production_impact(self, stress_history):
        """Calcola l'impatto sulla produzione"""
        base_production = 100  # kg/albero
        stress_impact = np.mean(stress_history)
        return base_production * (1 - stress_impact)


def create_growth_simulation_figure(sim_data: pd.DataFrame) -> go.Figure:
    """Crea il grafico della simulazione di crescita"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Aggiunge la linea di crescita
    fig.add_trace(
        go.Scatter(
            x=sim_data['date'],
            y=sim_data['growth_rate'],
            name="Tasso di Crescita",
            line=dict(color='#2E86C1', width=2)
        ),
        secondary_y=False
    )

    # Aggiunge l'indice di stress
    fig.add_trace(
        go.Scatter(
            x=sim_data['date'],
            y=sim_data['stress_index'],
            name="Indice di Stress",
            line=dict(color='#E74C3C', width=2)
        ),
        secondary_y=True
    )

    # Aggiungi indicatori delle fasi
    for phase in sim_data['phase'].unique():
        phase_data = sim_data[sim_data['phase'] == phase]
        fig.add_trace(
            go.Scatter(
                x=[phase_data['date'].iloc[0]],
                y=[0],
                name=phase,
                mode='markers+text',
                text=[phase],
                textposition='top center',
                marker=dict(size=10)
            ),
            secondary_y=False
        )

    # Configurazione layout
    fig.update_layout(
        title='Simulazione Crescita e Stress Ambientale',
        xaxis_title='Data',
        yaxis_title='Tasso di Crescita (%)',
        yaxis2_title='Indice di Stress',
        hovermode='x unified',
        showlegend=True,
        height=500
    )

    return fig


def create_production_impact_figure(sim_data: pd.DataFrame) -> go.Figure:
    """Crea il grafico dell'impatto sulla produzione"""
    # Calcola medie mensili
    monthly_data = sim_data.set_index('date').resample('M').mean()

    fig = go.Figure()

    # Aggiunge il grafico a barre della produzione stimata
    fig.add_trace(
        go.Bar(
            x=monthly_data.index,
            y=100 * (1 - monthly_data['stress_index']),
            name='Produzione Stimata (%)',
            marker_color='#27AE60'
        )
    )

    # Configurazione layout
    fig.update_layout(
        title='Impatto Stimato sulla Produzione',
        xaxis_title='Mese',
        yaxis_title='Produzione Stimata (%)',
        hovermode='x unified',
        showlegend=True,
        height=500
    )

    return fig