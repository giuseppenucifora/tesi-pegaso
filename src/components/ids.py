# components/ids.py
from dataclasses import dataclass

@dataclass
class Ids:
    # Auth Container
    AUTH_CONTAINER = 'auth-container'
    DASHBOARD_CONTAINER = 'dashboard-container'

    # Login Form
    LOGIN_FORM = 'login-form'
    LOGIN_USERNAME = 'login-username'
    LOGIN_PASSWORD = 'login-password'
    LOGIN_BUTTON = 'login-button'
    LOGIN_ERROR = 'login-error'

    # Register Form
    REGISTER_FORM = 'register-form'
    REGISTER_USERNAME = 'register-username'
    REGISTER_PASSWORD = 'register-password'
    REGISTER_CONFIRM = 'register-confirm'
    REGISTER_BUTTON = 'register-button'
    REGISTER_ERROR = 'register-error'
    REGISTER_SUCCESS = 'register-success'

    # Navigation
    SHOW_REGISTER_BUTTON = 'show-register-button'
    SHOW_LOGIN_BUTTON = 'show-login-button'

    # Inference
    INFERENCE_CONTAINER = 'inference-container'
    INFERENCE_STATUS = 'inference-status'
    INFERENCE_MODE = 'inference-mode'
    INFERENCE_LATENCY = 'inference-latency'
    INFERENCE_REQUESTS = 'inference-requests'
    INFERENCE_COUNTER = 'inference-counter'
    DEBUG_SWITCH = 'debug-switch'

    # Simulation
    SIMULATE_BUTTON = 'simulate-btn'
    GROWTH_CHART = 'growth-simulation-chart'
    PRODUCTION_CHART = 'production-simulation-chart'
    SIMULATION_SUMMARY = 'simulation-summary'
    KPI_CONTAINER = 'kpi-container'
    PRODUCTION_DEBUG_SWITCH = 'production-debug-switch'
    PRODUCTION_INFERENCE_REQUESTS = 'production-inference-requests'
    PRODUCTION_INFERENCE_MODE = 'production-inference-mode'

    # Environment Controls
    TEMP_SLIDER = 'temp-slider'
    HUMIDITY_SLIDER = 'humidity-slider'
    RAINFALL_INPUT = 'rainfall-input'
    RADIATION_INPUT = 'radiation-input'

    # Production Views
    OLIVE_PRODUCTION_HA = 'olive-production_ha'
    OIL_PRODUCTION_HA = 'oil-production_ha'
    WATER_NEED_HA = 'water-need_ha'
    OLIVE_PRODUCTION = 'olive-production'
    OIL_PRODUCTION = 'oil-production'
    WATER_NEED = 'water-need'
    PRODUCTION_DETAILS = 'production-details'
    WEATHER_IMPACT = 'weather-impact'
    WATER_NEEDS = 'water-needs'
    EXTRA_INFO = 'extra-info'

    # Configuration
    HECTARES_INPUT = 'hectares-input'
    VARIETY_1_DROPDOWN = 'variety-1-dropdown'
    TECHNIQUE_1_DROPDOWN = 'technique-1-dropdown'
    PERCENTAGE_1_INPUT = 'percentage-1-input'
    VARIETY_2_DROPDOWN = 'variety-2-dropdown'
    TECHNIQUE_2_DROPDOWN = 'technique-2-dropdown'
    PERCENTAGE_2_INPUT = 'percentage-2-input'
    VARIETY_3_DROPDOWN = 'variety-3-dropdown'
    TECHNIQUE_3_DROPDOWN = 'technique-3-dropdown'
    PERCENTAGE_3_INPUT = 'percentage-3-input'
    PERCENTAGE_WARNING = 'percentage-warning'

    # Cost Inputs
    COST_AMMORTAMENTO = 'cost-ammortamento'
    COST_ASSICURAZIONE = 'cost-assicurazione'
    COST_MANUTENZIONE = 'cost-manutenzione'
    COST_CERTIFICAZIONI = 'cost-certificazioni'
    COST_RACCOLTA = 'cost-raccolta'
    COST_POTATURA = 'cost-potatura'
    COST_FERTILIZZANTI = 'cost-fertilizzanti'
    COST_IRRIGAZIONE = 'cost-irrigazione'
    COST_MOLITURA = 'cost-molitura'
    COST_STOCCAGGIO = 'cost-stoccaggio'
    COST_BOTTIGLIA = 'cost-bottiglia'
    COST_ETICHETTATURA = 'cost-etichettatura'
    COST_MARKETING = 'cost-marketing'
    COST_COMMERCIALI = 'cost-commerciali'
    PRICE_OLIO = 'price-olio'
    PERC_VENDITA_DIRETTA = 'perc-vendita-diretta'

    # Other
    LOADING_ALERT = 'loading-alert'
    TABS = 'tabs'
    SAVE_CONFIG_BUTTON = 'save-config-button'
    SAVE_CONFIG_MESSAGE = 'save-config-message'
    LOGOUT_BUTTON = 'logout-button'
    DEV_MODE = 'dev-mode'