src/data/data_loader.py:
- load_weather_data()
- load_olive_varieties()
- read_json_files()
- load_single_model_and_scalers()
- save_single_model_and_scalers()

src/data/data_processor.py:
- preprocess_weather_data()
- prepare_solar_data()
- prepare_transformer_data()
- create_sequences()
- encode_techniques()
- decode_techniques()

src/data/data_simulator.py:
- simulate_zone()
- simulate_olive_production_parallel()
- calculate_weather_effect()
- calculate_water_need()
- add_olive_water_consumption_correlation()

src/features/temporal_features.py:
- add_time_features()
- get_season()
- get_time_period()
- create_time_based_features()

src/features/weather_features.py:
- add_solar_features()
- add_solar_specific_features()
- add_environmental_features()
- calculate_vpd()
- add_weather_indicators()

src/features/olive_features.py:
- create_technique_mapping()
- add_olive_features()
- calculate_stress_index()
- calculate_quality_indicators()
- add_production_features()

src/models/transformer.py:
- create_olive_oil_transformer()
- OliveTransformerBlock
- PositionalEncoding
- DataAugmentation

src/models/layers.py:
- MultiScaleAttention
- TemporalConvBlock
- WeatherEmbedding
- OliveVarietyEmbedding

src/models/callbacks.py:
- CustomCallback
- WarmUpLearningRateSchedule
- MetricLogger
- EarlyStoppingWithBest

src/models/training.py:
- compile_model()
- setup_transformer_training()
- train_transformer()
- retrain_model()
- create_callbacks()

src/visualization/plots.py:
- plot_variety_comparison()
- plot_efficiency_vs_production()
- plot_water_efficiency_vs_production()
- plot_water_need_vs_oil_production()
- save_plot()

src/visualization/dashboard.py:
- create_production_dashboard()
- create_weather_dashboard()
- create_efficiency_dashboard()
- update_dashboard_data()
- create_forecast_view()

src/utils/metrics.py:
- calculate_real_error()
- evaluate_model_performance()
- calculate_efficiency_metrics()
- calculate_forecast_accuracy()
- compute_confidence_intervals()

src/utils/helpers.py:
- get_optimal_workers()
- clean_column_name()
- clean_column_names()
- to_camel_case()
- get_full_data()