from landuse_tool import data_loader, utils, training, prediction, scenarios, visualization

# Load data
lc, mask, profile = data_loader.load_target()
X_stack = data_loader.load_predictors(mask)

# Sample and train
X, y = utils.sample_training_data(X_stack, lc, mask)
model = training.train_rf(X, y)

# Predict base map
predicted = prediction.predict_map(model, X_stack, mask)
visualization.plot_prediction(predicted, cmap_list, title="Baseline Prediction")

# Run scenario
X_stack_afforest = scenarios.scenario_afforestation(X_stack.copy())
afforest_map = prediction.predict_map(model, X_stack_afforest, mask)
visualization.plot_prediction(afforest_map, cmap_list, title="Afforestation Scenario")
