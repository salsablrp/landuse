import operator
import numpy as np

# Define allowed operators
OPS = {
    "multiply": operator.mul,
    "add": operator.add,
    "subtract": operator.sub,
    "divide": operator.truediv,
}

def apply_scenario(stack, predictor_files, scenario_def):
    """
    Apply a user-defined scenario to the predictor stack.

    Parameters
    ----------
    stack : np.ndarray
        3D array of predictor layers (n_layers, height, width).
    predictor_files : list of str
        List of predictor raster filenames in the same order as stack.
    scenario_def : dict
        Scenario definition, e.g.
        {
            "name": "afforestation",
            "changes": [
                {"layer": "dist_remote_area.tif", "op": "multiply", "value": 0.7},
                {"layer": "Kazakhstan_AGB.tif", "op": "multiply", "value": 1.15}
            ]
        }

    Returns
    -------
    np.ndarray
        Modified predictor stack.
    """
    modified_stack = stack.copy()

    for change in scenario_def.get("changes", []):
        layer_name = change["layer"]
        op_name = change["op"]
        value = change["value"]

        if layer_name not in predictor_files:
            print(f"⚠️ Layer {layer_name} not found in predictor_files, skipping.")
            continue

        if op_name not in OPS:
            print(f"⚠️ Operator {op_name} not recognized. Allowed: {list(OPS.keys())}")
            continue

        idx = predictor_files.index(layer_name)
        func = OPS[op_name]
        modified_stack[idx] = func(modified_stack[idx], value)

    return modified_stack

# from .config import PREDICTOR_FILES

# def scenario_afforestation(stack):
#     remote = PREDICTOR_FILES.index("dist_remote_area.tif")
#     agb = PREDICTOR_FILES.index("Kazakhstan_AGB.tif")
#     stack[remote] *= 0.7
#     stack[agb] *= 1.15
#     return stack

# def scenario_urban_growth(stack):
#     uni = PREDICTOR_FILES.index("dist_uni.tif")
#     pop = PREDICTOR_FILES.index("kaz_pd_2020_1km_UNadj.tif")
#     stack[uni] *= 0.6
#     stack[pop] *= 1.25
#     return stack

# def scenario_desertification(stack):
#     sm = PREDICTOR_FILES.index("Kazakhstan_SoilMoisture.tif")
#     water = PREDICTOR_FILES.index("dist_waterways.tif")
#     agb = PREDICTOR_FILES.index("Kazakhstan_AGB.tif")
#     stack[sm] *= 0.7
#     stack[water] *= 1.3
#     stack[agb] *= 0.6
#     return stack

# def scenario_flooding(stack):
#     sm = PREDICTOR_FILES.index("Kazakhstan_SoilMoisture.tif")
#     water = PREDICTOR_FILES.index("dist_waterways.tif")
#     agb = PREDICTOR_FILES.index("Kazakhstan_AGB.tif")
#     stack[sm] *= 1.3
#     stack[water] *= 0.7
#     stack[agb] *= 0.9
#     return stack

# def scenario_conservation(stack):
#     road = PREDICTOR_FILES.index("dist_road.tif")
#     pop = PREDICTOR_FILES.index("kaz_pd_2020_1km_UNadj.tif")
#     agb = PREDICTOR_FILES.index("Kazakhstan_AGB.tif")
#     sm = PREDICTOR_FILES.index("Kazakhstan_SoilMoisture.tif")
#     stack[road] *= 1.5
#     stack[pop] *= 0.8
#     stack[agb] *= 1.2
#     stack[sm] *= 1.1
#     return stack
