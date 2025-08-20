from .config import PREDICTOR_FILES

def scenario_afforestation(stack):
    remote = PREDICTOR_FILES.index("dist_remote_area.tif")
    agb = PREDICTOR_FILES.index("Kazakhstan_AGB.tif")
    stack[remote] *= 0.7
    stack[agb] *= 1.15
    return stack

def scenario_urban_growth(stack):
    uni = PREDICTOR_FILES.index("dist_uni.tif")
    pop = PREDICTOR_FILES.index("kaz_pd_2020_1km_UNadj.tif")
    stack[uni] *= 0.6
    stack[pop] *= 1.25
    return stack

def scenario_desertification(stack):
    sm = PREDICTOR_FILES.index("Kazakhstan_SoilMoisture.tif")
    water = PREDICTOR_FILES.index("dist_waterways.tif")
    agb = PREDICTOR_FILES.index("Kazakhstan_AGB.tif")
    stack[sm] *= 0.7
    stack[water] *= 1.3
    stack[agb] *= 0.6
    return stack

def scenario_flooding(stack):
    sm = PREDICTOR_FILES.index("Kazakhstan_SoilMoisture.tif")
    water = PREDICTOR_FILES.index("dist_waterways.tif")
    agb = PREDICTOR_FILES.index("Kazakhstan_AGB.tif")
    stack[sm] *= 1.3
    stack[water] *= 0.7
    stack[agb] *= 0.9
    return stack

def scenario_conservation(stack):
    road = PREDICTOR_FILES.index("dist_road.tif")
    pop = PREDICTOR_FILES.index("kaz_pd_2020_1km_UNadj.tif")
    agb = PREDICTOR_FILES.index("Kazakhstan_AGB.tif")
    sm = PREDICTOR_FILES.index("Kazakhstan_SoilMoisture.tif")
    stack[road] *= 1.5
    stack[pop] *= 0.8
    stack[agb] *= 1.2
    stack[sm] *= 1.1
    return stack
