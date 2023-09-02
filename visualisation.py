from model import KneppModel
import mesa
import mesa_geo as mg
from mesa.visualization.modules import ChartModule
from agents import FieldAgent, roe_deer_agent, exmoor_pony_agent, longhorn_cattle_agent, fallow_deer_agent, red_deer_agent, tamworth_pig_agent, european_bison_agent, european_elk_agent

# visualise 
def schelling_draw(agent):
    portrayal = dict()
    if isinstance(agent, FieldAgent):
        if agent.condition == "grassland":
            portrayal["color"] = "#3C873A"
        if agent.condition == "thorny_scrubland":
            portrayal["color"] = "#FFD43B"
        if agent.condition == "woodland":
            portrayal["color"] = "Blue"        
        if agent.condition == "bare_ground":
            portrayal["color"] = "Brown"
    elif isinstance(agent, roe_deer_agent):
        portrayal["radius"] = 1
        portrayal["shape"] = "circle"
        portrayal["color"] = "Red"
    elif isinstance(agent, exmoor_pony_agent):
        portrayal["radius"] = 1
        portrayal["shape"] = "circle"
        portrayal["color"] = "Purple"
    elif isinstance(agent, longhorn_cattle_agent):
        portrayal["radius"] = 1
        portrayal["shape"] = "circle"
        portrayal["color"] = "#4B8BBE"
    elif isinstance(agent, fallow_deer_agent):
        portrayal["radius"] = 1
        portrayal["shape"] = "circle"
        portrayal["color"] = "#F16529"
    elif isinstance(agent, red_deer_agent):
        portrayal["radius"] = 1
        portrayal["shape"] = "circle"
        portrayal["color"] = "#BC473A"
    elif isinstance(agent, tamworth_pig_agent):
        portrayal["radius"] = 1
        portrayal["shape"] = "circle"
        portrayal["color"] = "Black"
    return portrayal

map_element = mg.visualization.MapModule(schelling_draw, [50.971, -0.376], 14)

chart_element_herbivores = ChartModule([{"Label": "Roe deer", "Color": "Red"}, 
                            {"Label": "Exmoor pony", "Color": "Purple"},
                            {"Label": "Fallow deer", "Color": "#F16529"},
                            {"Label": "Longhorn cattle", "Color": "#4B8BBE"},
                            {"Label": "Red deer", "Color": "#BC473A"},
                            {"Label": "Tamworth pigs", "Color": "Black"}])

chart_element_habitats = ChartModule([{"Label": "Grassland", "Color": "#3C873A"},
                             {"Label": "Bare ground", "Color": "#560000"},
                             {"Label": "Thorny Scrub", "Color": "#FFD43B"},
                             {"Label": "Woodland", "Color": "Blue"}])


server = mesa.visualization.ModularServer(
    KneppModel, 
    [map_element, chart_element_herbivores, chart_element_habitats],
    "Knepp Estate", {"initial_roe":12,"roe_deer_reproduce":0.17, "roe_deer_gain_from_saplings":0.09,"roe_deer_gain_from_trees":0.14, "roe_deer_gain_from_scrub":0.04,"roe_deer_gain_from_young_scrub":0.01, "roe_deer_gain_from_grass":0.25,
                    "chance_youngScrubMatures":0.009, "chance_saplingBecomingTree":0.006, "chance_reproduceSapling":0.009,"chance_reproduceYoungScrub":0.96, "chance_regrowGrass":0.38, 
                    "chance_grassOutcompetedByTree": 0.89, "chance_grassOutcompetedByScrub":0.54,"chance_scrubOutcompetedByTree":0.83, 
                    "ponies_gain_from_saplings": 0.003, "ponies_gain_from_trees":0.24, "ponies_gain_from_scrub":0.2, "ponies_gain_from_young_scrub":0.02, "ponies_gain_from_grass":0.4, 
                    "cattle_reproduce":0.21, "cows_gain_from_grass":0.38, "cows_gain_from_trees":0.07, "cows_gain_from_scrub":0.01, "cows_gain_from_saplings":0.07, "cows_gain_from_young_scrub":0.07,
                    "fallow_deer_reproduce":0.3, "fallow_deer_gain_from_saplings":0.25, "fallow_deer_gain_from_trees":0.5, "fallow_deer_gain_from_scrub":0.5, "fallow_deer_gain_from_young_scrub":0.25, "fallow_deer_gain_from_grass":0.8,
                    "red_deer_reproduce":0.59, "red_deer_gain_from_saplings":0.07, "red_deer_gain_from_trees":0.01, "red_deer_gain_from_scrub":0.32, "red_deer_gain_from_young_scrub":0.01, "red_deer_gain_from_grass":0.34,
                    "tamworth_pig_reproduce":0.47, "tamworth_pig_gain_from_saplings":0.03,"tamworth_pig_gain_from_trees":0.18,"tamworth_pig_gain_from_scrub":0.24,"tamworth_pig_gain_from_young_scrub":0.07,"tamworth_pig_gain_from_grass":0.18,
                    "european_bison_reproduce":0.2, "european_bison_gain_from_grass":0.8, "european_bison_gain_from_trees":0.5, "european_bison_gain_from_scrub":0.5, "european_bison_gain_from_saplings":0.2, "european_bison_gain_from_young_scrub":0.2,
                    "european_elk_reproduce":0.2, "european_elk_gain_from_grass":0.8, "european_elk_gain_from_trees":0.5, "european_elk_gain_from_scrub":0.5, "european_elk_gain_from_saplings":0.2, "european_elk_gain_from_young_scrub":0.2,
                    "fallowDeer_stocking_forecast":247*50, "cattle_stocking_forecast":81*50, "redDeer_stocking_forecast":35*50, "tamworthPig_stocking_forecast":7*50, "exmoor_stocking_forecast":15*50, 
                    "chance_scrub_saves_saplings":0.17, "initial_wood":0.058, "initial_grass":0.899, "initial_scrub":0.043,
                    "exp_chance_reproduceSapling":0, "exp_chance_reproduceYoungScrub":0, "exp_chance_regrowGrass": 0, "duration":0, "tree_reduction":0,
                    "reintroduction": True, "introduce_euroBison": False, "introduce_elk": False, "experiment_growth": False, "experiment_wood": False, 
                    "experiment_linear_growth": False, 
                    "max_time": 6000, "max_roe": 500
}
)


server.port = 8521 # The default
server.launch()