# ------ Optimization towards desired end state --------
from model import KneppModel 
import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import statistics
import random


# GA - remember to change the equilibrium time-step, duration, growth changes

def objectiveFunction(x):

    # pick the best parameter
    best_parameter = pd.read_csv('best_parameter_set.csv').drop(['Unnamed: 0'], axis=1)


    # set parameters
    chance_reproduceSapling = best_parameter["chance_reproduceSapling"].item()
    chance_reproduceYoungScrub =  best_parameter["chance_reproduceYoungScrub"].item()
    chance_regrowGrass =  best_parameter["chance_regrowGrass"].item()
    chance_saplingBecomingTree =  best_parameter["chance_saplingBecomingTree"].item()
    chance_youngScrubMatures =  best_parameter["chance_youngScrubMatures"].item()
    chance_scrubOutcompetedByTree =  best_parameter["chance_scrubOutcompetedByTree"].item()
    chance_grassOutcompetedByTree =  best_parameter["chance_grassOutcompetedByTree"].item()
    chance_grassOutcompetedByScrub = best_parameter["chance_grassOutcompetedByScrub"].item()
    chance_scrub_saves_saplings = best_parameter["chance_scrub_saves_saplings"].item()

    initial_roe = 12
    initial_grass = 0.899
    initial_scrub = 0.043
    initial_wood = 0.058

    roe_deer_reproduce = best_parameter["roe_deer_reproduce"].item()
    roe_deer_gain_from_grass =  best_parameter["roe_deer_gain_from_grass"].item()
    roe_deer_gain_from_trees =  best_parameter["roe_deer_gain_from_trees"].item()
    roe_deer_gain_from_scrub =  best_parameter["roe_deer_gain_from_scrub"].item()
    roe_deer_gain_from_saplings =  best_parameter["roe_deer_gain_from_saplings"].item()
    roe_deer_gain_from_young_scrub =  best_parameter["roe_deer_gain_from_young_scrub"].item()
    ponies_gain_from_grass =  best_parameter["ponies_gain_from_grass"].item()
    ponies_gain_from_trees =  best_parameter["ponies_gain_from_trees"].item()
    ponies_gain_from_scrub =  best_parameter["ponies_gain_from_scrub"].item()
    ponies_gain_from_saplings =  best_parameter["ponies_gain_from_saplings"].item()
    ponies_gain_from_young_scrub =  best_parameter["ponies_gain_from_young_scrub"].item()
    cattle_reproduce =  best_parameter["cattle_reproduce"].item()
    cows_gain_from_grass =  best_parameter["cows_gain_from_grass"].item()
    cows_gain_from_trees =  best_parameter["cows_gain_from_trees"].item()
    cows_gain_from_scrub =  best_parameter["cows_gain_from_scrub"].item()
    cows_gain_from_saplings =  best_parameter["cows_gain_from_saplings"].item()
    cows_gain_from_young_scrub =  best_parameter["cows_gain_from_young_scrub"].item()
    fallow_deer_reproduce =  best_parameter["fallow_deer_reproduce"].item()
    fallow_deer_gain_from_grass =  best_parameter["fallow_deer_gain_from_grass"].item()
    fallow_deer_gain_from_trees =  best_parameter["fallow_deer_gain_from_trees"].item()
    fallow_deer_gain_from_scrub =  best_parameter["fallow_deer_gain_from_scrub"].item()
    fallow_deer_gain_from_saplings =  best_parameter["fallow_deer_gain_from_saplings"].item()
    fallow_deer_gain_from_young_scrub =  best_parameter["fallow_deer_gain_from_young_scrub"].item()
    red_deer_reproduce =  best_parameter["red_deer_reproduce"].item()
    red_deer_gain_from_grass =  best_parameter["red_deer_gain_from_grass"].item()
    red_deer_gain_from_trees =  best_parameter["red_deer_gain_from_trees"].item()
    red_deer_gain_from_scrub =  best_parameter["red_deer_gain_from_scrub"].item()
    red_deer_gain_from_saplings =  best_parameter["red_deer_gain_from_saplings"].item()
    red_deer_gain_from_young_scrub =  best_parameter["red_deer_gain_from_young_scrub"].item()
    tamworth_pig_reproduce =  best_parameter["tamworth_pig_reproduce"].item()
    tamworth_pig_gain_from_grass =  best_parameter["tamworth_pig_gain_from_grass"].item()
    tamworth_pig_gain_from_trees = best_parameter["tamworth_pig_gain_from_trees"].item()
    tamworth_pig_gain_from_scrub = best_parameter["tamworth_pig_gain_from_scrub"].item()
    tamworth_pig_gain_from_saplings =  best_parameter["tamworth_pig_gain_from_saplings"].item()
    tamworth_pig_gain_from_young_scrub =  best_parameter["tamworth_pig_gain_from_young_scrub"].item()

    # euro bison parameters
    european_bison_reproduce = random.uniform(cattle_reproduce-(cattle_reproduce*0.1), cattle_reproduce+(cattle_reproduce*0.1))
    # bison should have higher impact than any other consumer
    european_bison_gain_from_grass = random.uniform(cows_gain_from_grass, cows_gain_from_grass+(cows_gain_from_grass*0.1))
    european_bison_gain_from_trees =random.uniform(cows_gain_from_trees, cows_gain_from_trees+(cows_gain_from_trees*0.1))
    european_bison_gain_from_scrub =random.uniform(cows_gain_from_scrub, cows_gain_from_scrub+(cows_gain_from_scrub*0.1))
    european_bison_gain_from_saplings = random.uniform(cows_gain_from_saplings, cows_gain_from_saplings+(cows_gain_from_saplings*0.1))
    european_bison_gain_from_young_scrub = random.uniform(cows_gain_from_young_scrub, cows_gain_from_young_scrub+(cows_gain_from_young_scrub*0.1))
    # euro elk parameters
    european_elk_reproduce = random.uniform(red_deer_reproduce-(red_deer_reproduce*0.1), red_deer_reproduce+(red_deer_reproduce*0.1))
    # bison should have higher impact than any other consumer
    european_elk_gain_from_grass =  random.uniform(red_deer_gain_from_grass-(red_deer_gain_from_grass*0.1), red_deer_gain_from_grass)
    european_elk_gain_from_trees = random.uniform(red_deer_gain_from_trees-(red_deer_gain_from_trees*0.1), red_deer_gain_from_trees)
    european_elk_gain_from_scrub = random.uniform(red_deer_gain_from_scrub-(red_deer_gain_from_scrub*0.1), red_deer_gain_from_scrub)
    european_elk_gain_from_saplings =  random.uniform(red_deer_gain_from_saplings-(red_deer_gain_from_saplings*0.1), red_deer_gain_from_saplings)
    european_elk_gain_from_young_scrub =  random.uniform(red_deer_gain_from_young_scrub-(red_deer_gain_from_young_scrub*0.1), red_deer_gain_from_young_scrub)

    # forecasting parameters
    fallowDeer_stocking_forecast = 247
    cattle_stocking_forecast = 81
    redDeer_stocking_forecast =  35
    tamworthPig_stocking_forecast =  7
    exmoor_stocking_forecast = 15

    
    # way far ahead forecasting parameters
    far_fallowDeer_stocking_forecast = int(247 + (x[0]*247))
    far_cattle_stocking_forecast = int(81 + (x[1]*81))
    far_redDeer_stocking_forecast =  int(35 + (x[2]*35))
    far_tamworthPig_stocking_forecast =  int(7 + (x[3]*7))
    far_exmoor_stocking_forecast = int(15 + (x[4]*15))
    print(far_fallowDeer_stocking_forecast)

    growth_changes = 0 #### fill these in once we know
    duration = 0

    # experiment parameters
    exp_chance_reproduceSapling = best_parameter["chance_reproduceSapling"].item() * growth_changes
    exp_chance_reproduceYoungScrub =  best_parameter["chance_reproduceYoungScrub"].item() * growth_changes
    exp_chance_regrowGrass =  best_parameter["chance_regrowGrass"].item() * growth_changes

    
    noise_amount = 0

    random.seed(1)
    np.random.seed(1)

    model = KneppModel(initial_roe, roe_deer_reproduce, roe_deer_gain_from_saplings, roe_deer_gain_from_trees, roe_deer_gain_from_scrub, roe_deer_gain_from_young_scrub, roe_deer_gain_from_grass,
                        chance_youngScrubMatures, chance_saplingBecomingTree, 
                        chance_reproduceSapling, chance_reproduceYoungScrub, chance_regrowGrass, 
                        chance_grassOutcompetedByTree, chance_grassOutcompetedByScrub, chance_scrubOutcompetedByTree, 
                        ponies_gain_from_saplings, ponies_gain_from_trees, ponies_gain_from_scrub, ponies_gain_from_young_scrub, ponies_gain_from_grass, 
                        cattle_reproduce, cows_gain_from_grass, cows_gain_from_trees, cows_gain_from_scrub, cows_gain_from_saplings, cows_gain_from_young_scrub, 
                        fallow_deer_reproduce, fallow_deer_gain_from_saplings, fallow_deer_gain_from_trees, fallow_deer_gain_from_scrub, fallow_deer_gain_from_young_scrub, fallow_deer_gain_from_grass,
                        red_deer_reproduce, red_deer_gain_from_saplings, red_deer_gain_from_trees, red_deer_gain_from_scrub, red_deer_gain_from_young_scrub, red_deer_gain_from_grass,
                        tamworth_pig_reproduce, tamworth_pig_gain_from_saplings,tamworth_pig_gain_from_trees,tamworth_pig_gain_from_scrub,tamworth_pig_gain_from_young_scrub,tamworth_pig_gain_from_grass,
                        european_bison_reproduce, european_bison_gain_from_grass, european_bison_gain_from_trees, european_bison_gain_from_scrub, european_bison_gain_from_saplings, european_bison_gain_from_young_scrub,
                        european_elk_reproduce, european_elk_gain_from_grass, european_elk_gain_from_trees, european_elk_gain_from_scrub, european_elk_gain_from_saplings, european_elk_gain_from_young_scrub,
                        fallowDeer_stocking_forecast, cattle_stocking_forecast, redDeer_stocking_forecast, tamworthPig_stocking_forecast, exmoor_stocking_forecast,
                        chance_scrub_saves_saplings, initial_wood, initial_grass, initial_scrub,
                        exp_chance_reproduceSapling, exp_chance_reproduceYoungScrub, exp_chance_regrowGrass, duration, noise_amount,
                        far_fallowDeer_stocking_forecast, far_cattle_stocking_forecast, far_redDeer_stocking_forecast, far_tamworthPig_stocking_forecast, far_exmoor_stocking_forecast,
                        reintroduction = True, introduce_euroBison = False, introduce_elk = False, 
                        experiment_growth = True, experiment_wood = False, experiment_linear_growth = False, run_ga = True
                        )


    model.reset_randomizer(seed=1)

    model.run_model()

    results = model.datacollector.get_model_vars_dataframe()

    # get the last year
    last_year = results.iloc[-1]

    print(last_year)

    # target the desired state for vegetation - change this!
    filtered_result = (
        (((last_year['Grassland']-58.4)/58.4)**2) +
        (((last_year['Thorny Scrub']-58.4)/58.4)**2) +
        (((last_year['Woodland']-58.4)/58.4)**2))


    # only print the last year's result if it's reasonably close to the filters
    if filtered_result < 2:
        print("r:", filtered_result)
        with pd.option_context('display.max_columns',None):
            print(last_year[["Time", "Roe deer", "Grassland", "Woodland", "Thorny Scrub", "Bare ground"]])
    else:
        print("n:", int(filtered_result))

    # return the output
    return filtered_result
   


def run_optimizer():
    # Define the bounds
    bds = np.array([
        [-1, 5],[-1, 5],[-1, 5],[-1, 5],[-1, 5]])


    # popsize and maxiter are defined at the top of the page, was 10x100
    algorithm_param = {'max_num_iteration':15,
                    'population_size':50,\
                    'mutation_probability':0.1,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.5,\
                    'parents_portion': 0.3,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv': 8}


    optimization =  ga(function = objectiveFunction, dimension = 5, variable_type = 'real',variable_boundaries= bds, algorithm_parameters = algorithm_param, function_timeout=6000)
    optimization.run()
    outputs = list(optimization.output_dict["variable"]) + [(optimization.output_dict["function"])]
    # return excel with rows = output values and number of filters passed
    pd.DataFrame(outputs).to_excel('ga_outputs.xlsx', header=False, index=False)
    return optimization.output_dict


run_optimizer()