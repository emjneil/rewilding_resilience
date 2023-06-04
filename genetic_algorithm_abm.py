# ------ Optimization of the Knepp ABM model --------
from model import KneppModel 
import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import statistics
import random

# ------ Optimization of the Knepp ABM model --------

print("running 1")

def objectiveFunction(x):

    # define the parameters (12)
    chance_reproduceSapling = x[0]
    chance_reproduceYoungScrub =x[1]
    chance_regrowGrass =x[2]
    chance_saplingBecomingTree =x[3]
    chance_youngScrubMatures =x[4]
    chance_scrubOutcompetedByTree = x[5]
    chance_grassOutcompetedByTree =x[6]
    chance_grassOutcompetedByScrub =x[7]

    # initial values
    roe_deer_reproduce =x[8]
    roe_deer_gain_from_grass =x[9]
    roe_deer_gain_from_trees = x[10]
    roe_deer_gain_from_scrub = x[11]
    roe_deer_gain_from_saplings = x[12]
    roe_deer_gain_from_young_scrub = x[13]
    fallow_deer_reproduce = x[14]
    fallow_deer_gain_from_grass = x[15]
    fallow_deer_gain_from_trees = x[16]
    fallow_deer_gain_from_scrub = x[17]
    fallow_deer_gain_from_saplings = x[18]
    fallow_deer_gain_from_young_scrub = x[19]
    red_deer_reproduce = x[20]
    red_deer_gain_from_grass = x[21]
    red_deer_gain_from_trees = x[22]
    red_deer_gain_from_scrub = x[23]
    red_deer_gain_from_saplings = x[24]
    red_deer_gain_from_young_scrub = x[25]
    ponies_gain_from_grass = x[26]
    ponies_gain_from_trees = x[27]
    ponies_gain_from_scrub = x[28]
    ponies_gain_from_saplings = x[29]
    ponies_gain_from_young_scrub = x[30]
    cattle_reproduce = x[31]
    cows_gain_from_grass = x[32]
    cows_gain_from_trees = x[33]
    cows_gain_from_scrub = x[34]
    cows_gain_from_saplings = x[35]
    cows_gain_from_young_scrub = x[36]
    tamworth_pig_reproduce = x[37]
    tamworth_pig_gain_from_grass =x[38]
    tamworth_pig_gain_from_trees = x[39]
    tamworth_pig_gain_from_scrub = x[40]
    tamworth_pig_gain_from_saplings =x[41]
    tamworth_pig_gain_from_young_scrub = x[42]
    european_bison_reproduce = 0
    # bison should have higher impact than any other consumer
    european_bison_gain_from_grass =  0
    european_bison_gain_from_trees =0
    european_bison_gain_from_scrub =0
    european_bison_gain_from_saplings = 0
    european_bison_gain_from_young_scrub = 0  
    # euro elk parameters
    european_elk_reproduce = 0
    # bison should have higher impact than any other consumer
    european_elk_gain_from_grass =  0
    european_elk_gain_from_trees = 0
    european_elk_gain_from_scrub = 0
    european_elk_gain_from_saplings =  0
    european_elk_gain_from_young_scrub =  0
    # reindeer parameters
    reindeer_reproduce = 0
    # reindeer should have impacts between red and fallow deer
    reindeer_gain_from_grass = 0
    reindeer_gain_from_trees =0
    reindeer_gain_from_scrub =0
    reindeer_gain_from_saplings = 0
    reindeer_gain_from_young_scrub = 0
    # stocking values
    initial_roe = 12
    fallowDeer_stocking = 247
    cattle_stocking = 81
    redDeer_stocking = 35
    tamworthPig_stocking = 7
    exmoor_stocking = 15
    initial_wood = 0.058
    initial_grass = 0.899
    initial_scrub = 0.043

    # forecasting
    fallowDeer_stocking_forecast = 86
    cattle_stocking_forecast = 76
    redDeer_stocking_forecast = 21
    tamworthPig_stocking_forecast = 7
    exmoor_stocking_forecast = 15
    roeDeer_stocking_forecast = 12
    introduced_species_stocking_forecast = 0
    # tree mortality
    chance_scrub_saves_saplings = x[43]

    exp_chance_reproduceSapling = 0
    exp_chance_reproduceYoungScrub =  0
    exp_chance_regrowGrass = 0
    duration = 0
    tree_reduction = 0
    
    # run the model
    model = KneppModel(initial_roe, roe_deer_reproduce, roe_deer_gain_from_saplings, roe_deer_gain_from_trees, roe_deer_gain_from_scrub, roe_deer_gain_from_young_scrub, roe_deer_gain_from_grass,
                        chance_youngScrubMatures, chance_saplingBecomingTree, chance_reproduceSapling,chance_reproduceYoungScrub, chance_regrowGrass, 
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
                        exp_chance_reproduceSapling, exp_chance_reproduceYoungScrub, exp_chance_regrowGrass, duration, tree_reduction,
                        reintroduction = True, introduce_euroBison = False, introduce_elk = False, 
                        experiment_growth = False, experiment_wood = False, experiment_linear_growth = False)


    model.run_model()

    # remember the results of the model (dominant conditions, # of agents)
    results = model.datacollector.get_model_vars_dataframe()

    # get the last year
    end = results.iloc[-1]

    # if it's < 6000, return a high number (means roe deer were too high)
    if end["Time"] < 5999:
        print("failed", end["Roe deer"])
        return 1000-(abs(end["Roe deer"]-50))


    else:
        # calculate gradient - has it reached equilibrium?
        last_hundred = results.loc[results["Time"] >= 4800]
        # only species I care about - ones that aren't controlled
        uncontrolled_only = results[['Grassland', 'Woodland', 'Thorny Scrub', 'Roe deer']]

        gradients = {}

        for column in uncontrolled_only.columns:
            # Get the population values for the current species
            population_values = uncontrolled_only[column].values
            # Create an array representing the years (assuming the index represents the years)
            years = uncontrolled_only.index.values
            # Fit a first-degree polynomial (line) to the population values
            coefficients = np.polyfit(years, population_values, deg=1)
            # Extract the gradient (slope) from the coefficients
            gradient = coefficients[0]
            # Store the gradient in the dictionary
            gradients[column] = gradient

        # if it's not at equilibrium, fail it
        if abs(gradients["Roe deer"]) > 0.05 or abs(gradients["Thorny Scrub"]) > 0.05 or abs(gradients["Grassland"]) > 0.05 or abs(gradients["Woodland"]) > 0.05:
            print("not at equilibrium", gradients)
            return 5000


        else:  
            # find the middle of each filter
            filtered_result = (
                # pre-reintro model
                ((((list(results.loc[results['Time'] == 49, 'Roe deer'])[0])-26)/26)**2) +
                ((((list(results.loc[results['Time'] == 49, 'Grassland'])[0])-58.4)/58.4)**2) +
                ((((list(results.loc[results['Time'] == 49, 'Thorny Scrub'])[0])-28.1)/28.1)**2) +
                ((((list(results.loc[results['Time'] == 49, 'Woodland'])[0])-11.4)/11.4)**2) +
                # March 2019
                ((((list(results.loc[results['Time'] == 169, 'Fallow deer'])[0])-278)/278)**2) +
                ((((list(results.loc[results['Time'] == 169, 'Red deer'])[0])-37)/37)**2) +
                # March 2020
                ((((list(results.loc[results['Time'] == 181, 'Fallow deer'])[0])-247)/247)**2) +
                ((((list(results.loc[results['Time'] == 181, 'Red deer'])[0])-35)/35)**2) +
                ((((list(results.loc[results['Time'] == 181, 'Longhorn cattle'])[0])-81)/81)**2) +
                # May 2020
                ((((list(results.loc[results['Time'] == 183, 'Exmoor pony'])[0])-15)/15)**2) +
                ((((list(results.loc[results['Time'] == 183, 'Longhorn cattle'])[0])-81)/81)**2) +
                ((((list(results.loc[results['Time'] == 183, 'Roe deer'])[0])-50)/50)**2) +
                ((((list(results.loc[results['Time'] == 183, 'Grassland'])[0])-26.8)/26.8)**2) +
                ((((list(results.loc[results['Time'] == 183, 'Thorny Scrub'])[0])-51.8)/51.8)**2) +
                ((((list(results.loc[results['Time'] == 183, 'Woodland'])[0])-21.5)/21.5)**2) +
                # Tamworth pigs: average pop between 2015-2021 (based monthly data) = 15 
                ((((statistics.mean(list(results.loc[results['Time'] > 121, 'Tamworth pigs'])))-15)/15)**2)
                )


            # only print the last year's result if it's reasonably close to the filters
            if filtered_result < 10:
                print("r:", filtered_result)
                with pd.option_context('display.max_columns',None):
                    just_nodes = results[results['Time'] == 5999]
                    print(just_nodes[["Time", "Roe deer", "Exmoor pony", "Fallow deer", "Longhorn cattle", "Red deer", "Tamworth pigs", "Grassland", "Woodland", "Thorny Scrub", "Bare ground"]])
            else:
                print("n:", int(filtered_result))
            
            # return the output
            return filtered_result
   

# assumptions: 
## herbivores need to eat 1 sapling/young scrub per day if no other food, and 10 trees/mo
## inter-habitat competition and reproduction of saplings/scrbu capped at 10% per month to allow herbivory to have a tangible impact
## first and last filters for (roe, habitats), last two sets of filters for each introduced species. So each species targeted with 2 filters. Last two were chosen as these were hardest to pass for fallow/red deer
## no roe deer explosion


# try high herbivory, low competition 1% 10-50yrs to mature
def run_optimizer():
    # Define the bounds
    bds = np.array([
        [0,0.15],[0,1],[0,1],[0.0017,0.0083],[0.0024,0.028],
        [0,1], # mature scrub competition
        [0,1],[0,1], # grass competition
        # roe parameters
        [0,0.75],[0,1],[0,0.33],[0,0.33],[0,0.1],[0,0.1],
        # fallow deer parameters
        [0.3,0.75],[0,1],[0,0.33],[0,0.33],[0,0.1],[0,0.1],
        # red deer
        [0.3,0.75],[0,1],[0,0.33],[0,0.33],[0,0.1],[0,0.1],
        # exmoor pony parameters
                [0,1],[0,0.33],[0,0.33],[0,0.1],[0,0.1],
        # cattle parameters
        [0,0.5],[0,1],[0,0.33],[0,0.33],[0,0.1],[0,0.1],
        # pig parameters
        [0,0.75],[0,1],[0,0.33],[0,0.33],[0,0.1],[0,0.1],
        # sapling protection parameter
        [0,1]])


    # popsize and maxiter are defined at the top of the page, was 10x100
    algorithm_param = {'max_num_iteration':15,
                    'population_size':100,\
                    'mutation_probability':0.1,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.5,\
                    'parents_portion': 0.3,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv': 8}


    optimization =  ga(function = objectiveFunction, dimension = 44, variable_type = 'real',variable_boundaries= bds, algorithm_parameters = algorithm_param, function_timeout=6000)
    optimization.run()
    outputs = list(optimization.output_dict["variable"]) + [(optimization.output_dict["function"])]
    # return excel with rows = output values and number of filters passed
    pd.DataFrame(outputs).to_excel('optim_outputs_1.xlsx', header=False, index=False)
    return optimization.output_dict


run_optimizer()

