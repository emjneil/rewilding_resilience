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

print("running 105")

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
    
    
    random.seed(1)
    np.random.seed(1)
    
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


    model.reset_randomizer(seed=1)

    model.run_model()

    # remember the results of the model (dominant conditions, # of agents)
    results = model.datacollector.get_model_vars_dataframe()

    # get the last year
    end = results.iloc[-1]

    # if it's < 6000, return a high number (means roe deer were too high)
    if end["Time"] < 5999:
        return 10000


    else:
        # if it ran for 500 years, check the last 100 years and calculate gradient - has it reached equilibrium?
        last_hundred = results.loc[results["Time"] >= 4800]
        # only species I care about - ones that aren't controlled
        uncontrolled_only = last_hundred[['Grassland', 'Woodland', 'Thorny Scrub', 'Roe deer']]

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
        if abs(gradients["Roe deer"]) > 0.01 or abs(gradients["Thorny Scrub"]) > 0.01 or abs(gradients["Grassland"]) > 0.01 or abs(gradients["Woodland"]) > 0.01:
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
   


def run_optimizer():
    # Define the bounds
    bds = np.array([
        [0,0.2],[0,1],[0,1],[0.0017,0.0083],[0.0024,0.028],
        [0,1], # mature scrub competition
        [0,1],[0,1], # grass competition
        # roe parameters
        [0.1,0.75],[0,1*0.25],[0,0.33*0.25],[0,0.33*0.25],[0,0.1*0.25],[0,0.1*0.25],
        # fallow deer parameters
        [0.3,0.75],[0,1*0.25],[0,0.33*0.25],[0,0.33*0.25],[0,0.1*0.25],[0,0.1*0.25],
        # red deer
        [0.3,0.75],[0,1*0.25],[0,0.33*0.25],[0,0.33*0.25],[0,0.1*0.25],[0,0.1*0.25],
        # exmoor pony parameters
                [0,1*0.25],[0,0.33*0.25],[0,0.33*0.25],[0,0.1*0.25],[0,0.1*0.25],
        # cattle parameters
        [0,0.5],[0,1*0.25],[0,0.33*0.25],[0,0.33*0.25],[0,0.1*0.25],[0,0.1*0.25],
        # pig parameters
        [0,0.75],[0,1*0.25],[0,0.33*0.25],[0,0.33*0.25],[0,0.1*0.25],[0,0.1*0.25],
        # sapling protection parameter
        [0,1]])


    # popsize and maxiter are defined at the top of the page, was 10x100
    algorithm_param = {'max_num_iteration':10,
                    'population_size':50,\
                    'mutation_probability':0.1,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.5,\
                    'parents_portion': 0.3,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv': 2}


    optimization =  ga(function = objectiveFunction, dimension = 44, variable_type = 'real',variable_boundaries= bds, algorithm_parameters = algorithm_param, function_timeout=6000)
    optimization.run()
    outputs = list(optimization.output_dict["variable"]) + [(optimization.output_dict["function"])]
    # return excel with rows = output values and number of filters passed
    pd.DataFrame(outputs).to_excel('optim_outputs_105.xlsx', header=False, index=False)
    return optimization.output_dict


run_optimizer()









def graph_results():
    # output_parameters = run_optimizer()

    # define the parameters
    initial_roe = 12

    chance_reproduceSapling = output_parameters["variable"][0]
    chance_reproduceYoungScrub = output_parameters["variable"][1]
    chance_regrowGrass =output_parameters["variable"][2]
    chance_saplingBecomingTree =output_parameters["variable"][3]
    chance_youngScrubMatures =output_parameters["variable"][4]
    chance_scrubOutcompetedByTree =output_parameters["variable"][5]
    chance_grassOutcompetedByTree =output_parameters["variable"][6]
    chance_grassOutcompetedByScrub =output_parameters["variable"][7]

    # consumer values
    roe_deer_reproduce =output_parameters["variable"][8]
    roe_deer_gain_from_grass =output_parameters["variable"][9]
    roe_deer_gain_from_trees =output_parameters["variable"][10]
    roe_deer_gain_from_scrub = output_parameters["variable"][11]
    roe_deer_gain_from_saplings = output_parameters["variable"][12]
    roe_deer_gain_from_young_scrub = output_parameters["variable"][13]
    fallow_deer_reproduce = output_parameters["variable"][14]
    fallow_deer_gain_from_grass = output_parameters["variable"][15]
    fallow_deer_gain_from_trees = output_parameters["variable"][16]
    fallow_deer_gain_from_scrub = output_parameters["variable"][17]
    fallow_deer_gain_from_saplings = output_parameters["variable"][18]
    fallow_deer_gain_from_young_scrub = output_parameters["variable"][19]
    red_deer_reproduce = output_parameters["variable"][20]
    red_deer_gain_from_grass = output_parameters["variable"][21]
    red_deer_gain_from_trees = output_parameters["variable"][22]
    red_deer_gain_from_scrub = output_parameters["variable"][23]
    red_deer_gain_from_saplings = output_parameters["variable"][24]
    red_deer_gain_from_young_scrub = output_parameters["variable"][25]
    ponies_gain_from_grass = output_parameters["variable"][26]
    ponies_gain_from_trees = output_parameters["variable"][27]
    ponies_gain_from_scrub = output_parameters["variable"][28]
    ponies_gain_from_saplings = output_parameters["variable"][29]
    ponies_gain_from_young_scrub = output_parameters["variable"][30]
    cattle_reproduce = output_parameters["variable"][31]
    cows_gain_from_grass = output_parameters["variable"][32]
    cows_gain_from_trees = output_parameters["variable"][33]
    cows_gain_from_scrub = output_parameters["variable"][34]
    cows_gain_from_saplings = output_parameters["variable"][35]
    cows_gain_from_young_scrub = output_parameters["variable"][36]
    tamworth_pig_reproduce = output_parameters["variable"][37]
    tamworth_pig_gain_from_grass =output_parameters["variable"][38]
    tamworth_pig_gain_from_trees = output_parameters["variable"][39]
    tamworth_pig_gain_from_scrub = output_parameters["variable"][40]
    tamworth_pig_gain_from_saplings =output_parameters["variable"][41]
    tamworth_pig_gain_from_young_scrub = output_parameters["variable"][42]
    chance_scrub_saves_saplings = output_parameters["variable"][43]
    # stocking values
    fallowDeer_stocking = 247
    cattle_stocking = 81
    redDeer_stocking = 35
    tamworthPig_stocking = 7
    exmoor_stocking = 15
    initial_wood = 0.058
    initial_grass = 0.899
    initial_scrub = 0.043

    # euro bison parameters
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
    # forecasting
    fallowDeer_stocking_forecast = 86
    cattle_stocking_forecast = 76
    redDeer_stocking_forecast = 21
    tamworthPig_stocking_forecast = 7
    exmoor_stocking_forecast = 15
    roeDeer_stocking_forecast = 12
    introduced_species_stocking_forecast = 0

    exp_chance_reproduceSapling = 0
    exp_chance_reproduceYoungScrub =  0
    exp_chance_regrowGrass = 0
    duration = 0
    tree_reduction = 0

    
    random.seed(1)
    np.random.seed(1)
    
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


    model.reset_randomizer(seed=1)

    model.run_model()


    # first graph:  does it pass the filters? looking at the number of individual trees, etc.
    final_results = model.datacollector.get_model_vars_dataframe()

    # how many years did it run? 
    end = final_results.iloc[-1]
    end_year = end[["Time"]].item() + 2

    # does it pass the filters - conditions?
    y_values_conditions =  final_results[["Roe deer", "Exmoor pony", "Fallow deer", "Longhorn cattle", "Red deer", "Tamworth pigs", "Grassland", "Woodland", "Thorny Scrub", "Bare ground"]].values.flatten()            
    # indices
    indices_conditions = np.repeat(final_results['Time'], 10)
    # species list. this should be +1 the number of simulations
    species_list_conditions = np.tile(["Roe deer", "Exmoor pony", "Fallow deer", "Longhorn cattle", "Red deer", "Tamworth pigs", "Grassland", "Woodland", "Thorny Scrub", "Bare ground"], end_year) 

    final_df_condit = pd.DataFrame(
    {'Abundance': y_values_conditions, 'Ecosystem Element': species_list_conditions, 'Time': indices_conditions})

    colors = ["#6788ee"]
    # first graph: counterfactual & forecasting
    f = sns.FacetGrid(final_df_condit, col="Ecosystem Element", palette = colors, col_wrap=4, sharey = False)
    f.map(sns.lineplot, 'Time', 'Abundance')

    axes = f.axes.flatten()
    # fill between the quantiles
    axes = f.axes.flatten()
    axes[0].set_title("Roe deer")
    axes[1].set_title("Exmoor pony")
    axes[2].set_title("Fallow deer")
    axes[3].set_title("Longhorn cattle")
    axes[4].set_title("Red deer")
    axes[5].set_title("Tamworth pigs")
    axes[6].set_title("Grassland")
    axes[7].set_title("Woodland")
    axes[8].set_title("Thorny scrub")
    axes[9].set_title("Bare ground")
    # add filter lines
    f.axes[0].vlines(x=50,ymin=6,ymax=40, color='r')
    f.axes[6].vlines(x=50,ymin=16.8,ymax=89.9, color='r')
    f.axes[7].vlines(x=50,ymin=5.8,ymax=17, color='r')
    f.axes[8].vlines(x=50,ymin=4.3,ymax=51.8, color='r')
    f.axes[1].vlines(x=123,ymin=9,ymax=11, color='r')
    f.axes[3].vlines(x=123,ymin=90,ymax=140, color='r')
    f.axes[5].vlines(x=123,ymin=12,ymax=32, color='r')
    # May 2015
    f.axes[3].vlines(x=124,ymin=104,ymax=154, color='r')
    f.axes[5].vlines(x=124,ymin=4,ymax=24, color='r')
    f.axes[1].vlines(x=124,ymin=9,ymax=11, color='r')
    # June 2015
    f.axes[3].vlines(x=125,ymin=104,ymax=154, color='r')
    f.axes[1].vlines(x=125,ymin=9,ymax=11, color='r')
    f.axes[5].vlines(x=125,ymin=4,ymax=24, color='r')
    # July 2015
    f.axes[3].vlines(x=126,ymin=104,ymax=154, color='r')
    f.axes[1].vlines(x=126,ymin=9,ymax=11, color='r')
    f.axes[5].vlines(x=126,ymin=4,ymax=24, color='r')
    # Aug 2015
    f.axes[3].vlines(x=127,ymin=104,ymax=154, color='r')
    f.axes[1].vlines(x=127,ymin=9,ymax=11, color='r')
    f.axes[5].vlines(x=127,ymin=4,ymax=24, color='r')
    # Sept 2015
    f.axes[3].vlines(x=128,ymin=105,ymax=155, color='r')
    f.axes[1].vlines(x=128,ymin=9,ymax=11, color='r')
    f.axes[5].vlines(x=128,ymin=4,ymax=24, color='r')
    # Oct 2015
    f.axes[3].vlines(x=129,ymin=66,ymax=116, color='r')
    f.axes[1].vlines(x=129,ymin=9,ymax=11, color='r')
    f.axes[5].vlines(x=129,ymin=4,ymax=24, color='r')
    # Nov 2015
    f.axes[3].vlines(x=130,ymin=66,ymax=116, color='r')
    f.axes[1].vlines(x=130,ymin=9,ymax=11, color='r')
    f.axes[5].vlines(x=130,ymin=3,ymax=23, color='r')
    # Dec 2015
    f.axes[3].vlines(x=131,ymin=61,ymax=111, color='r')
    f.axes[1].vlines(x=131,ymin=9,ymax=11, color='r')
    f.axes[5].vlines(x=131,ymin=3,ymax=23, color='r')
    # Jan 2016
    f.axes[3].vlines(x=132,ymin=61,ymax=111, color='r')
    f.axes[1].vlines(x=132,ymin=9,ymax=11, color='r')
    f.axes[5].vlines(x=132,ymin=1,ymax=20, color='r')
    # Feb 2016
    f.axes[1].vlines(x=133,ymin=9,ymax=11, color='r')
    f.axes[3].vlines(x=133,ymin=61,ymax=111, color='r')
    f.axes[5].vlines(x=133,ymin=1,ymax=20, color='r')
    # March 2016
    f.axes[1].vlines(x=134,ymin=10,ymax=12, color='r')
    f.axes[3].vlines(x=134,ymin=61,ymax=111, color='r')
    f.axes[2].vlines(x=134,ymin=90,ymax=190, color='r')
    f.axes[4].vlines(x=134,ymin=21,ymax=31, color='r')
    f.axes[5].vlines(x=134,ymin=1,ymax=19, color='r')
    # April 2016
    f.axes[1].vlines(x=135,ymin=10,ymax=12, color='r')
    f.axes[3].vlines(x=135,ymin=78,ymax=128, color='r')
    f.axes[5].vlines(x=135,ymin=1,ymax=19, color='r')
    # May 2016
    f.axes[1].vlines(x=136,ymin=10,ymax=12, color='r')
    f.axes[3].vlines(x=136,ymin=83,ymax=133, color='r')
    f.axes[5].vlines(x=136,ymin=7,ymax=27, color='r')
    # June 2016
    f.axes[1].vlines(x=137,ymin=10,ymax=12, color='r')
    f.axes[3].vlines(x=137,ymin=64,ymax=114, color='r')
    f.axes[5].vlines(x=137,ymin=7,ymax=27, color='r')
    # July 2016
    f.axes[1].vlines(x=138,ymin=10,ymax=12, color='r')
    f.axes[3].vlines(x=138,ymin=62,ymax=112, color='r')
    f.axes[5].vlines(x=138,ymin=7,ymax=27, color='r')
    # Aug 2016
    f.axes[1].vlines(x=139,ymin=10,ymax=12, color='r')
    f.axes[3].vlines(x=139,ymin=62,ymax=112, color='r')
    f.axes[5].vlines(x=139,ymin=7,ymax=27, color='r')
    # Sept 2016
    f.axes[1].vlines(x=140,ymin=10,ymax=12, color='r')
    f.axes[3].vlines(x=140,ymin=72,ymax=122, color='r')
    f.axes[5].vlines(x=140,ymin=7,ymax=27, color='r')
    # Oct 2016
    f.axes[1].vlines(x=141,ymin=10,ymax=12, color='r')
    f.axes[3].vlines(x=141,ymin=72,ymax=122, color='r')
    f.axes[5].vlines(x=141,ymin=7,ymax=27, color='r')
    # Nov 2016
    f.axes[1].vlines(x=142,ymin=10,ymax=12, color='r')
    f.axes[3].vlines(x=142,ymin=67,ymax=117, color='r')
    f.axes[5].vlines(x=142,ymin=7,ymax=27, color='r')
    # Dec 2016
    f.axes[1].vlines(x=143,ymin=10,ymax=12, color='r')
    f.axes[3].vlines(x=143,ymin=54,ymax=104, color='r')
    f.axes[5].vlines(x=143,ymin=3,ymax=23, color='r')
    # Jan 2017
    f.axes[1].vlines(x=144,ymin=10,ymax=12, color='r')
    f.axes[3].vlines(x=144,ymin=54,ymax=104, color='r')
    f.axes[5].vlines(x=144,ymin=1,ymax=19, color='r')
    # Feb 2017
    f.axes[1].vlines(x=145,ymin=10,ymax=12, color='r')
    f.axes[3].vlines(x=145,ymin=54,ymax=104, color='r')
    f.axes[5].vlines(x=145,ymin=1,ymax=17, color='r')
    # March 2017
    f.axes[1].vlines(x=146,ymin=9,ymax=11, color='r')
    f.axes[2].vlines(x=146,ymin=115,ymax=200, color='r')
    f.axes[3].vlines(x=146,ymin=54,ymax=104, color='r')
    f.axes[5].vlines(x=146,ymin=1,ymax=17, color='r')
    # April 2017
    f.axes[1].vlines(x=147,ymin=9,ymax=11, color='r')
    f.axes[3].vlines(x=147,ymin=75,ymax=125, color='r')
    f.axes[5].vlines(x=147,ymin=12,ymax=32, color='r')
    # May 2017
    f.axes[1].vlines(x=148,ymin=9,ymax=11, color='r')
    f.axes[3].vlines(x=148,ymin=84,ymax=134, color='r')
    f.axes[5].vlines(x=148,ymin=12,ymax=32, color='r')
    # June 2017
    f.axes[1].vlines(x=149,ymin=9,ymax=11, color='r')
    f.axes[3].vlines(x=149,ymin=69,ymax=119, color='r')
    f.axes[5].vlines(x=149,ymin=12,ymax=32, color='r')
    # July 2017
    f.axes[1].vlines(x=150,ymin=9,ymax=11, color='r')
    f.axes[3].vlines(x=150,ymin=69,ymax=119, color='r')
    f.axes[5].vlines(x=150,ymin=12,ymax=32, color='r')
    # Aug 2017
    f.axes[1].vlines(x=151,ymin=9,ymax=11, color='r')
    f.axes[3].vlines(x=151,ymin=69,ymax=119, color='r')
    f.axes[5].vlines(x=151,ymin=12,ymax=32, color='r')
    # Sept 2017
    f.axes[1].vlines(x=152,ymin=9,ymax=11, color='r')
    f.axes[3].vlines(x=152,ymin=65,ymax=115, color='r')
    f.axes[5].vlines(x=152,ymin=20,ymax=24, color='r')
    # Oct 2017
    f.axes[1].vlines(x=153,ymin=9,ymax=11, color='r')
    f.axes[3].vlines(x=153,ymin=63,ymax=113, color='r')
    f.axes[5].vlines(x=153,ymin=12,ymax=32, color='r')
    # Nov 2017
    f.axes[1].vlines(x=154,ymin=9,ymax=11, color='r')
    f.axes[3].vlines(x=154,ymin=63,ymax=113, color='r')
    f.axes[5].vlines(x=154,ymin=12,ymax=32, color='r')
    # Dec 2017
    f.axes[1].vlines(x=155,ymin=9,ymax=11, color='r')
    f.axes[3].vlines(x=155,ymin=63,ymax=113, color='r')
    f.axes[5].vlines(x=155,ymin=8,ymax=28, color='r')
    # Jan 2018
    f.axes[1].vlines(x=156,ymin=9,ymax=11, color='r')
    f.axes[3].vlines(x=156,ymin=63,ymax=113, color='r')
    f.axes[5].vlines(x=156,ymin=1,ymax=21, color='r')
    # Feb 2018
    f.axes[1].vlines(x=157,ymin=9,ymax=11, color='r')
    f.axes[3].vlines(x=157,ymin=63,ymax=113, color='r')
    f.axes[5].vlines(x=157,ymin=6,ymax=26, color='r')
    # March 2018
    f.axes[1].vlines(x=158,ymin=8,ymax=10, color='r')
    f.axes[3].vlines(x=158,ymin=63,ymax=113, color='r')
    f.axes[4].vlines(x=158,ymin=19,ymax=29, color='r')
    f.axes[5].vlines(x=158,ymin=6,ymax=26, color='r')
    # April 2018
    f.axes[1].vlines(x=159,ymin=8,ymax=10, color='r')
    f.axes[3].vlines(x=159,ymin=76,ymax=126, color='r')
    f.axes[5].vlines(x=159,ymin=6,ymax=26, color='r')
    # May 2018
    f.axes[1].vlines(x=160,ymin=8,ymax=10, color='r')
    f.axes[3].vlines(x=160,ymin=92,ymax=142, color='r')
    f.axes[5].vlines(x=160,ymin=13,ymax=33, color='r')
    # June 2018
    f.axes[1].vlines(x=161,ymin=8,ymax=10, color='r')
    f.axes[3].vlines(x=161,ymin=78,ymax=128, color='r')
    f.axes[5].vlines(x=161,ymin=13,ymax=33, color='r')
    # July 2018
    f.axes[1].vlines(x=162,ymin=8,ymax=10, color='r')
    f.axes[3].vlines(x=162,ymin=78,ymax=128, color='r')
    f.axes[5].vlines(x=162,ymin=12,ymax=32, color='r')
    # Aug 2018
    f.axes[3].vlines(x=163,ymin=77,ymax=127, color='r')
    f.axes[5].vlines(x=163,ymin=12,ymax=32, color='r')
    # Sept 2018
    f.axes[3].vlines(x=164,ymin=81,ymax=131, color='r')
    f.axes[5].vlines(x=164,ymin=12,ymax=32, color='r')
    # Oct 2018
    f.axes[3].vlines(x=165,ymin=76,ymax=126, color='r')
    f.axes[5].vlines(x=165,ymin=11,ymax=31, color='r')
    # Nov 2018
    f.axes[3].vlines(x=166,ymin=68,ymax=118, color='r')
    f.axes[5].vlines(x=166,ymin=1,ymax=19, color='r')
    # Dec 2018
    f.axes[3].vlines(x=167,ymin=64,ymax=114, color='r')
    f.axes[5].vlines(x=167,ymin=1,ymax=19, color='r')
    # Jan 2019
    f.axes[3].vlines(x=168,ymin=64,ymax=114, color='r')
    f.axes[5].vlines(x=168,ymin=1,ymax=19, color='r')
    # Feb 2019
    f.axes[3].vlines(x=169,ymin=62,ymax=112, color='r')
    f.axes[5].vlines(x=169,ymin=1,ymax=20, color='r')
    # March 2019
    f.axes[2].vlines(x=170,ymin=253,ymax=303, color='r')
    f.axes[3].vlines(x=170,ymin=62,ymax=112, color='r')
    f.axes[4].vlines(x=170,ymin=32,ymax=42, color='r')
    f.axes[5].vlines(x=170,ymin=1,ymax=19, color='r')
    # April 2019
    f.axes[3].vlines(x=171,ymin=76,ymax=126, color='r')
    f.axes[5].vlines(x=171,ymin=1,ymax=18, color='r')
    # May 2019
    f.axes[3].vlines(x=172,ymin=85,ymax=135, color='r')
    f.axes[5].vlines(x=172,ymin=1,ymax=18, color='r')
    # June 2019
    f.axes[3].vlines(x=173,ymin=64,ymax=114, color='r')
    f.axes[5].vlines(x=173,ymin=1,ymax=18, color='r')
    # July 2019
    f.axes[3].vlines(x=174,ymin=66,ymax=116, color='r')
    f.axes[5].vlines(x=174,ymin=1,ymax=19, color='r')
    # Aug 2019
    f.axes[3].vlines(x=175,ymin=66,ymax=116, color='r')
    f.axes[5].vlines(x=175,ymin=1,ymax=19, color='r')  
    # Sept 2019
    f.axes[3].vlines(x=176,ymin=68,ymax=118, color='r')
    f.axes[5].vlines(x=176,ymin=1,ymax=19, color='r')
    # Oct 2019
    f.axes[3].vlines(x=177,ymin=63,ymax=113, color='r')
    f.axes[5].vlines(x=177,ymin=1,ymax=19, color='r')
    # Nov 2019
    f.axes[3].vlines(x=178,ymin=62,ymax=112, color='r')
    f.axes[5].vlines(x=178,ymin=1,ymax=19, color='r')
    # Dec 2019
    f.axes[3].vlines(x=179,ymin=55,ymax=105, color='r')
    f.axes[5].vlines(x=179,ymin=1,ymax=20, color='r')
    # Jan 2020
    f.axes[3].vlines(x=180,ymin=55,ymax=105, color='r')
    f.axes[5].vlines(x=180,ymin=1,ymax=20, color='r')
    # Feb 2020
    f.axes[3].vlines(x=181,ymin=54,ymax=104, color='r')
    f.axes[5].vlines(x=181,ymin=1,ymax=18, color='r')
    # March 2020
    f.axes[2].vlines(x=182,ymin=222,ymax=272, color='r')
    f.axes[4].vlines(x=182,ymin=30,ymax=40, color='r')
    f.axes[3].vlines(x=182,ymin=56,ymax=106, color='r')
    f.axes[5].vlines(x=182,ymin=1,ymax=17, color='r')
    # April 2020
    f.axes[1].vlines(x=183,ymin=14,ymax=17, color='r')
    f.axes[3].vlines(x=183,ymin=56,ymax=106, color='r')
    f.axes[5].vlines(x=183,ymin=1,ymax=17, color='r')
    # plot next set of filter lines
    f.axes[0].vlines(x=184,ymin=20,ymax=80, color='r')
    f.axes[1].vlines(x=184,ymin=14,ymax=17, color='r')
    f.axes[3].vlines(x=184,ymin=56,ymax=106, color='r')
    f.axes[5].vlines(x=184,ymin=9,ymax=29, color='r')
    f.axes[6].vlines(x=184,ymin=16.8,ymax=36.8, color='r')
    f.axes[7].vlines(x=184,ymin=16.5,ymax=26.5, color='r')
    f.axes[8].vlines(x=184,ymin=41.8,ymax=61.8, color='r')
    # stop the plots from overlapping
    f.fig.suptitle("Optimizer Outputs")
    plt.tight_layout()
    plt.savefig('optimizer_outputs_conditions.png')
    plt.show()


# graph_results()