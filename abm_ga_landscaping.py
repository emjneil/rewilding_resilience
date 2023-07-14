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

# alter one species at a time to get to 33% of each habitat type
# save timeseries



def objectiveFunction(x):

    # open the best parameter set
    best_ps = pd.read_excel("optimizer_outputs_resilience.xlsx",  engine="openpyxl")
    best_ps = best_ps.loc[best_ps["run_number"] == 1]

    exmoor_stocking_change = x[0]
    fallow_stocking_change = x[1]
    cattle_stocking_change = x[2]
    red_stocking_change = x[3]
    pig_stocking_change = x[4]

    # define the parameters
    chance_reproduceSapling = best_ps["chance_reproduceSapling"].item()
    chance_reproduceYoungScrub =  best_ps["chance_reproduceYoungScrub"].item()
    chance_regrowGrass =  best_ps["chance_regrowGrass"].item()
    chance_saplingBecomingTree =  best_ps["chance_saplingBecomingTree"].item()
    chance_youngScrubMatures =  best_ps["chance_youngScrubMatures"].item()
    chance_scrubOutcompetedByTree =  best_ps["chance_scrubOutcompetedByTree"].item()
    chance_grassOutcompetedByTree =  best_ps["chance_grassOutcompetedByTree"].item()
    chance_grassOutcompetedByScrub = best_ps["chance_grassOutcompetedByScrub"].item()

    initial_roe = 12
    fallowDeer_stocking = 247
    cattle_stocking = 81
    redDeer_stocking = 35
    tamworthPig_stocking = 7
    exmoor_stocking = 15
    initial_wood = 0.058
    initial_grass = 0.899
    initial_scrub = 0.043

    roe_deer_reproduce = best_ps["roe_deer_reproduce"].item()
    roe_deer_gain_from_grass =  best_ps["roe_deer_gain_from_grass"].item()
    roe_deer_gain_from_trees =  best_ps["roe_deer_gain_from_trees"].item()
    roe_deer_gain_from_scrub =  best_ps["roe_deer_gain_from_scrub"].item()
    roe_deer_gain_from_saplings =  best_ps["roe_deer_gain_from_saplings"].item()
    roe_deer_gain_from_young_scrub =  best_ps["roe_deer_gain_from_young_scrub"].item()
    ponies_gain_from_grass =  best_ps["ponies_gain_from_grass"].item()
    ponies_gain_from_trees =  best_ps["ponies_gain_from_trees"].item()
    ponies_gain_from_scrub =  best_ps["ponies_gain_from_scrub"].item()
    ponies_gain_from_saplings =  best_ps["ponies_gain_from_saplings"].item()
    ponies_gain_from_young_scrub =  best_ps["ponies_gain_from_young_scrub"].item()
    cattle_reproduce =  best_ps["cattle_reproduce"].item()
    cows_gain_from_grass =  best_ps["cows_gain_from_grass"].item()
    cows_gain_from_trees =  best_ps["cows_gain_from_trees"].item()
    cows_gain_from_scrub =  best_ps["cows_gain_from_scrub"].item()
    cows_gain_from_saplings =  best_ps["cows_gain_from_saplings"].item()
    cows_gain_from_young_scrub =  best_ps["cows_gain_from_young_scrub"].item()
    fallow_deer_reproduce =  best_ps["fallow_deer_reproduce"].item()
    fallow_deer_gain_from_grass =  best_ps["fallow_deer_gain_from_grass"].item()
    fallow_deer_gain_from_trees =  best_ps["fallow_deer_gain_from_trees"].item()
    fallow_deer_gain_from_scrub =  best_ps["fallow_deer_gain_from_scrub"].item()
    fallow_deer_gain_from_saplings =  best_ps["fallow_deer_gain_from_saplings"].item()
    fallow_deer_gain_from_young_scrub =  best_ps["fallow_deer_gain_from_young_scrub"].item()  
    red_deer_reproduce =  best_ps["red_deer_reproduce"].item()
    red_deer_gain_from_grass =  best_ps["red_deer_gain_from_grass"].item()
    red_deer_gain_from_trees =  best_ps["red_deer_gain_from_trees"].item()
    red_deer_gain_from_scrub =  best_ps["red_deer_gain_from_scrub"].item()
    red_deer_gain_from_saplings =  best_ps["red_deer_gain_from_saplings"].item()
    red_deer_gain_from_young_scrub =  best_ps["red_deer_gain_from_young_scrub"].item()
    tamworth_pig_reproduce =  best_ps["tamworth_pig_reproduce"].item()
    tamworth_pig_gain_from_grass =  best_ps["tamworth_pig_gain_from_grass"].item()
    tamworth_pig_gain_from_trees = best_ps["tamworth_pig_gain_from_trees"].item()
    tamworth_pig_gain_from_scrub = best_ps["tamworth_pig_gain_from_scrub"].item()
    tamworth_pig_gain_from_saplings =  best_ps["tamworth_pig_gain_from_saplings"].item()
    tamworth_pig_gain_from_young_scrub =  best_ps["tamworth_pig_gain_from_young_scrub"].item()

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
    # forecasting parameters
    fallowDeer_stocking_forecast = int(247 * fallow_stocking_change)
    cattle_stocking_forecast = int(81 * cattle_stocking_change)
    redDeer_stocking_forecast = int(35 * red_stocking_change)
    tamworthPig_stocking_forecast = int(7 * pig_stocking_change)
    exmoor_stocking_forecast = int(15 * exmoor_stocking_change)
    introduced_species_stocking_forecast = 0
    chance_scrub_saves_saplings = best_ps["chance_scrub_saves_saplings"].item()

    # new resilience experiment parameters - not relevant here
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

    # want 33% of everything
    filtered_result = (
        # pre-reintro model
        ((((list(results.loc[results['Time'] == 5999, 'Grassland'])[0])-33)/33)**2) +
        ((((list(results.loc[results['Time'] == 5999, 'Thorny Scrub'])[0])-33)/33)**2) +
        ((((list(results.loc[results['Time'] == 5999, 'Woodland'])[0])-33)/33)**2)
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
    # Define the bounds for stocking densities
    bds = np.array([
        [0,10],[0,10],[0,10],[0,10],[0,10]]
        # [0,25]
        )


    # popsize and maxiter are defined at the top of the page, was 10x100
    algorithm_param = {'max_num_iteration':10,
                    'population_size':50,\
                    'mutation_probability':0.1,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.5,\
                    'parents_portion': 0.3,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv': 2}


    optimization =  ga(function = objectiveFunction, dimension = 5, variable_type = 'real',variable_boundaries= bds, algorithm_parameters = algorithm_param, function_timeout=6000)
    optimization.run()
    outputs = list(optimization.output_dict["variable"]) + [(optimization.output_dict["function"])]
    # return excel with rows = output values and number of filters passed
    pd.DataFrame(outputs).to_excel('abm_landscaping_ga.xlsx', header=False, index=False)
    return optimization.output_dict


run_optimizer()









def graph_results():

    stocking_changes = [30, 4.2, 11.9, 17.6, 25] 
    index_species = [fallowDeer_stocking_forecast, cattle_stocking_forecast, redDeer_stocking_forecast, tamworthPig_stocking_forecast, exmoor_stocking_forecast]

    final_parameters = pd.read_excel("optimizer_outputs_resilience.xlsx",  engine="openpyxl")
    final_parameters = final_parameters.loc[final_parameters["run_number"] <= 100]


    all_timeseries = []
    failed_roe = 0
    failed_equilibrium = 0

    for index, row in final_parameters.iterrows():

        for i, j in zip(stocking_changes, index_species):

            print(row["run_number"])

            chance_reproduceSapling = row["chance_reproduceSapling"]
            chance_reproduceYoungScrub =  row["chance_reproduceYoungScrub"]
            chance_regrowGrass =  row["chance_regrowGrass"]
            chance_saplingBecomingTree =  row["chance_saplingBecomingTree"]
            chance_youngScrubMatures =  row["chance_youngScrubMatures"]
            chance_scrubOutcompetedByTree =  row["chance_scrubOutcompetedByTree"]
            chance_grassOutcompetedByTree =  row["chance_grassOutcompetedByTree"]
            chance_grassOutcompetedByScrub = row["chance_grassOutcompetedByScrub"]

            initial_roe = 12
            fallowDeer_stocking = 247
            cattle_stocking = 81
            redDeer_stocking = 35
            tamworthPig_stocking = 7
            exmoor_stocking = 15
            initial_wood = 0.058
            initial_grass = 0.899
            initial_scrub = 0.043

            roe_deer_reproduce = row["roe_deer_reproduce"]
            roe_deer_gain_from_grass =  row["roe_deer_gain_from_grass"]
            roe_deer_gain_from_trees =  row["roe_deer_gain_from_trees"]
            roe_deer_gain_from_scrub =  row["roe_deer_gain_from_scrub"]
            roe_deer_gain_from_saplings =  row["roe_deer_gain_from_saplings"]
            roe_deer_gain_from_young_scrub =  row["roe_deer_gain_from_young_scrub"]
            ponies_gain_from_grass =  row["ponies_gain_from_grass"]
            ponies_gain_from_trees =  row["ponies_gain_from_trees"]
            ponies_gain_from_scrub =  row["ponies_gain_from_scrub"]
            ponies_gain_from_saplings =  row["ponies_gain_from_saplings"]
            ponies_gain_from_young_scrub =  row["ponies_gain_from_young_scrub"]
            cattle_reproduce =  row["cattle_reproduce"]
            cows_gain_from_grass =  row["cows_gain_from_grass"]
            cows_gain_from_trees =  row["cows_gain_from_trees"]
            cows_gain_from_scrub =  row["cows_gain_from_scrub"]
            cows_gain_from_saplings =  row["cows_gain_from_saplings"]
            cows_gain_from_young_scrub =  row["cows_gain_from_young_scrub"]
            fallow_deer_reproduce =  row["fallow_deer_reproduce"]
            fallow_deer_gain_from_grass =  row["fallow_deer_gain_from_grass"]
            fallow_deer_gain_from_trees =  row["fallow_deer_gain_from_trees"]
            fallow_deer_gain_from_scrub =  row["fallow_deer_gain_from_scrub"]
            fallow_deer_gain_from_saplings =  row["fallow_deer_gain_from_saplings"]
            fallow_deer_gain_from_young_scrub =  row["fallow_deer_gain_from_young_scrub"]   
            red_deer_reproduce =  row["red_deer_reproduce"]
            red_deer_gain_from_grass =  row["red_deer_gain_from_grass"]
            red_deer_gain_from_trees =  row["red_deer_gain_from_trees"]
            red_deer_gain_from_scrub =  row["red_deer_gain_from_scrub"]
            red_deer_gain_from_saplings =  row["red_deer_gain_from_saplings"]
            red_deer_gain_from_young_scrub =  row["red_deer_gain_from_young_scrub"]
            tamworth_pig_reproduce =  row["tamworth_pig_reproduce"]
            tamworth_pig_gain_from_grass =  row["tamworth_pig_gain_from_grass"]
            tamworth_pig_gain_from_trees = row["tamworth_pig_gain_from_trees"]
            tamworth_pig_gain_from_scrub = row["tamworth_pig_gain_from_scrub"]
            tamworth_pig_gain_from_saplings =  row["tamworth_pig_gain_from_saplings"]
            tamworth_pig_gain_from_young_scrub =  row["tamworth_pig_gain_from_young_scrub"]

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
            # forecasting parameters
            fallowDeer_stocking_forecast = 247
            cattle_stocking_forecast = 81
            redDeer_stocking_forecast = 35
            tamworthPig_stocking_forecast = 7
            exmoor_stocking_forecast = 15
            introduced_species_stocking_forecast = 0

            # reset whichever one we need
            j = int(j * i)

            chance_scrub_saves_saplings = row["chance_scrub_saves_saplings"]

            # new resilience experiment parameters - not relevant here
            exp_chance_reproduceSapling = 0
            exp_chance_reproduceYoungScrub =  0
            exp_chance_regrowGrass = 0
            duration = 0
            tree_reduction = 0


            random.seed(1)
            np.random.seed(1)


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
            results['run_number'] = row["run_number"]
            results['Stocking'] = i
            results["Species"] = species[j]

            all_timeseries.append(results)

            print(run_number)


    all_timeseries = pd.concat(all_timeseries)
    print(all_timeseries)
    all_timeseries.to_csv("abm_ga_timeseries_stockingchanges.csv")



# graph_results()











def all_varied_together():

    final_parameters = pd.read_excel("optimizer_outputs_resilience.xlsx",  engine="openpyxl")
    final_parameters = final_parameters.loc[final_parameters["run_number"] <= 100]


    all_timeseries = []
    failed_roe = 0
    failed_equilibrium = 0

    for index, row in final_parameters.iterrows():

        print(row["run_number"])

        chance_reproduceSapling = row["chance_reproduceSapling"]
        chance_reproduceYoungScrub =  row["chance_reproduceYoungScrub"]
        chance_regrowGrass =  row["chance_regrowGrass"]
        chance_saplingBecomingTree =  row["chance_saplingBecomingTree"]
        chance_youngScrubMatures =  row["chance_youngScrubMatures"]
        chance_scrubOutcompetedByTree =  row["chance_scrubOutcompetedByTree"]
        chance_grassOutcompetedByTree =  row["chance_grassOutcompetedByTree"]
        chance_grassOutcompetedByScrub = row["chance_grassOutcompetedByScrub"]

        initial_roe = 12
        fallowDeer_stocking = 247
        cattle_stocking = 81
        redDeer_stocking = 35
        tamworthPig_stocking = 7
        exmoor_stocking = 15
        initial_wood = 0.058
        initial_grass = 0.899
        initial_scrub = 0.043

        roe_deer_reproduce = row["roe_deer_reproduce"]
        roe_deer_gain_from_grass =  row["roe_deer_gain_from_grass"]
        roe_deer_gain_from_trees =  row["roe_deer_gain_from_trees"]
        roe_deer_gain_from_scrub =  row["roe_deer_gain_from_scrub"]
        roe_deer_gain_from_saplings =  row["roe_deer_gain_from_saplings"]
        roe_deer_gain_from_young_scrub =  row["roe_deer_gain_from_young_scrub"]
        ponies_gain_from_grass =  row["ponies_gain_from_grass"]
        ponies_gain_from_trees =  row["ponies_gain_from_trees"]
        ponies_gain_from_scrub =  row["ponies_gain_from_scrub"]
        ponies_gain_from_saplings =  row["ponies_gain_from_saplings"]
        ponies_gain_from_young_scrub =  row["ponies_gain_from_young_scrub"]
        cattle_reproduce =  row["cattle_reproduce"]
        cows_gain_from_grass =  row["cows_gain_from_grass"]
        cows_gain_from_trees =  row["cows_gain_from_trees"]
        cows_gain_from_scrub =  row["cows_gain_from_scrub"]
        cows_gain_from_saplings =  row["cows_gain_from_saplings"]
        cows_gain_from_young_scrub =  row["cows_gain_from_young_scrub"]
        fallow_deer_reproduce =  row["fallow_deer_reproduce"]
        fallow_deer_gain_from_grass =  row["fallow_deer_gain_from_grass"]
        fallow_deer_gain_from_trees =  row["fallow_deer_gain_from_trees"]
        fallow_deer_gain_from_scrub =  row["fallow_deer_gain_from_scrub"]
        fallow_deer_gain_from_saplings =  row["fallow_deer_gain_from_saplings"]
        fallow_deer_gain_from_young_scrub =  row["fallow_deer_gain_from_young_scrub"]   
        red_deer_reproduce =  row["red_deer_reproduce"]
        red_deer_gain_from_grass =  row["red_deer_gain_from_grass"]
        red_deer_gain_from_trees =  row["red_deer_gain_from_trees"]
        red_deer_gain_from_scrub =  row["red_deer_gain_from_scrub"]
        red_deer_gain_from_saplings =  row["red_deer_gain_from_saplings"]
        red_deer_gain_from_young_scrub =  row["red_deer_gain_from_young_scrub"]
        tamworth_pig_reproduce =  row["tamworth_pig_reproduce"]
        tamworth_pig_gain_from_grass =  row["tamworth_pig_gain_from_grass"]
        tamworth_pig_gain_from_trees = row["tamworth_pig_gain_from_trees"]
        tamworth_pig_gain_from_scrub = row["tamworth_pig_gain_from_scrub"]
        tamworth_pig_gain_from_saplings =  row["tamworth_pig_gain_from_saplings"]
        tamworth_pig_gain_from_young_scrub =  row["tamworth_pig_gain_from_young_scrub"]

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
        # forecasting parameters
        fallowDeer_stocking_forecast = int(247 * x)
        cattle_stocking_forecast = int(81 * x)
        redDeer_stocking_forecast = int(35 * x)
        tamworthPig_stocking_forecast =int(7 * x)
        exmoor_stocking_forecast = int(15 * x)
        introduced_species_stocking_forecast = 0
        chance_scrub_saves_saplings = row["chance_scrub_saves_saplings"]

        # new resilience experiment parameters - not relevant here
        exp_chance_reproduceSapling = 0
        exp_chance_reproduceYoungScrub =  0
        exp_chance_regrowGrass = 0
        duration = 0
        tree_reduction = 0


        random.seed(1)
        np.random.seed(1)


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
        results['run_number'] = row["run_number"]
        results['Stocking'] = 0
        # results["Species"] = "All varied together"
        # results["Species"] = "No control"
        results["Species"] = "Current dynamics"


        all_timeseries.append(results)

        print(run_number)


    all_timeseries = pd.concat(all_timeseries)
    print(all_timeseries)
    # all_timeseries.to_csv("abm_ga_timeseries_stockingchanges_alltogether.csv")    
    all_timeseries.to_csv("abm_ga_timeseries_stockingchanges_currentdynamics.csv")
    # all_timeseries.to_csv("abm_ga_timeseries_stockingchanges_nocontrol.csv")





# all_varied_together()