# choose the best parameter set for the ABM

from model import KneppModel 
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def choose_parameters():

    # open the choice of parameters
    accepted_parameters = pd.read_csv('param_options_abm.csv').drop("Unnamed: 0", axis=1)

    # run forecasting
    final_results_list = pd.DataFrame(columns=accepted_parameters.columns)
    run_number = 0

    # take the accepted parameters, and go row by row, running the model
    for _, row in accepted_parameters.iterrows():

        chance_reproduceSapling = row["chance_reproduceSapling"]
        chance_reproduceYoungScrub =  row["chance_reproduceYoungScrub"]
        chance_regrowGrass =  row["chance_regrowGrass"]
        chance_saplingBecomingTree =  row["chance_saplingBecomingTree"]
        chance_youngScrubMatures =  row["chance_youngScrubMatures"]
        chance_scrubOutcompetedByTree =  row["chance_scrubOutcompetedByTree"]
        chance_grassOutcompetedByTree =  row["chance_grassOutcompetedByTree"]
        chance_grassOutcompetedByScrub = row["chance_grassOutcompetedByScrub"]

        # set initial conditions
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

        # reindeer parameters
        reindeer_reproduce = 0
        # reindeer should have impacts between red and fallow deer
        reindeer_gain_from_grass = 0
        reindeer_gain_from_trees =0
        reindeer_gain_from_scrub =0
        reindeer_gain_from_saplings = 0
        reindeer_gain_from_young_scrub = 0
        # forecasting parameters - REMEMBER TO CHANGE THESE
        fallowDeer_stocking_forecast = 247
        cattle_stocking_forecast = 81
        redDeer_stocking_forecast = 35
        tamworthPig_stocking_forecast = 7
        exmoor_stocking_forecast = 15
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

        run_number +=1
        print(run_number)

        results = model.datacollector.get_model_vars_dataframe()
        results['run_number'] = row["run_number"]

        # look at the last year 
        roe_only = results[['Roe deer']]
        # if roe deer is over 500, discard it
        end = roe_only.iloc[-1]
        print("last roe deer value", end["Roe deer"])

        if end["Roe deer"] < 500:

            # check equilibrium
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
            
            print(gradients)

            # if it's not at equilibrium, fail it
            if abs(gradients["Roe deer"]) < 0.05 or abs(gradients["Thorny Scrub"]) < 0.05 or abs(gradients["Grassland"]) < 0.05 or abs(gradients["Woodland"]) < 0.05:
                # remember the parameters 
                final_results_list = final_results_list.append(row)
                print(final_results_list)

    # final df
    print(final_results_list)
    final_results_list.to_csv("param_options_abm_both_filters.csv")



choose_parameters()