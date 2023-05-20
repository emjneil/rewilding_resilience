from model import KneppModel 
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# first function checks the existence of alternative stable states under current conditions
# second function compared to if rewilding hadn't happened (roe deer there, but no large herbivores)


def check_resilience():

    # pick the best parameter set
    best_parameter = pd.read_csv('best_parameter_set.csv').drop(['Unnamed: 0'], axis=1)

    # pick a few starting conditions 
    stocking_changes = [0, 0.5, 1, 2, 5]
    # different vegetation
    # initial_wood = [0.058, 0, 1, 0, 0, 0.33, 0.5, 0.3, 0.1]
    # initial_grass = [0.899, 0, 0, 1, 0, 0.33, 0.25, 0.1, 0.2]
    # initial_scrub = [0.043, 0, 0, 0, 1, 0.33, 0.25, 0.1, 0.1]

    initial_wood = [0.058, 0]
    initial_grass = [0.899, 0]
    initial_scrub = [0.043, 0]

    # for each starting condition, run the model from 2005 and keep track of final conditions
    final_results_list = []
    run_number = 0


    # take the accepted parameters, and go row by row, running the model
    for x, y, z in zip(initial_wood, initial_grass, initial_scrub):
        for i in stocking_changes:
            
            chance_reproduceSapling = best_parameter["chance_reproduceSapling"].item()
            chance_reproduceYoungScrub =  best_parameter["chance_reproduceYoungScrub"].item()
            chance_regrowGrass =  best_parameter["chance_regrowGrass"].item()
            chance_saplingBecomingTree =  best_parameter["chance_saplingBecomingTree"].item()
            chance_youngScrubMatures =  best_parameter["chance_youngScrubMatures"].item()
            chance_scrubOutcompetedByTree =  best_parameter["chance_scrubOutcompetedByTree"].item()
            chance_grassOutcompetedByTree =  best_parameter["chance_grassOutcompetedByTree"].item()
            chance_grassOutcompetedByScrub = best_parameter["chance_grassOutcompetedByScrub"].item()
            chance_scrub_saves_saplings = best_parameter["chance_scrub_saves_saplings"].item()

            initial_roe = int(12 * i)

            initial_wood = x
            initial_grass = y
            initial_scrub = z

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
            fallowDeer_stocking_forecast = int(247 * i)
            cattle_stocking_forecast = int(81 * i)
            redDeer_stocking_forecast = int(35 * i)
            tamworthPig_stocking_forecast = int(7 * i)
            exmoor_stocking_forecast = int(15 * i)

            # have it stop when max time is reached

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
                                max_time = 25, reintroduction = True, introduce_euroBison = False, introduce_elk = False)


            model.reset_randomizer(seed=1)

            model.run_model()

            run_number +=1
            print(run_number)

            results = model.datacollector.get_model_vars_dataframe()

            # we only want to look at the habitat types
            habs_only = results[['Grassland', 'Woodland', 'Thorny Scrub', 'Bare ground']]
            initial = habs_only.iloc[0]
            end = habs_only.iloc[-1]

            # collect their initial and end states and put into df
            final_df =  pd.DataFrame(
                        (initial.values.flatten()), columns=['Initial Condition'])
            final_df["End Condition"] = pd.DataFrame(end.values.flatten())
            final_df["Ecosystem Element"] = pd.DataFrame(["Grassland", "Woodland", "Thorny Scrub", "Bare ground"])
            final_df["Stocking Change"] = pd.DataFrame(np.concatenate([np.repeat(i,4)], axis=0))

            final_results_list.append(final_df)



    # append to dataframe
    forecasting = pd.concat(final_results_list).reset_index(drop=True)

    forecasting.to_csv("resilience_landscape.csv")


check_resilience()
