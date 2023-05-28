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

    # pick a few starting conditions - we want the first run to be initial conditions
    initial_stocking = [1]
    stocking_changes = random.choices([1, 0, 0.5, 2, 5], k=100)
    stocking_changes_list = initial_stocking + stocking_changes

    # now do initial vegetation values
    initial_wood_list = [0.058]
    initial_grass_list = [0.899]
    initial_scrub_list = [0.043]
    # add the rest
    for i in range(100):
        wood = random.uniform(0, 1)
        grass = random.uniform(0, 1)
        scrub = random.uniform(0, 1)
        initial_veg = [wood, grass, scrub]
        # normalize them if they're greater than 100%
        if (wood+grass+scrub) > 1:
            norm = [float(i)/sum(initial_veg) for i in initial_veg]
            wood = norm[0]
            grass = norm[1]
            scrub = norm[2]
        # append to list
        initial_wood_list.append(wood)
        initial_grass_list.append(grass)
        initial_scrub_list.append(scrub)


    # for each starting condition, run the model from 2005 and keep track of final conditions
    run_number = 0
    final_df = {}


    # take the accepted parameters, and go row by row, running the model
    for i, w, g, s in zip(stocking_changes_list, initial_wood_list, initial_grass_list, initial_scrub_list):
        
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
        initial_wood = w
        initial_grass = g
        initial_scrub = s

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


        # experiment parameters
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

        # we only want to look at the habitat types
        habs_only = results[['Grassland', 'Woodland', 'Thorny Scrub', 'Bare ground', 'Roe deer']]
        initial = habs_only.iloc[0]
        end = habs_only.iloc[-1]

        # collect their conditions
        final_df[run_number] = {"Grassland Initial": initial['Grassland'].item(), "Grassland End": end['Grassland'].item(), 
                                "Woodland Initial": initial['Woodland'].item(), "Woodland End": end['Woodland'].item(), 
                                "Thorny Scrub Initial": initial['Thorny Scrub'].item(), "Thorny Scrub End": end['Thorny Scrub'].item(), 
                                "Stocking Density": i,
                                "Run Number": run_number
        }


    # append to dataframe
    forecasting = pd.DataFrame.from_dict(final_df, "index")

    print(forecasting)

    forecasting.to_csv("resilience_landscape.csv")


check_resilience()
