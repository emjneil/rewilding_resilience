from model import KneppModel 
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def experiment_growth():
    
    # pick the best parameter set
    best_parameter = pd.read_csv('abm_best_ps.csv').drop(['Unnamed: 0'], axis=1)
    final_results_list = []
    run_number = 0

    # make a list of my choices (what growth rates to change, and for how long)
    growth_changes_list = random.choices([0, 0.01, 0.1, 0.5, 1, 1.5], k=5)
    duration_choice_list = np.random.uniform(0,50,5)

    # take the accepted parameters, and go row by row, running the model
    for i,j in zip(growth_changes_list, duration_choice_list):

        print(i, int(j))
        
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
        redDeer_stocking_forecast = 35
        tamworthPig_stocking_forecast = 7
        exmoor_stocking_forecast = 15

        growth_changes = i

        # experiment parameters
        exp_chance_reproduceSapling = best_parameter["chance_reproduceSapling"].item() * growth_changes
        exp_chance_reproduceYoungScrub =  best_parameter["chance_reproduceYoungScrub"].item() * growth_changes
        exp_chance_regrowGrass =  best_parameter["chance_regrowGrass"].item() * growth_changes
        duration = int(j*12)

        tree_reduction = 0

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
                            exp_chance_reproduceSapling, exp_chance_reproduceYoungScrub, exp_chance_regrowGrass, duration, tree_reduction,
                            reintroduction = True, introduce_euroBison = False, introduce_elk = False, 
                            experiment_growth = True, experiment_wood = False, experiment_linear_growth = False, 
                            max_time = 6500, max_roe = 500
                            )


        model.reset_randomizer(seed=1)

        model.run_model()

        run_number +=1
        print(run_number)

        results = model.datacollector.get_model_vars_dataframe()


        # we only want to look at the habitat types        
        results['run_number'] = run_number
        results['growth'] = growth_changes
        results['duration'] = duration

        habs_only = results[['Roe deer', 'Grassland', 'Woodland', 'Thorny Scrub', 'Bare ground', 'Time', 'run_number', 'growth', 'duration']]
        
        final_results_list.append(habs_only)

    # append to dataframe
    experiment_growth = pd.concat(final_results_list)

    final_df =  pd.DataFrame(
                    (experiment_growth[['Roe deer', 'Grassland', 'Woodland', 'Thorny Scrub', 'Bare ground']].values.flatten()), columns=['Abundance'])
    final_df["Run Number"] = pd.DataFrame(np.concatenate([np.repeat(experiment_growth['run_number'],5)], axis=0))
    final_df["Time"] = pd.DataFrame(np.concatenate([np.repeat(experiment_growth['Time'], 5)], axis=0))
    final_df["Ecosystem Element"] = pd.DataFrame(np.tile(['Roe deer', "Grassland", "Woodland", "Thorny Scrub", "Bare ground"], len(experiment_growth)))
    final_df["Growth Rate Change"] = pd.DataFrame(np.concatenate([np.repeat(experiment_growth['growth'],5)], axis=0))
    final_df["Duration"] = pd.DataFrame(np.concatenate([np.repeat(experiment_growth['duration'],5)], axis=0))


    final_df.to_csv("experiment_growth_test.csv")


# experiment_growth()








def experiment_linear_growth():
    
    # pick the best parameter set
    best_parameter = pd.read_csv('abm_best_ps.csv').drop(['Unnamed: 0'], axis=1)
    # best_parameter = pd.read_csv('abm_mosaic_param.csv')

    final_results_list = []
    run_number = 0

    # is there a critical amount of noise?
    noise_amount = np.random.choice([0, 0.001, 0.002, 0.003], 2)

    total_noise = []

    for i in noise_amount: 
        steps = 0
        sd = 0
        while steps < 6000:
            # start with mean of 1 and sd of 0
            sd += i
            noise = np.random.normal(1, sd, 1).item()
            total_noise.append(noise)
            steps += 1
    
    # to array
    all_noise_arr = np.asarray(total_noise)
    all_noise = all_noise_arr.reshape(2,6000)


     # for each starting condition, run the model from 2005 and keep track of final conditions
    run_number = 0
    final_results_list = []


    # take the accepted parameters, and go row by row, running the model
    for i, j in zip (noise_amount, all_noise):
        

        print(run_number)

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
        initial_wood = 0.058
        initial_grass = 0.899
        initial_scrub = 0.043

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
        redDeer_stocking_forecast = 35
        tamworthPig_stocking_forecast = 7
        exmoor_stocking_forecast = 15

        # experiment parameters
        exp_chance_reproduceSapling = 0
        exp_chance_reproduceYoungScrub =  0
        exp_chance_regrowGrass = 0
        duration = 0
        tree_reduction = j

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
                            experiment_growth = False, experiment_wood = False, experiment_linear_growth = True, 
                            max_time = 6000, max_roe = 500)


        model.reset_randomizer(seed=1)

        model.run_model()

        run_number +=1

        results = model.datacollector.get_model_vars_dataframe()
        results["Noise"] = i
        results["run_number"] = run_number

        habs_only = results[['Roe deer', 'Grassland', 'Woodland', 'Thorny Scrub', 'Bare ground', 'Time', 'run_number', 'Noise', 'Grass', 'Trees', 'Mature Scrub', 'Saplings', 'Young Scrub']]
        
        final_results_list.append(habs_only)


    # append to dataframe
    experiment_growth = pd.concat(final_results_list)


    final_df =  pd.DataFrame(
                    (experiment_growth[['Roe deer', 'Grassland', 'Woodland', 'Thorny Scrub', 'Bare ground', 'Grass', 'Trees', 'Mature Scrub', 'Saplings', 'Young Scrub']].values.flatten()), columns=['Abundance'])
    final_df["Run Number"] = pd.DataFrame(np.concatenate([np.repeat(experiment_growth['run_number'],10)], axis=0))
    final_df["Time"] = pd.DataFrame(np.concatenate([np.repeat(experiment_growth['Time'], 10)], axis=0))
    final_df["Ecosystem Element"] = pd.DataFrame(np.tile(['Roe deer', "Grassland", "Woodland", "Thorny Scrub", "Bare ground", 'Grass', 'Trees', 'Mature Scrub', 'Saplings', 'Young Scrub'], len(experiment_growth)))
    final_df["Noise"] = pd.DataFrame(np.concatenate([np.repeat(experiment_growth['Noise'],10)], axis=0))


    # final_df =  pd.DataFrame(
    #                 (experiment_growth[['Roe deer', 'Grassland', 'Woodland', 'Thorny Scrub', 'Bare ground']].values.flatten()), columns=['Abundance'])
    # final_df["Run Number"] = pd.DataFrame(np.concatenate([np.repeat(experiment_growth['run_number'],5)], axis=0))
    # final_df["Time"] = pd.DataFrame(np.concatenate([np.repeat(experiment_growth['Time'], 5)], axis=0))
    # final_df["Ecosystem Element"] = pd.DataFrame(np.tile(['Roe deer', "Grassland", "Woodland", "Thorny Scrub", "Bare ground"], len(experiment_growth)))
    # final_df["Noise"] = pd.DataFrame(np.concatenate([np.repeat(experiment_growth['Noise'],5)], axis=0))


    final_df.to_csv("experiment_linear_growth_test.csv")



# experiment_linear_growth()









def no_control():

    # loop through all PS
    best_parameters = pd.read_csv('abm_final_results_accepted_params_reshaped.csv').drop(['Unnamed: 0'], axis=1)

    final_results_list = []
    run_number = 0


    # for each starting condition, run the model from 2005 and keep track of final conditions
    run_number = 0
    final_results_list = []


    # take the accepted parameters, and go row by row, running the model
    for index, best_parameter in best_parameters.iterrows():        

        print(run_number)

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
        initial_wood = 0.058
        initial_grass = 0.899
        initial_scrub = 0.043

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
        exmoor_stocking_forecast = int(15 * 1.8)
        fallowDeer_stocking_forecast = int(247 * 0.95)
        cattle_stocking_forecast = int(81 * 0.23)
        redDeer_stocking_forecast = int(35 * 2.3)
        tamworthPig_stocking_forecast = int(7 * 0.56)

        # fallowDeer_stocking_forecast = 247
        # cattle_stocking_forecast = 81
        # redDeer_stocking_forecast = 35
        # tamworthPig_stocking_forecast = 7
        # exmoor_stocking_forecast = 15

        # experiment parameters
        # exp_chance_reproduceSapling = 0
        # exp_chance_reproduceYoungScrub =  0
        # exp_chance_regrowGrass = 0
        # duration = 0

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
                            experiment_growth = False, experiment_wood = False, experiment_linear_growth = False, 
                            max_time = 3000, max_roe = 500)


        model.reset_randomizer(seed=1)

        model.run_model()

        run_number +=1

        results = model.datacollector.get_model_vars_dataframe()
        results["run_number"] = run_number

        habs_only = results[['Roe deer', 'Exmoor pony', 'Longhorn cattle', 'Fallow deer', 'Red deer', 'Tamworth pigs', 'Grassland', 'Woodland', 'Thorny Scrub', 'Bare ground', 'Time', 'run_number']]


        print(habs_only)
        
        final_results_list.append(habs_only)


    # append to dataframe
    experiment_growth = pd.concat(final_results_list)



    final_df =  pd.DataFrame(
                    (experiment_growth[['Roe deer', 'Exmoor pony', 'Longhorn cattle', 'Fallow deer', 'Red deer', 'Tamworth pigs', 'Grassland', 'Woodland', 'Thorny Scrub', 'Bare ground']].values.flatten()), columns=['Abundance'])
    final_df["Run Number"] = pd.DataFrame(np.concatenate([np.repeat(experiment_growth['run_number'],10)], axis=0))
    final_df["Time"] = pd.DataFrame(np.concatenate([np.repeat(experiment_growth['Time'], 10)], axis=0))
    final_df["Ecosystem Element"] = pd.DataFrame(np.tile(['Roe deer', 'Exmoor pony', 'Longhorn cattle', 'Fallow deer', 'Red deer', 'Tamworth pigs', 'Grassland', 'Woodland', 'Thorny Scrub', 'Bare ground'], len(experiment_growth)))
    final_df["Experiment"] = "33% Scrubland"


    # final_df.to_csv("experiment_ga_no_control.csv")
    # final_df.to_csv("experiment_ps1_all_100_runs.csv")
    # final_df.to_csv("experiment_ga_outputs_all_100_runs.csv")

    # GA experiments
    final_df.to_csv("ga1_all_timeseries_abm_scrubland.csv")



no_control()