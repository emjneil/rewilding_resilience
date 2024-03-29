import mesa
import mesa_geo as mg
import geopandas as gpd
import random
import numpy as np
import uuid
from agents import FieldAgent, roe_deer_agent, exmoor_pony_agent, longhorn_cattle_agent, fallow_deer_agent, red_deer_agent, tamworth_pig_agent, european_bison_agent, european_elk_agent
from space import FieldSpace
from schedule import RandomActivationByBreed
from movements import browser_move, grazer_move, mixed_diet_move, eat_habitats, random_move
from mesa.datacollection import DataCollector
from random import randrange


class KneppModel(mesa.Model):

    def __init__(self, initial_roe, roe_deer_reproduce, roe_deer_gain_from_saplings, roe_deer_gain_from_trees, roe_deer_gain_from_scrub, roe_deer_gain_from_young_scrub, roe_deer_gain_from_grass,
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
                        reintroduction, introduce_euroBison, introduce_elk, 
                        experiment_growth, experiment_wood, experiment_linear_growth, 
                        max_time, max_roe):

        # add the schedule and experiments 
        self.schedule = RandomActivationByBreed(self)
        self.reintroduction = reintroduction
        self.introduce_euroBison = introduce_euroBison
        self.introduce_elk = introduce_elk
        self.experiment_growth = experiment_growth
        self.experiment_wood = experiment_wood
        self.experiment_linear_growth = experiment_linear_growth
        self.max_time = max_time
        self.max_roe = max_roe


        # first define the space and add the field polygon agents
        self.space = FieldSpace()

        ac = mg.AgentCreator(agent_class=FieldAgent, model=self, crs="epsg:27700", agent_kwargs={"initial_wood": initial_wood, "initial_grass": initial_grass, "initial_scrub": initial_scrub})
        fields = ac.from_file("cleaned_shp.shp", unique_id="id")
        self.fields = fields 
        self.space.add_fields(fields)
        # now add fields and calculate all the neighbors for each one
        self.saved_neighbors={}
        for field in fields:
            self.schedule.add(field)
            my_neighbors = self.space.get_neighbors_within_distance(field, 1)
            self.saved_neighbors[field.unique_id] = list(my_neighbors)
        # add other parameters
        self.initial_roe = initial_roe
        self.roe_deer_reproduce = roe_deer_reproduce
        self.roe_deer_gain_from_saplings = roe_deer_gain_from_saplings
        self.roe_deer_gain_from_trees = roe_deer_gain_from_trees
        self.roe_deer_gain_from_scrub = roe_deer_gain_from_scrub
        self.roe_deer_gain_from_young_scrub = roe_deer_gain_from_young_scrub
        self.roe_deer_gain_from_grass = roe_deer_gain_from_grass
        # vegetation growth
        self.chance_youngScrubMatures = chance_youngScrubMatures
        self.chance_saplingBecomingTree = chance_saplingBecomingTree

        self.chance_reproduceSapling = chance_reproduceSapling
        self.chance_reproduceYoungScrub = chance_reproduceYoungScrub
        self.chance_regrowGrass = chance_regrowGrass
        # remember what it was originally  (for experiments)
        self.original_chance_reproduceSapling = chance_reproduceSapling
        self.original_chance_reproduceYoungScrub = chance_reproduceYoungScrub
        self.original_chance_regrowGrass = chance_regrowGrass
        # competition parameters
        self.chance_grassOutcompetedByTree = chance_grassOutcompetedByTree
        self.chance_grassOutcompetedByScrub = chance_grassOutcompetedByScrub
        self.chance_scrubOutcompetedByTree = chance_scrubOutcompetedByTree
        # pony parameters
        self.ponies_gain_from_saplings = ponies_gain_from_saplings
        self.ponies_gain_from_trees = ponies_gain_from_trees
        self.ponies_gain_from_scrub = ponies_gain_from_scrub
        self.ponies_gain_from_young_scrub = ponies_gain_from_young_scrub
        self.ponies_gain_from_grass = ponies_gain_from_grass
        # cattle parameters
        self.cattle_reproduce = cattle_reproduce 
        self.cows_gain_from_grass = cows_gain_from_grass 
        self.cows_gain_from_trees = cows_gain_from_trees
        self.cows_gain_from_scrub = cows_gain_from_scrub
        self.cows_gain_from_saplings = cows_gain_from_saplings
        self.cows_gain_from_young_scrub = cows_gain_from_young_scrub
        # fallow deer parameters
        self.fallow_deer_reproduce = fallow_deer_reproduce
        self.fallow_deer_gain_from_saplings = fallow_deer_gain_from_saplings
        self.fallow_deer_gain_from_trees = fallow_deer_gain_from_trees
        self.fallow_deer_gain_from_scrub = fallow_deer_gain_from_scrub
        self.fallow_deer_gain_from_young_scrub = fallow_deer_gain_from_young_scrub
        self.fallow_deer_gain_from_grass = fallow_deer_gain_from_grass
        # red deer parameters
        self.red_deer_reproduce = red_deer_reproduce
        self.red_deer_gain_from_saplings = red_deer_gain_from_saplings
        self.red_deer_gain_from_trees = red_deer_gain_from_trees
        self.red_deer_gain_from_scrub = red_deer_gain_from_scrub
        self.red_deer_gain_from_young_scrub = red_deer_gain_from_young_scrub
        self.red_deer_gain_from_grass = red_deer_gain_from_grass
        # tamworth pig parameters
        self.tamworth_pig_reproduce = tamworth_pig_reproduce
        self.tamworth_pig_gain_from_saplings = tamworth_pig_gain_from_saplings
        self.tamworth_pig_gain_from_trees = tamworth_pig_gain_from_trees
        self.tamworth_pig_gain_from_scrub = tamworth_pig_gain_from_scrub
        self.tamworth_pig_gain_from_young_scrub = tamworth_pig_gain_from_young_scrub
        self.tamworth_pig_gain_from_grass = tamworth_pig_gain_from_grass
        # european bison parameters
        self.european_bison_reproduce = european_bison_reproduce
        self.european_bison_gain_from_grass = european_bison_gain_from_grass
        self.european_bison_gain_from_trees = european_bison_gain_from_trees
        self.european_bison_gain_from_scrub = european_bison_gain_from_scrub
        self.european_bison_gain_from_saplings = european_bison_gain_from_saplings
        self.european_bison_gain_from_young_scrub = european_bison_gain_from_young_scrub
        # european elk parameters
        self.european_elk_reproduce = european_elk_reproduce
        self.european_elk_gain_from_grass = european_elk_gain_from_grass
        self.european_elk_gain_from_trees = european_elk_gain_from_trees
        self.european_elk_gain_from_scrub = european_elk_gain_from_scrub
        self.european_elk_gain_from_saplings = european_elk_gain_from_saplings
        self.european_elk_gain_from_young_scrub = european_elk_gain_from_young_scrub
        # and stocking for forecasting 
        self.fallowDeer_stocking_forecast = fallowDeer_stocking_forecast
        self.cattle_stocking_forecast = cattle_stocking_forecast
        self.redDeer_stocking_forecast = redDeer_stocking_forecast
        self.tamworthPig_stocking_forecast = tamworthPig_stocking_forecast
        self.exmoor_stocking_forecast = exmoor_stocking_forecast
        # chance of tree and scrub mortality
        self.chance_scrub_saves_saplings = chance_scrub_saves_saplings
        self.exp_chance_reproduceSapling = exp_chance_reproduceSapling
        self.exp_chance_reproduceYoungScrub = exp_chance_reproduceYoungScrub
        self.exp_chance_regrowGrass = exp_chance_regrowGrass
        self.duration = duration
        self.tree_reduction = tree_reduction


        # then add the herbivores as points
        for _ in range(initial_roe): # number of roe deer
            field = random.choice(fields) # randomly pick field
            energy = 0
            roe = roe_deer_agent(
                unique_id=uuid.uuid4().int,
                model = self,
                crs=self.space.crs,
                geometry=field.random_point(),
                field_id=field.unique_id,
                energy = energy,
                move = browser_move, 
                eat = eat_habitats
                )
            self.space.add_herbivore_agent(roe, field_id=field.unique_id)
            self.schedule.add(roe)


        # get data organized
        self.datacollector = DataCollector(

        model_reporters = {
            # number and type of habitats
            "Time": lambda m: m.schedule.time, 
            "Roe deer": lambda m: m.schedule.get_breed_count(roe_deer_agent),
            "Exmoor pony": lambda m: m.schedule.get_breed_count(exmoor_pony_agent),
            "Longhorn cattle": lambda m: m.schedule.get_breed_count(longhorn_cattle_agent),
            "Fallow deer": lambda m: m.schedule.get_breed_count(fallow_deer_agent),
            "Red deer": lambda m: m.schedule.get_breed_count(red_deer_agent),
            "Tamworth pigs": lambda m: m.schedule.get_breed_count(tamworth_pig_agent),
            # "European bison": lambda m: m.schedule.get_breed_count(european_bison_agent),
            # "European elk": lambda m: m.schedule.get_breed_count(european_elk_agent),

            # habitat types
            "Grassland": lambda m: self.count_condition(m, "grassland"),
            "Woodland": lambda m: self.count_condition(m, "woodland"),
            "Thorny Scrub": lambda m: self.count_condition(m, "thorny_scrubland"),
            "Bare ground": lambda m: self.count_condition(m, "bare_ground"),     

            # number of habitat types
            "Grass": lambda m: self.count_habitat_numbers(m, "grass"),
            "Trees": lambda m: self.count_habitat_numbers(m, "trees"),
            "Mature Scrub": lambda m: self.count_habitat_numbers(m, "scrub"),
            "Saplings": lambda m: self.count_habitat_numbers(m, "saplings"),
            "Young Scrub": lambda m: self.count_habitat_numbers(m,"youngScrub"),
            "Bare Areas": lambda m: self.count_habitat_numbers(m, "bare_ground"),

            }
            )


        self.running = True
        self.datacollector.collect(self)



    def count_condition(self, model, habitat_condition):
        # want to count grass, wood, scrub, bare ground in each patch
        count = 0
        for key, value in model.schedule.agents_by_breed[FieldAgent].items():
            if value.condition == habitat_condition:
                count += 1
        # return percentage of entire area (number of grid cells)
        return round((count/506)*100)



    def count_habitat_numbers(self, model, habitat_thing):
        # want to count grass, wood, scrub, bare ground in each patch
        count_item = 0
        for key, value in model.schedule.agents_by_breed[FieldAgent].items():
            count_item += value.edibles[habitat_thing]
        # return percentage of entire area
        return count_item
        

    ### ------ functions related to reintroductions ---- ###

    def get_month(self):
        return (self.schedule.time % 12) + 1 

    def add_herbivores(self, herbivore, count, movement):
        field = random.choice(self.fields)
        for i in range(count):
            to_add = herbivore(
                unique_id=uuid.uuid4().int,
                model = self,
                crs=self.space.crs,
                geometry=field.random_point(),
                field_id=field.unique_id,
                energy = 0,
                move = movement, 
                eat = eat_habitats
            )
            self.space.add_herbivore_agent(to_add, field_id=field.unique_id)
            self.schedule.add(to_add)


    def remove_herbivores(self, herbivore, count):
        to_remove = self.schedule.agents_by_breed[herbivore].items()
        my_choices = random.sample(list(to_remove), k = min(count, len(to_remove)))
        for my_choice in my_choices:
            my_choice = my_choice[1]
            self.space.remove_herbivore_agent(my_choice)
            self.schedule.remove(my_choice)


    def add_pig(self, herbivore, count_piglets, count_sow, count_boar, movement):
        # pick a field to put it in
        field = random.choice(self.fields)
        # assign energy
        for i in range(count_piglets):
            to_add = herbivore(
                unique_id=uuid.uuid4().int,
                model = self,
                crs=self.space.crs,
                geometry=field.random_point(),
                field_id=field.unique_id,
                energy = 0,
                condition = "piglet",
                move = movement, 
                eat = eat_habitats
            )
            self.space.add_herbivore_agent(to_add, field_id=field.unique_id)
            self.schedule.add(to_add)
        for i in range(count_sow):
            to_add = herbivore(
                unique_id=uuid.uuid4().int,
                model = self,
                crs=self.space.crs,
                geometry=field.random_point(),
                field_id=field.unique_id,
                energy = 0,
                condition = "sow",
                move = movement, 
                eat = eat_habitats
            )
            self.space.add_herbivore_agent(to_add, field_id=field.unique_id)
            self.schedule.add(to_add)
        for i in range(count_boar):
            to_add = herbivore(
                unique_id=uuid.uuid4().int,
                model = self,
                crs=self.space.crs,
                geometry=field.random_point(),
                field_id=field.unique_id,
                energy = 0,
                condition = "boar",
                move = movement, 
                eat = eat_habitats
            )
            self.space.add_herbivore_agent(to_add, field_id=field.unique_id)
            self.schedule.add(to_add)


    def remove_pig(self, herbivore, count_piglets, count_sow, count_boar):
        to_remove = self.schedule.agents_by_breed[herbivore].items()
        to_remove_piglets = [i for (k, i) in to_remove if i.condition == "piglet"]
        to_remove_sow = [i for (k, i) in to_remove if i.condition == "sow"]
        to_remove_boars = [i for (k, i) in to_remove if i.condition == "boar"]
        my_choice_piglet = random.sample(list(to_remove_piglets), k = min(count_piglets, len(to_remove_piglets)))
        my_choice_sow = random.sample(list(to_remove_sow), k = min(count_sow, len(to_remove_sow)))
        my_choices_boar = random.sample(list(to_remove_boars), k = min(count_boar, len(to_remove_boars)))

        # remove non-boars
        for my_choice in my_choice_piglet:
            my_choice = my_choice
            self.space.remove_herbivore_agent(my_choice)
            self.schedule.remove(my_choice)
        for my_choice_sows in my_choice_sow:
            my_choice_sow = my_choice_sows
            self.space.remove_herbivore_agent(my_choice_sow)
            self.schedule.remove(my_choice_sow)
        # remove boars
        for my_choice_boar in my_choices_boar:
            my_choice_boar = my_choice_boar
            self.space.remove_herbivore_agent(my_choice_boar)
            self.schedule.remove(my_choice_boar)




    ### ------ time steps ---- ###

    def step(self):

        self.datacollector.collect(self)

        self.schedule.step()

        # # set it unrealistically high for now, it will be reset later
        # experiment_start = 100000
        # experiment_end = 110000
        # # remember the original parameters
        # original_growth = self.chance_reproduceSapling

        # reintroduce species
        if self.reintroduction == True:
            # March 2009
            if self.schedule.time == 49:
            
                self.add_herbivores(exmoor_pony_agent, 23, grazer_move) 
                self.add_herbivores(longhorn_cattle_agent, 53, grazer_move)
                number_sows = random.randint(4,8)
                number_piglets = 20-number_sows
                self.add_pig(tamworth_pig_agent,number_piglets, number_sows, 0, random_move)
  
            # March 2010
            if self.schedule.time == 61:

                outputs = self.datacollector.get_model_vars_dataframe()
                if (outputs.iloc[-1]["Roe deer"] <= 160): # don't let roe deer explode
                    # exmoor ponies
                    if outputs.iloc[61]['Exmoor pony'] < 13:
                        number_to_add = int(13 - outputs.iloc[61]['Exmoor pony'])
                        self.add_herbivores(exmoor_pony_agent, number_to_add, grazer_move)
                    # longhorn cows
                    if outputs.iloc[61]['Longhorn cattle'] >= 77:
                        number_to_subtract = int(-77 + outputs.iloc[61]['Longhorn cattle'])
                        self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                    else:
                        number_to_add = int(77 - outputs.iloc[61]['Longhorn cattle'])
                        self.add_herbivores(longhorn_cattle_agent, number_to_add, grazer_move)
                    # add fallow deer
                    self.add_herbivores(fallow_deer_agent, 53, mixed_diet_move)
                    # tamworth pigs
                    if outputs.iloc[61]['Tamworth pigs'] >= 17:
                        number_to_subtract = int(-17 + outputs.iloc[61]['Tamworth pigs'])
                        self.remove_pig(tamworth_pig_agent, number_to_subtract, 0, 0)
                    else:
                        number_to_add = int(17 - outputs.iloc[61]['Tamworth pigs'])
                        self.add_pig(tamworth_pig_agent, number_to_add, 0, 0, random_move)
                else: 
                    self.running = False


            # March 2011
            if self.schedule.time == 73:
                
                outputs = self.datacollector.get_model_vars_dataframe()
                # exmoor ponies
                if outputs.iloc[73]['Exmoor pony'] >= 15: 
                    number_to_subtract = int(-15 + outputs.iloc[73]['Exmoor pony'])
                    self.remove_herbivores(exmoor_pony_agent, number_to_subtract)
                else:
                    number_to_add = int(15 - outputs.iloc[73]['Exmoor pony'])
                    self.add_herbivores(exmoor_pony_agent, number_to_add, grazer_move)
                # longhorn cattle
                if outputs.iloc[73]['Longhorn cattle'] >= 92:
                    number_to_subtract = int(-92 + outputs.iloc[73]['Longhorn cattle'])
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                else:
                    number_to_add = int(92 - outputs.iloc[73]['Longhorn cattle'])
                    self.add_herbivores(longhorn_cattle_agent, number_to_add, grazer_move)
                # fallow deer
                if outputs.iloc[73]['Fallow deer'] >= 81:
                    number_to_subtract = int(-81 + outputs.iloc[73]['Fallow deer'])
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                else:
                    number_to_add = int(81 - outputs.iloc[73]['Fallow deer'])
                    self.add_herbivores(fallow_deer_agent, number_to_add, mixed_diet_move)
                # tamworth pigs
                if outputs.iloc[73]['Tamworth pigs'] >= 22:
                    number_to_subtract = int(-22 + outputs.iloc[73]['Tamworth pigs'])
                    self.remove_pig(tamworth_pig_agent, number_to_subtract, 0, 0)
                else:
                    number_to_add = int(22 - outputs.iloc[73]['Tamworth pigs'])
                    self.add_pig(tamworth_pig_agent, number_to_add, 0, 0, random_move)
            # March 2012
            if self.schedule.time == 85:
                outputs = self.datacollector.get_model_vars_dataframe()
                if outputs.iloc[85]['Exmoor pony'] >= 17: # randomly choose that many exmoor ponies and delete them
                    number_to_subtract = int(-17 + outputs.iloc[85]['Exmoor pony'])
                    self.remove_herbivores(exmoor_pony_agent, number_to_subtract)
                else:
                    number_to_add = int(17 - outputs.iloc[85]['Exmoor pony'])
                    self.add_herbivores(exmoor_pony_agent, number_to_add, grazer_move)
                if outputs.iloc[85]['Longhorn cattle'] >= 116:
                    number_to_subtract = int(-116 + outputs.iloc[85]['Longhorn cattle'])
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                else:
                    number_to_add = int(116 - outputs.iloc[85]['Longhorn cattle'])
                    self.add_herbivores(longhorn_cattle_agent, number_to_add, grazer_move)
                if outputs.iloc[85]['Fallow deer'] >= 100:
                    number_to_subtract = int(-100 + outputs.iloc[85]['Fallow deer'])
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                else:
                    number_to_add = int(100 - outputs.iloc[85]['Fallow deer'])
                    self.add_herbivores(fallow_deer_agent, number_to_add, mixed_diet_move)
                if outputs.iloc[85]['Tamworth pigs'] >= 33:
                    number_to_subtract = int(-33 + outputs.iloc[85]['Tamworth pigs'])
                    self.remove_pig(tamworth_pig_agent, number_to_subtract, 0, 0)
                else:
                    number_to_add = int(33 - outputs.iloc[85]['Tamworth pigs'])
                    self.add_pig(tamworth_pig_agent, number_to_add, 0, 0, random_move)

            # March 2013
            if self.schedule.time == 97:
                outputs = self.datacollector.get_model_vars_dataframe()
                # exmoor ponies
                if outputs.iloc[97]['Exmoor pony'] >= 10: # randomly choose that many exmoor ponies and delete them
                    number_to_subtract = int(-10 + outputs.iloc[97]['Exmoor pony'])
                    self.remove_herbivores(exmoor_pony_agent, number_to_subtract)
                else:
                    number_to_add = int(10 - outputs.iloc[97]['Exmoor pony'])
                    self.add_herbivores(exmoor_pony_agent, number_to_add, grazer_move)
                # red deer
                self.add_herbivores(red_deer_agent, 13, mixed_diet_move)
                # longhorn cattle
                if outputs.iloc[97]['Longhorn cattle'] >= 129:
                    number_to_subtract = int(-129 + outputs.iloc[97]['Longhorn cattle'])
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                else:
                    number_to_add = int(129 - outputs.iloc[97]['Longhorn cattle'])
                    self.add_herbivores(longhorn_cattle_agent, number_to_add, grazer_move)
                # fallow deer
                if outputs.iloc[97]['Fallow deer'] >= 100:
                    number_to_subtract = int(-100 + outputs.iloc[97]['Fallow deer'])
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                else:
                    number_to_add = int(100 - outputs.iloc[97]['Fallow deer'])
                    self.add_herbivores(fallow_deer_agent, number_to_add, mixed_diet_move)
                # tamworth pigs
                if outputs.iloc[97]['Tamworth pigs'] >= 6:
                    number_to_subtract = int(-6 + outputs.iloc[97]['Tamworth pigs'])
                    self.remove_pig(tamworth_pig_agent, number_to_subtract, 0, 0)
                else:
                    number_to_add = int(6 - outputs.iloc[97]['Tamworth pigs'])
                    self.add_pig(tamworth_pig_agent, number_to_add, 0, 0, random_move)
            # March 2014
            if self.schedule.time == 109:
                outputs = self.datacollector.get_model_vars_dataframe()
                if outputs.iloc[109]['Exmoor pony'] >= 10: # randomly choose that many exmoor ponies and delete them
                    number_to_subtract = int(-10 + outputs.iloc[109]['Exmoor pony'])
                    self.remove_herbivores(exmoor_pony_agent, number_to_subtract)
                else:
                    number_to_add = int(10 - outputs.iloc[109]['Exmoor pony'])
                    self.add_herbivores(exmoor_pony_agent, number_to_add, grazer_move)
                if outputs.iloc[109]['Longhorn cattle'] >= 264:
                    number_to_subtract = int(-264 + outputs.iloc[109]['Longhorn cattle'])
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                else:
                    number_to_add = int(264 - outputs.iloc[109]['Longhorn cattle'])
                    self.add_herbivores(longhorn_cattle_agent, number_to_add, grazer_move)
                if outputs.iloc[109]['Fallow deer'] >= 100:
                    number_to_subtract = int(-100 + outputs.iloc[109]['Fallow deer'])
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                else:
                    number_to_add = int(100 - outputs.iloc[109]['Fallow deer'])
                    self.add_herbivores(fallow_deer_agent, number_to_add, mixed_diet_move)
                if outputs.iloc[109]['Tamworth pigs'] >= 18:
                    number_to_subtract = int(-18 + outputs.iloc[109]['Tamworth pigs'])
                    self.remove_pig(tamworth_pig_agent, number_to_subtract, 0, 0)
                else:
                    number_to_add = int(18 - outputs.iloc[109]['Tamworth pigs'])
                    self.add_pig(tamworth_pig_agent, number_to_add, 0, 0, random_move)
                if outputs.iloc[109]['Red deer'] >= 13:
                    number_to_subtract = int(-13 + outputs.iloc[109]['Red deer'])
                    self.remove_herbivores(red_deer_agent, number_to_subtract)
                else:
                    number_to_add = int(13 - outputs.iloc[109]['Red deer'])
                    self.add_herbivores(red_deer_agent, number_to_add, mixed_diet_move)
            
                                    # # # # # # # 2015 # # # # # # #

            # Jan 2015 - assumed 1 boar added 
            if self.schedule.time == 119:
                self.add_pig(tamworth_pig_agent, 0, 0, 1, random_move)
            # Feb 2015 - assumed 1 boar removed, -2 cows
            if self.schedule.time == 120:
                self.remove_pig(tamworth_pig_agent, 0, 0, 1)
            # March 2015
            if self.schedule.time == 121:
                outputs = self.datacollector.get_model_vars_dataframe()
                #  exmoor ponies
                if outputs.iloc[121]['Exmoor pony'] >= 10:
                    number_to_subtract = int(-10 + outputs.iloc[121]['Exmoor pony'])
                    self.remove_herbivores(exmoor_pony_agent, number_to_subtract)
                else:
                    number_to_add = int(10 - outputs.iloc[121]['Exmoor pony'])
                    self.add_herbivores(exmoor_pony_agent, number_to_add, grazer_move)
                # longhorn cattle
                if outputs.iloc[121]['Longhorn cattle'] >= 107:
                    number_to_subtract = int(-107 + outputs.iloc[121]['Longhorn cattle'])
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                else:
                    number_to_add = int(107 - outputs.iloc[121]['Longhorn cattle'])
                    self.add_herbivores(longhorn_cattle_agent, number_to_add, grazer_move)
                # fallow deer
                if outputs.iloc[121]['Fallow deer'] >= 100:
                    number_to_subtract = int(-100 + outputs.iloc[121]['Fallow deer'])
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                else:
                    number_to_add = int(100 - outputs.iloc[121]['Fallow deer'])
                    self.add_herbivores(fallow_deer_agent, number_to_add, mixed_diet_move)
                # tamworth pigs
                total_pigs = self.schedule.agents_by_breed[tamworth_pig_agent].items()
                number_piglets = [i for (k, i) in total_pigs if i.condition == "piglet"]
                number_sows = [i for (k, i) in total_pigs if i.condition == "sow"]
                if len(number_piglets) >= 13:
                    number_to_subtract_piglets = -13 + len(number_piglets)
                    self.remove_pig(tamworth_pig_agent, number_to_subtract_piglets, 0, 0)
                else:
                    number_to_add_piglets = 13 - len(number_piglets)
                    self.add_pig(tamworth_pig_agent, number_to_add_piglets, 0, 0, random_move)
                if len(number_sows) >= 5:
                    number_to_subtract_sows = -5 + len(number_sows)
                    self.remove_pig(tamworth_pig_agent, 0, number_to_subtract_sows, 0)
                else:
                    number_to_add_sows = 5 - len(number_sows)
                    self.add_pig(tamworth_pig_agent, 0, number_to_add_sows, 0, random_move)
                # red deer
                if outputs.iloc[121]['Red deer'] >= 13:
                    number_to_subtract = int(-13 + outputs.iloc[121]['Red deer'])
                    self.remove_herbivores(red_deer_agent, number_to_subtract)
                else:
                    number_to_add = int(13 - outputs.iloc[121]['Red deer'])
                    self.add_herbivores(red_deer_agent, number_to_add, mixed_diet_move)
            # April 2015
            if self.schedule.time == 122:
                self.remove_pig(tamworth_pig_agent, 0, 1, 0)
            # May 2015
            if self.schedule.time == 123:
                # May 2015: 8 piglets culled
                self.remove_pig(tamworth_pig_agent, 8, 0, 0)
            # June 2015: 5 cows culled
            if self.schedule.time == 124:
                self.remove_herbivores(longhorn_cattle_agent, 5)
            # August 2015: 2 fallow deer culled
            if self.schedule.time == 126:
                self.remove_herbivores(fallow_deer_agent, 2)
            # September 2015: 2 male fallow deer culled; 2 cattle culled and 3 bulls added
            if self.schedule.time == 128:
                self.remove_herbivores(fallow_deer_agent, 2)
                self.add_herbivores(longhorn_cattle_agent, 1, grazer_move)
            # Oct 2015: 2 female and 1 male fallow deer culled; 38 female cows and 1 bull removed
            if self.schedule.time == 128:
                self.remove_herbivores(fallow_deer_agent, 3)
                self.remove_herbivores(longhorn_cattle_agent, 39)
            # Nov 2015: -7 fallow deer, -1 piglet
            if self.schedule.time == 129:
                self.remove_herbivores(fallow_deer_agent, 7)                 
                self.remove_pig(tamworth_pig_agent, 1, 0, 0)       
            # Dec 2015: 6 fallow deer culled; 5 cows removed;
            if self.schedule.time == 130:
                self.remove_herbivores(fallow_deer_agent, 6)
                self.remove_herbivores(longhorn_cattle_agent, 5)
            # Jan 2016: 7 fallow deer culled; 4 pigs culled and 1 added
            if self.schedule.time == 131:
                self.remove_herbivores(fallow_deer_agent, 7) 
                self.remove_pig(tamworth_pig_agent, 4, 0, 0)
                self.add_pig(tamworth_pig_agent, 0, 0, 1, random_move)
            # Feb 2016: 10 fallow deer culled; 2 pigs culled
            if self.schedule.time == 132:
                self.remove_herbivores(fallow_deer_agent, 10)
                self.remove_pig(tamworth_pig_agent, 2, 0, 0)


                                    # # # # # # # 2016 # # # # # # #

            # March 2016
            if self.schedule.time == 133:
                self.add_herbivores(exmoor_pony_agent, 1, grazer_move)
                self.remove_pig(tamworth_pig_agent, 1, 0, 1)
                self.add_pig(tamworth_pig_agent, 0, 3, 0, random_move)
            # April 2016
            if self.schedule.time == 134:
                self.add_herbivores(longhorn_cattle_agent, 1, grazer_move)
            # May 2016
            if self.schedule.time == 135:
                self.remove_herbivores(longhorn_cattle_agent, 2)
            # June 2016
            if self.schedule.time == 136:
                self.remove_herbivores(longhorn_cattle_agent, 26)
            # July 2016
            if self.schedule.time == 137:
                self.remove_herbivores(longhorn_cattle_agent, 2)
            # August 2016
            if self.schedule.time == 138:
                self.remove_herbivores(fallow_deer_agent, 5)
            # September & Oct 2016
            if self.schedule.time == 139:
                self.add_herbivores(longhorn_cattle_agent, 9, grazer_move)
            # November 2016
            if self.schedule.time == 141:
                self.remove_herbivores(fallow_deer_agent, 3)
                self.remove_herbivores(longhorn_cattle_agent, 5)
            # December 2016
            if self.schedule.time == 142:
                self.remove_herbivores(fallow_deer_agent, 9)
                self.remove_herbivores(longhorn_cattle_agent, 13)
                self.remove_pig(tamworth_pig_agent, 4, 0, 0)
            # January 2017
            if self.schedule.time == 143:
                self.remove_pig(tamworth_pig_agent, 2, 2, 0)
                self.add_pig(tamworth_pig_agent, 0, 0, 1, random_move)
            # February 2017
            if self.schedule.time == 144:
                self.remove_herbivores(fallow_deer_agent, 8)
                self.remove_pig(tamworth_pig_agent, 1, 1, 1)


                                    # # # # # # # # 2017 # # # # # # #

            # March 2017
            if self.schedule.time == 145:
                self.remove_herbivores(exmoor_pony_agent, 1)
                outputs = self.datacollector.get_model_vars_dataframe()
                if (outputs.iloc[-1]["Roe deer"] <= 160): # don't let roe deer explode
                    if outputs.iloc[145]['Red deer'] >= 14:
                        number_to_subtract = int(-14 + outputs.iloc[145]['Red deer'])
                        self.remove_herbivores(red_deer_agent, number_to_subtract)
                    else:
                        number_to_add = int(14 - outputs.iloc[145]['Red deer'])
                        self.add_herbivores(red_deer_agent, number_to_add, mixed_diet_move)
                else:
                    self.running = False
            # April 2017
            if self.schedule.time == 146:
                self.add_herbivores(longhorn_cattle_agent, 3, grazer_move)
            # June & July 2017
            if self.schedule.time == 148:
                self.remove_herbivores(longhorn_cattle_agent, 21)
            # August 2017
            if self.schedule.time == 150:
                self.remove_herbivores(fallow_deer_agent, 16)
            # September 2017
            if self.schedule.time == 151:
                self.remove_herbivores(fallow_deer_agent, 5)
                self.remove_herbivores(longhorn_cattle_agent, 2)
            # October 2017
            if self.schedule.time == 152:
                self.remove_herbivores(fallow_deer_agent, 4)
                self.remove_herbivores(longhorn_cattle_agent, 2)
            # November 2017
            if self.schedule.time == 153:
                self.remove_herbivores(fallow_deer_agent, 2)
            # December 2017
            if self.schedule.time == 154:
                self.remove_herbivores(fallow_deer_agent, 46)
                self.remove_herbivores(red_deer_agent, 1)
                self.remove_pig(tamworth_pig_agent, 4, 0, 0)
            # January 2018
            if self.schedule.time == 155:
                self.remove_pig(tamworth_pig_agent, 9, 0, 0)
                self.add_pig(tamworth_pig_agent, 0, 0, 1, random_move)
            # February 2018
            if self.schedule.time == 156:
                self.remove_herbivores(fallow_deer_agent, 14)
                self.remove_herbivores(red_deer_agent, 1)
                self.remove_pig(tamworth_pig_agent, 0, 0, 1)


                                        # # # # # # # # 2018 # # # # # # #
                
            # March 2018
            if self.schedule.time == 157:
                self.remove_herbivores(exmoor_pony_agent, 1)
                output = self.datacollector.get_model_vars_dataframe()
                if output.iloc[157]['Fallow deer'] >= 251: 
                    number_to_subtract = int(-251 + output.iloc[157]['Fallow deer'])
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                else:
                    number_to_add = int(251 - output.iloc[157]['Fallow deer'])
                    self.add_herbivores(fallow_deer_agent, number_to_add,mixed_diet_move)
                self.remove_pig(tamworth_pig_agent, 3, 0, 0)
                self.add_pig(tamworth_pig_agent, 0, 3, 0, random_move)

            # April 2018
            if self.schedule.time == 158:
                self.add_herbivores(longhorn_cattle_agent, 1, grazer_move)
            # June 2018
            if self.schedule.time == 160:
                self.remove_herbivores(longhorn_cattle_agent, 18)
            # July 2018
            if self.schedule.time == 161:
                self.remove_herbivores(red_deer_agent, 1)
                self.remove_pig(tamworth_pig_agent, 0, 1, 0)
            # August 2018
            if self.schedule.time == 162:
                self.remove_herbivores(exmoor_pony_agent, 9)
                self.remove_herbivores(fallow_deer_agent, 15)
                self.remove_herbivores(longhorn_cattle_agent, 1)
                self.remove_herbivores(red_deer_agent, 1)
            # September 2018
            if self.schedule.time == 163:
                self.remove_herbivores(fallow_deer_agent, 19)
                self.add_herbivores(longhorn_cattle_agent, 4, grazer_move)
            # October 2018
            if self.schedule.time == 164:
                self.remove_herbivores(longhorn_cattle_agent, 5)
                self.remove_herbivores(fallow_deer_agent, 4)
                self.remove_pig(tamworth_pig_agent, 0, 1, 0)
            # November 2018
            if self.schedule.time == 165:
                self.remove_herbivores(longhorn_cattle_agent, 7)
                self.remove_pig(tamworth_pig_agent, 11, 1, 0)
            # December & January 2018/2019
            if self.schedule.time == 166:
                self.remove_herbivores(longhorn_cattle_agent, 4)
                self.remove_herbivores(fallow_deer_agent, 19)
                self.remove_herbivores(red_deer_agent, 1)
            # # February 2019
            if self.schedule.time == 168:
                self.add_pig(tamworth_pig_agent, 0, 0, 1, random_move)                                                                                   
                self.remove_herbivores(longhorn_cattle_agent, 2)
    

                                        # # # # # # # # 2019 # # # # # # #

            # March 2019
            if self.schedule.time == 169:
                self.remove_herbivores(fallow_deer_agent, 7)
                self.remove_herbivores(red_deer_agent, 7)
                self.remove_pig(tamworth_pig_agent, 5, 0, 0)                                                                                   
                self.add_pig(tamworth_pig_agent, 0, 4, 0, random_move)  
            # April 2019                                                                                
            if self.schedule.time == 170:
                self.remove_pig(tamworth_pig_agent, 0, 0, 1)
            # June 2019
            if self.schedule.time == 172:
                self.remove_herbivores(longhorn_cattle_agent, 28)
            # July & Aug 2019
            if self.schedule.time == 173:
                self.remove_pig(tamworth_pig_agent, 27, 0, 0)
                self.add_herbivores(longhorn_cattle_agent, 4, grazer_move)
            # Sept 2019
            if self.schedule.time == 175:
                self.remove_herbivores(fallow_deer_agent, 15)
                self.add_herbivores(longhorn_cattle_agent, 2, grazer_move)
            # Oct 2019
            if self.schedule.time == 176:
                self.remove_herbivores(longhorn_cattle_agent, 5)
            # November 2019
            if self.schedule.time == 177:
                self.remove_herbivores(longhorn_cattle_agent, 1)
                self.remove_herbivores(fallow_deer_agent, 7)
                self.remove_herbivores(red_deer_agent, 3)
            # December 2019
            if self.schedule.time == 178:
                self.remove_herbivores(fallow_deer_agent, 12)
                self.remove_herbivores(longhorn_cattle_agent, 7)
                self.remove_herbivores(red_deer_agent, 4)
                self.add_pig(tamworth_pig_agent, 0, 0, 1, random_move)
            # January 2020: -24 fallow deer
            if self.schedule.time == 179:
                self.remove_herbivores(fallow_deer_agent, 24)
            # February 2020: -12 fallow; -1 cow; -2 red; -2 pigs
            if self.schedule.time == 180:
                self.remove_herbivores(fallow_deer_agent, 12)
                self.remove_herbivores(red_deer_agent, 2)
                self.remove_pig(tamworth_pig_agent, 0, 1, 1)
                self.remove_herbivores(longhorn_cattle_agent, 1)                               

                                        
                                        # # # # # # 2020 # # # # # #

            # March & April 2020
            if self.schedule.time == 181:
                self.add_herbivores(exmoor_pony_agent, 15, grazer_move)
                self.add_herbivores(longhorn_cattle_agent, 2, grazer_move)
                self.remove_pig(tamworth_pig_agent, 0, 1, 0)

                             
            # July 2020             
            if self.schedule.time == 185:
                outputs = self.datacollector.get_model_vars_dataframe()
                # first make sure that exmoor ponies are at their stocking density
                if outputs.iloc[185]['Exmoor pony'] > self.exmoor_stocking_forecast:
                    number_to_subtract = int(-self.exmoor_stocking_forecast + int(outputs.iloc[185]['Exmoor pony']))
                    self.remove_herbivores(exmoor_pony_agent, number_to_subtract)
                else:
                    number_to_add = self.exmoor_stocking_forecast - int(outputs.iloc[185]['Exmoor pony'])
                    self.add_herbivores(exmoor_pony_agent, number_to_add, grazer_move)
                    # Longhorn cattle can be culled in July
                    if outputs.iloc[185]['Longhorn cattle'] > self.cattle_stocking_forecast:
                        number_to_subtract = int(random.randint(0,self.cattle_stocking_forecast))
                        self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                if self.introduce_euroBison == True:
                    self.add_herbivores(european_bison_agent, 50, mixed_diet_move)
                if self.introduce_elk == True:
                    self.add_herbivores(european_elk_agent, 50, browser_move)
            # August 2020
            if self.schedule.time == 186:
                outputs = self.datacollector.get_model_vars_dataframe()
                if outputs.iloc[186]['Longhorn cattle'] > self.cattle_stocking_forecast:
                    number_to_subtract = int(random.randint(0,self.cattle_stocking_forecast))
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                if outputs.iloc[186]['Fallow deer'] > self.fallowDeer_stocking_forecast:
                    number_to_subtract = int(random.randint(0,self.fallowDeer_stocking_forecast))
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                if outputs.iloc[186]['Red deer'] > self.redDeer_stocking_forecast:
                    number_to_subtract = int(random.randint(0,self.redDeer_stocking_forecast))
                    self.remove_herbivores(red_deer_agent, number_to_subtract)
            # Sept 2020
            if self.schedule.time == 187:
                outputs = self.datacollector.get_model_vars_dataframe()
                if outputs.iloc[187]['Longhorn cattle'] > self.cattle_stocking_forecast:
                    number_to_subtract = int(random.randint(0,self.cattle_stocking_forecast))
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                if outputs.iloc[187]['Fallow deer'] > self.fallowDeer_stocking_forecast:
                    number_to_subtract = int(random.randint(0,self.fallowDeer_stocking_forecast))
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
            # Oct 2020
            if self.schedule.time == 188:
                outputs = self.datacollector.get_model_vars_dataframe()
                if outputs.iloc[188]['Longhorn cattle'] > self.cattle_stocking_forecast:
                    number_to_subtract = int(random.randint(0,self.cattle_stocking_forecast))
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
            # Nov 2020
            if self.schedule.time == 189:
                results = self.datacollector.get_model_vars_dataframe()
                if results.iloc[189]['Longhorn cattle'] > self.cattle_stocking_forecast:
                    number_to_subtract = int(random.randint(0,self.cattle_stocking_forecast))
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                if results.iloc[189]['Fallow deer'] > self.fallowDeer_stocking_forecast:
                    number_to_subtract = int(random.randint(0,self.fallowDeer_stocking_forecast))
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                if results.iloc[189]['Red deer'] > self.redDeer_stocking_forecast:
                    number_to_subtract = int(random.randint(0,self.redDeer_stocking_forecast))
                    self.remove_herbivores(red_deer_agent, number_to_subtract)
            # Dec 2020
            if self.schedule.time == 190:
                results = self.datacollector.get_model_vars_dataframe()
                if results.iloc[190]['Longhorn cattle'] > self.cattle_stocking_forecast:
                    number_to_subtract = int(random.randint(0,self.cattle_stocking_forecast))
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                if results.iloc[190]['Fallow deer'] > self.fallowDeer_stocking_forecast:
                    number_to_subtract = int(random.randint(0,self.fallowDeer_stocking_forecast))
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                if results.iloc[190]['Red deer'] > self.redDeer_stocking_forecast:
                    number_to_subtract = int(random.randint(0,self.redDeer_stocking_forecast))
                    self.remove_herbivores(red_deer_agent, number_to_subtract)
                if results.iloc[190]['Tamworth pigs'] > self.tamworthPig_stocking_forecast:
                    number_to_subtract = int(random.randint(0,self.tamworthPig_stocking_forecast))
                    self.remove_pig(tamworth_pig_agent,number_to_subtract,0,0)
            # Jan 2021  
            if self.schedule.time == 191:
                results = self.datacollector.get_model_vars_dataframe()
                if results.iloc[191]['Longhorn cattle'] > self.cattle_stocking_forecast:
                    number_to_subtract = int(random.randint(0,self.cattle_stocking_forecast))
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                if results.iloc[191]['Fallow deer'] > self.fallowDeer_stocking_forecast:
                    number_to_subtract = int(random.randint(0,self.fallowDeer_stocking_forecast))
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                if results.iloc[191]['Red deer'] > self.redDeer_stocking_forecast:
                    number_to_subtract = int(random.randint(0,self.redDeer_stocking_forecast))
                    self.remove_herbivores(red_deer_agent, number_to_subtract)
                if results.iloc[191]['Tamworth pigs'] > self.tamworthPig_stocking_forecast:
                    number_to_subtract = int(random.randint(0,self.tamworthPig_stocking_forecast))
                    self.remove_pig(tamworth_pig_agent, number_to_subtract,0,0)
            # Feb 2021: cull them all back to stocking values
            if self.schedule.time == 192:
                results = self.datacollector.get_model_vars_dataframe()
                if results.iloc[192]['Longhorn cattle'] > self.cattle_stocking_forecast:
                    number_to_subtract = int(-self.cattle_stocking_forecast + results.iloc[192]['Longhorn cattle'])
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                if results.iloc[192]['Fallow deer'] > self.fallowDeer_stocking_forecast:
                    number_to_subtract = int(-self.fallowDeer_stocking_forecast + results.iloc[192]['Fallow deer'])
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                if results.iloc[192]['Red deer'] > self.redDeer_stocking_forecast:
                    number_to_subtract = int(-self.redDeer_stocking_forecast + results.iloc[192]['Red deer'])
                    self.remove_herbivores(red_deer_agent, number_to_subtract)
                if results.iloc[192]['Tamworth pigs'] > self.tamworthPig_stocking_forecast:
                    number_to_subtract = int(-self.tamworthPig_stocking_forecast + results.iloc[192]['Tamworth pigs'])
                    self.remove_pig(tamworth_pig_agent,int(number_to_subtract),0,0)




        # # no control only
        #     if self.schedule.time >= self.max_time:                                    
        #         self.running = False







                                                ### Forecasting and perturbation experiments ###
            
            # March 2021
            if self.schedule.time >= 193 and ((self.schedule.time % 12) + 1) == 1:


                                    # # # # # # # EXPERIMENT 1: Lower growth rates for a duration - do this every time step  # # # # # # # 

                # change to 1742
                if self.experiment_growth == True:
                    if self.schedule.time >= 193 and (self.schedule.time < (193 + self.duration)):
                        # collect my initial time
                        self.chance_reproduceSapling = self.exp_chance_reproduceSapling
                        if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                        self.chance_reproduceYoungScrub = self.exp_chance_reproduceYoungScrub
                        if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                        self.chance_regrowGrass = self.exp_chance_regrowGrass
                        if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    else: 
                        self.chance_reproduceSapling = self.original_chance_reproduceSapling
                        self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub
                        self.chance_regrowGrass = self.original_chance_regrowGrass




                                    # # # # # # # EXPERIMENT 3: gradually increase vegetation growth rates; 1% per year # # # # # # # 


                if self.experiment_linear_growth == True and self.schedule.time >= 193:
                    noise = self.tree_reduction[self.schedule.time-193]
                    # set saplings
                    self.chance_reproduceSapling = self.original_chance_reproduceSapling * noise 
                    if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                    if self.chance_reproduceSapling < 0: self.chance_reproduceSapling = 0
                    print("original", self.original_chance_reproduceSapling, "new", self.chance_reproduceSapling)

                    # young scrub
                    self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub * noise  
                    if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                    if self.chance_reproduceYoungScrub < 0: self.chance_reproduceYoungScrub = 0
                    # grass
                    self.chance_regrowGrass = self.original_chance_regrowGrass * noise
                    if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    if self.chance_regrowGrass < 0: self.chance_regrowGrass = 0

                    # drop woodland 
                    if self.schedule.time == 1392:
                        # how many woodland fields do we have? 
                        for field in self.fields:
                            if field.condition == "woodland": # if it's woodland, 50% of them lose 
                                if random.random() < 0.5:
                                    field.edibles["trees"] = 0
                                    field.edibles["saplings"] = 0

                                    
                                    
                                    # # # # # # # IS IT AT EQUILIBRIUM? # # # # # # # 



                # get the time-step of ten years previously; if it hasn't changed, stop
                results = self.datacollector.get_model_vars_dataframe() 
                ten_years_ago = self.schedule.time - 120

                # check once per year if it's at equilibrium or if roe deer explode 
                # print(results.iloc[-1]["Roe deer"])
                if self.schedule.time >= self.max_time:                                    
                
                # remove roe deer criteria for landscape graph
                # if self.experiment_growth == False and self.experiment_wood == False and self.experiment_linear_growth == False and self.schedule.time >= 6000:                                    
                    self.running = False


                # otherwise keep going
                else: 
                    # print("continuing:", self.schedule.time)
                    results = self.datacollector.get_model_vars_dataframe()
                    # first make sure that exmoor ponies are at their stocking density
                    if results.iloc[-1]['Exmoor pony'] < self.exmoor_stocking_forecast: # shouldn't have to subtract anything since they don't grow
                        number_to_add = int(self.exmoor_stocking_forecast - int(results.iloc[-1]['Exmoor pony']))
                        self.add_herbivores(exmoor_pony_agent, number_to_add, grazer_move)
                    # reset fallow deer values (they are culled)
                    if results.iloc[-1]['Fallow deer'] > self.fallowDeer_stocking_forecast:
                        number_to_subtract = random.randint(0,self.fallowDeer_stocking_forecast)
                        self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                    else:
                        number_to_add = self.fallowDeer_stocking_forecast - int(results.iloc[-1]['Fallow deer'])
                        self.add_herbivores(fallow_deer_agent, number_to_add,mixed_diet_move)
                    # reset red deer values  (they are culled)
                    if results.iloc[-1]['Red deer'] > self.redDeer_stocking_forecast:
                        number_to_subtract = random.randint(0,self.redDeer_stocking_forecast)
                        self.remove_herbivores(red_deer_agent, number_to_subtract)
                    else:
                        number_to_add = self.redDeer_stocking_forecast - int(results.iloc[-1]['Red deer'])
                        self.add_herbivores(red_deer_agent, number_to_add, mixed_diet_move)
                    # reset longhorn cattle values (they aren't culled this month)
                    if results.iloc[-1]['Longhorn cattle'] < self.cattle_stocking_forecast:
                        number_to_add = int(self.cattle_stocking_forecast - int(results.iloc[-1]['Longhorn cattle']))
                        self.add_herbivores(longhorn_cattle_agent, number_to_add, grazer_move)
                    # reset tamworth pig values (they aren't culled this month)
                    if results.iloc[-1]['Tamworth pigs'] < self.tamworthPig_stocking_forecast:
                        number_to_add = int(self.tamworthPig_stocking_forecast - int(results.iloc[-1]['Tamworth pigs']))
                        self.add_pig(tamworth_pig_agent, 0,number_to_add,0, random_move)
                    if self.introduce_euroBison == True:
                        if results.iloc[-1]['European bison'] > 50:
                            number_to_remove = int(-50 + int(results.iloc[-1]['European bison']))
                            self.remove_herbivores(european_bison_agent, number_to_remove)
                    if self.introduce_elk == True:
                        if results.iloc[-1]['European elk'] > 50:
                            number_to_remove = int(-50 + int(results.iloc[-1]['European elk']))
                            self.remove_herbivores(european_elk_agent, number_to_remove)



            # April 2021
            if self.schedule.time >= 193 and ((self.schedule.time % 12) + 1) == 2:
                if self.experiment_growth == True:
                    if self.schedule.time >= 193 and (self.schedule.time < (193 + self.duration)):
                        # collect my initial time
                        self.chance_reproduceSapling = self.exp_chance_reproduceSapling
                        if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                        self.chance_reproduceYoungScrub = self.exp_chance_reproduceYoungScrub
                        if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                        self.chance_regrowGrass = self.exp_chance_regrowGrass
                        if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    else: 
                        self.chance_reproduceSapling = self.original_chance_reproduceSapling
                        self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub
                        self.chance_regrowGrass = self.original_chance_regrowGrass
                if self.experiment_linear_growth == True and self.schedule.time >= 193:
                    noise = self.tree_reduction[self.schedule.time-193]
                    # set saplings
                    self.chance_reproduceSapling = self.original_chance_reproduceSapling * noise 
                    if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                    if self.chance_reproduceSapling < 0: self.chance_reproduceSapling = 0
                    # young scrub
                    self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub * noise  
                    if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                    if self.chance_reproduceYoungScrub < 0: self.chance_reproduceYoungScrub = 0
                    # grass
                    self.chance_regrowGrass = self.original_chance_regrowGrass * noise
                    if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    if self.chance_regrowGrass < 0: self.chance_regrowGrass = 0

                results = self.datacollector.get_model_vars_dataframe()
                if results.iloc[-1]['Fallow deer'] > self.fallowDeer_stocking_forecast:
                    number_to_subtract = int(-self.fallowDeer_stocking_forecast + int(results.iloc[-1]['Fallow deer']))
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                if results.iloc[-1]['Red deer'] > self.redDeer_stocking_forecast:
                    number_to_subtract = int(-self.redDeer_stocking_forecast + int(results.iloc[-1]['Red deer']))
                    self.remove_herbivores(red_deer_agent, number_to_subtract)
                

            # May 2021
            if self.schedule.time >= 193 and ((self.schedule.time % 12) + 1) == 3:
                if self.experiment_growth == True:
                    if self.schedule.time >= 193 and (self.schedule.time < (193 + self.duration)):
                        # collect my initial time
                        self.chance_reproduceSapling = self.exp_chance_reproduceSapling
                        if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                        self.chance_reproduceYoungScrub = self.exp_chance_reproduceYoungScrub
                        if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                        self.chance_regrowGrass = self.exp_chance_regrowGrass
                        if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    else: 
                        self.chance_reproduceSapling = self.original_chance_reproduceSapling
                        self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub
                        self.chance_regrowGrass = self.original_chance_regrowGrass
                if self.experiment_linear_growth == True and self.schedule.time >= 193:
                    noise = self.tree_reduction[self.schedule.time-193]
                    # set saplings
                    self.chance_reproduceSapling = self.original_chance_reproduceSapling * noise 
                    if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                    if self.chance_reproduceSapling < 0: self.chance_reproduceSapling = 0
                    # young scrub
                    self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub * noise  
                    if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                    if self.chance_reproduceYoungScrub < 0: self.chance_reproduceYoungScrub = 0
                    # grass
                    self.chance_regrowGrass = self.original_chance_regrowGrass * noise
                    if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    if self.chance_regrowGrass < 0: self.chance_regrowGrass = 0
                
                results = self.datacollector.get_model_vars_dataframe()
                if results.iloc[-1]['Tamworth pigs'] > self.tamworthPig_stocking_forecast:
                    number_to_subtract = random.randint(0,self.tamworthPig_stocking_forecast)
                    self.remove_pig(tamworth_pig_agent, number_to_subtract,0,0)


            # June 2021
            if self.schedule.time >= 193 and ((self.schedule.time % 12) + 1) == 4:
                if self.experiment_growth == True:
                    if self.schedule.time >= 193 and (self.schedule.time < (193 + self.duration)):
                        # collect my initial time
                        self.chance_reproduceSapling = self.exp_chance_reproduceSapling
                        if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                        self.chance_reproduceYoungScrub = self.exp_chance_reproduceYoungScrub
                        if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                        self.chance_regrowGrass = self.exp_chance_regrowGrass
                        if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    else: 
                        self.chance_reproduceSapling = self.original_chance_reproduceSapling
                        self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub
                        self.chance_regrowGrass = self.original_chance_regrowGrass
                if self.experiment_linear_growth == True and self.schedule.time >= 193:
                    noise = self.tree_reduction[self.schedule.time-193]
                    # set saplings
                    self.chance_reproduceSapling = self.original_chance_reproduceSapling * noise 
                    if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                    if self.chance_reproduceSapling < 0: self.chance_reproduceSapling = 0
                    # young scrub
                    self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub * noise  
                    if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                    if self.chance_reproduceYoungScrub < 0: self.chance_reproduceYoungScrub = 0
                    # grass
                    self.chance_regrowGrass = self.original_chance_regrowGrass * noise
                    if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    if self.chance_regrowGrass < 0: self.chance_regrowGrass = 0

                
                results = self.datacollector.get_model_vars_dataframe()
                if results.iloc[-1]['Longhorn cattle'] >= self.cattle_stocking_forecast:
                    number_to_subtract = random.randint(0,self.cattle_stocking_forecast)
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)


            # July 2021
            if self.schedule.time >= 193 and ((self.schedule.time % 12) + 1) == 5:
                if self.experiment_growth == True:
                    if self.schedule.time >= 193 and (self.schedule.time < (193 + self.duration)):
                        # collect my initial time
                        self.chance_reproduceSapling = self.exp_chance_reproduceSapling
                        if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                        self.chance_reproduceYoungScrub = self.exp_chance_reproduceYoungScrub
                        if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                        self.chance_regrowGrass = self.exp_chance_regrowGrass
                        if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    else: 
                        self.chance_reproduceSapling = self.original_chance_reproduceSapling
                        self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub
                        self.chance_regrowGrass = self.original_chance_regrowGrass
                if self.experiment_linear_growth == True and self.schedule.time >= 193:
                    noise = self.tree_reduction[self.schedule.time-193]
                    # set saplings
                    self.chance_reproduceSapling = self.original_chance_reproduceSapling * noise 
                    if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                    if self.chance_reproduceSapling < 0: self.chance_reproduceSapling = 0
                    # young scrub
                    self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub * noise  
                    if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                    if self.chance_reproduceYoungScrub < 0: self.chance_reproduceYoungScrub = 0
                    # grass
                    self.chance_regrowGrass = self.original_chance_regrowGrass * noise
                    if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    if self.chance_regrowGrass < 0: self.chance_regrowGrass = 0
                
                
                results = self.datacollector.get_model_vars_dataframe()
                if results.iloc[-1]['Longhorn cattle'] >= self.cattle_stocking_forecast:
                    number_to_subtract = random.randint(0,self.cattle_stocking_forecast)
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)


            # August 2021
            if self.schedule.time >= 193 and ((self.schedule.time % 12) + 1) == 6:
                if self.experiment_growth == True:
                    if self.schedule.time >= 193 and (self.schedule.time < (193 + self.duration)):
                        # collect my initial time
                        self.chance_reproduceSapling = self.exp_chance_reproduceSapling
                        if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                        self.chance_reproduceYoungScrub = self.exp_chance_reproduceYoungScrub
                        if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                        self.chance_regrowGrass = self.exp_chance_regrowGrass
                        if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    else: 
                        self.chance_reproduceSapling = self.original_chance_reproduceSapling
                        self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub
                        self.chance_regrowGrass = self.original_chance_regrowGrass
                if self.experiment_linear_growth == True and self.schedule.time >= 193:
                    noise = self.tree_reduction[self.schedule.time-193]
                    # set saplings
                    self.chance_reproduceSapling = self.original_chance_reproduceSapling * noise 
                    if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                    if self.chance_reproduceSapling < 0: self.chance_reproduceSapling = 0
                    # young scrub
                    self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub * noise  
                    if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                    if self.chance_reproduceYoungScrub < 0: self.chance_reproduceYoungScrub = 0
                    # grass
                    self.chance_regrowGrass = self.original_chance_regrowGrass * noise
                    if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    if self.chance_regrowGrass < 0: self.chance_regrowGrass = 0
                
                results = self.datacollector.get_model_vars_dataframe()
                if results.iloc[-1]['Longhorn cattle'] > self.cattle_stocking_forecast:
                    number_to_subtract = random.randint(0,self.cattle_stocking_forecast)
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                if results.iloc[-1]['Fallow deer'] > self.fallowDeer_stocking_forecast:
                    number_to_subtract = random.randint(0,self.fallowDeer_stocking_forecast)
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                if results.iloc[-1]['Red deer'] > self.redDeer_stocking_forecast:
                    number_to_subtract = random.randint(0,self.redDeer_stocking_forecast)
                    self.remove_herbivores(red_deer_agent, number_to_subtract)
 

            # Sept 2021
            if self.schedule.time >= 193 and ((self.schedule.time % 12) + 1) == 7:
                if self.experiment_growth == True:
                    if self.schedule.time >= 193 and (self.schedule.time < (193 + self.duration)):
                        # collect my initial time
                        self.chance_reproduceSapling = self.exp_chance_reproduceSapling
                        if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                        self.chance_reproduceYoungScrub = self.exp_chance_reproduceYoungScrub
                        if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                        self.chance_regrowGrass = self.exp_chance_regrowGrass
                        if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    else: 
                        self.chance_reproduceSapling = self.original_chance_reproduceSapling
                        self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub
                        self.chance_regrowGrass = self.original_chance_regrowGrass
                results = self.datacollector.get_model_vars_dataframe()
                if results.iloc[-1]['Longhorn cattle'] > self.cattle_stocking_forecast:
                    number_to_subtract = random.randint(0,self.cattle_stocking_forecast)
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                if results.iloc[-1]['Fallow deer'] > self.fallowDeer_stocking_forecast:
                    number_to_subtract = random.randint(0,self.fallowDeer_stocking_forecast)
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)


            # Oct 2021
            if self.schedule.time >= 193 and ((self.schedule.time % 12) + 1) == 8:
                if self.experiment_growth == True:
                    if self.schedule.time >= 193 and (self.schedule.time < (193 + self.duration)):
                        # collect my initial time
                        self.chance_reproduceSapling = self.exp_chance_reproduceSapling
                        if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                        self.chance_reproduceYoungScrub = self.exp_chance_reproduceYoungScrub
                        if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                        self.chance_regrowGrass = self.exp_chance_regrowGrass
                        if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    else: 
                        self.chance_reproduceSapling = self.original_chance_reproduceSapling
                        self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub
                        self.chance_regrowGrass = self.original_chance_regrowGrass
                if self.experiment_linear_growth == True and self.schedule.time >= 193:
                    noise = self.tree_reduction[self.schedule.time-193]
                    # set saplings
                    self.chance_reproduceSapling = self.original_chance_reproduceSapling * noise 
                    if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                    if self.chance_reproduceSapling < 0: self.chance_reproduceSapling = 0
                    # young scrub
                    self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub * noise  
                    if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                    if self.chance_reproduceYoungScrub < 0: self.chance_reproduceYoungScrub = 0
                    # grass
                    self.chance_regrowGrass = self.original_chance_regrowGrass * noise
                    if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    if self.chance_regrowGrass < 0: self.chance_regrowGrass = 0
                
                results = self.datacollector.get_model_vars_dataframe()
                if results.iloc[-1]['Longhorn cattle'] > self.cattle_stocking_forecast:
                    number_to_subtract = random.randint(0,self.cattle_stocking_forecast)
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)


            # Nov 2021
            if self.schedule.time >= 193 and ((self.schedule.time % 12) + 1) == 9:
                if self.experiment_growth == True:
                    if self.schedule.time >= 193 and (self.schedule.time < (193 + self.duration)):
                        # collect my initial time
                        self.chance_reproduceSapling = self.exp_chance_reproduceSapling
                        if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                        self.chance_reproduceYoungScrub = self.exp_chance_reproduceYoungScrub
                        if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                        self.chance_regrowGrass = self.exp_chance_regrowGrass
                        if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    else: 
                        self.chance_reproduceSapling = self.original_chance_reproduceSapling
                        self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub
                        self.chance_regrowGrass = self.original_chance_regrowGrass
                if self.experiment_linear_growth == True and self.schedule.time >= 193:
                    noise = self.tree_reduction[self.schedule.time-193]
                    # set saplings
                    self.chance_reproduceSapling = self.original_chance_reproduceSapling * noise 
                    if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                    if self.chance_reproduceSapling < 0: self.chance_reproduceSapling = 0
                    # young scrub
                    self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub * noise  
                    if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                    if self.chance_reproduceYoungScrub < 0: self.chance_reproduceYoungScrub = 0
                    # grass
                    self.chance_regrowGrass = self.original_chance_regrowGrass * noise
                    if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    if self.chance_regrowGrass < 0: self.chance_regrowGrass = 0
                
                results = self.datacollector.get_model_vars_dataframe()
                if results.iloc[-1]['Longhorn cattle'] > self.cattle_stocking_forecast:
                    number_to_subtract = random.randint(0,self.cattle_stocking_forecast)
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                if results.iloc[-1]['Fallow deer'] > self.fallowDeer_stocking_forecast:
                    number_to_subtract = random.randint(0,self.fallowDeer_stocking_forecast)
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                if results.iloc[-1]['Red deer'] > self.redDeer_stocking_forecast:
                    number_to_subtract = random.randint(0,self.redDeer_stocking_forecast)
                    self.remove_herbivores(red_deer_agent, number_to_subtract)


            # Dec 2021
            if self.schedule.time >= 193 and ((self.schedule.time % 12) + 1) == 10:
                if self.experiment_growth == True:
                    if self.schedule.time >= 193 and (self.schedule.time < (193 + self.duration)):
                        # collect my initial time
                        self.chance_reproduceSapling = self.exp_chance_reproduceSapling
                        if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                        self.chance_reproduceYoungScrub = self.exp_chance_reproduceYoungScrub
                        if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                        self.chance_regrowGrass = self.exp_chance_regrowGrass
                        if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    else: 
                        self.chance_reproduceSapling = self.original_chance_reproduceSapling
                        self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub
                        self.chance_regrowGrass = self.original_chance_regrowGrass
                if self.experiment_linear_growth == True and self.schedule.time >= 193:
                    noise = self.tree_reduction[self.schedule.time-193]
                    # set saplings
                    self.chance_reproduceSapling = self.original_chance_reproduceSapling * noise 
                    if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                    if self.chance_reproduceSapling < 0: self.chance_reproduceSapling = 0
                    # young scrub
                    self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub * noise  
                    if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                    if self.chance_reproduceYoungScrub < 0: self.chance_reproduceYoungScrub = 0
                    # grass
                    self.chance_regrowGrass = self.original_chance_regrowGrass * noise
                    if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    if self.chance_regrowGrass < 0: self.chance_regrowGrass = 0
                
                results = self.datacollector.get_model_vars_dataframe()
                if results.iloc[-1]['Longhorn cattle'] > self.cattle_stocking_forecast:
                    number_to_subtract = random.randint(0,self.cattle_stocking_forecast)
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                if results.iloc[-1]['Fallow deer'] > self.fallowDeer_stocking_forecast:
                    number_to_subtract = random.randint(0,self.fallowDeer_stocking_forecast)
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                if results.iloc[-1]['Red deer'] > self.redDeer_stocking_forecast:
                    number_to_subtract = random.randint(0,self.redDeer_stocking_forecast)
                    self.remove_herbivores(red_deer_agent, number_to_subtract)
                if results.iloc[-1]['Tamworth pigs'] > self.tamworthPig_stocking_forecast:
                    number_to_subtract = random.randint(0,self.tamworthPig_stocking_forecast)
                    self.remove_pig(tamworth_pig_agent,number_to_subtract,0,0)
                # add boars
                self.add_pig(tamworth_pig_agent, 0, 0, 1, random_move)


            # Jan 2022
            if self.schedule.time >= 193 and ((self.schedule.time % 12) + 1) == 11:
                if self.experiment_growth == True:
                    if self.schedule.time >= 193 and (self.schedule.time < (193 + self.duration)):
                        # collect my initial time
                        self.chance_reproduceSapling = self.exp_chance_reproduceSapling
                        if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                        self.chance_reproduceYoungScrub = self.exp_chance_reproduceYoungScrub
                        if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                        self.chance_regrowGrass = self.exp_chance_regrowGrass
                        if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    else: 
                        self.chance_reproduceSapling = self.original_chance_reproduceSapling
                        self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub
                        self.chance_regrowGrass = self.original_chance_regrowGrass
                if self.experiment_linear_growth == True and self.schedule.time >= 193:
                    noise = self.tree_reduction[self.schedule.time-193]
                    # set saplings
                    self.chance_reproduceSapling = self.original_chance_reproduceSapling * noise 
                    if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                    if self.chance_reproduceSapling < 0: self.chance_reproduceSapling = 0
                    # young scrub
                    self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub * noise  
                    if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                    if self.chance_reproduceYoungScrub < 0: self.chance_reproduceYoungScrub = 0
                    # grass
                    self.chance_regrowGrass = self.original_chance_regrowGrass * noise
                    if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    if self.chance_regrowGrass < 0: self.chance_regrowGrass = 0
                
                results = self.datacollector.get_model_vars_dataframe()
                if results.iloc[-1]['Longhorn cattle'] >self.cattle_stocking_forecast:
                    number_to_subtract = random.randint(0,self.cattle_stocking_forecast)
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                if results.iloc[-1]['Fallow deer'] > self.fallowDeer_stocking_forecast:
                    number_to_subtract = random.randint(0,self.fallowDeer_stocking_forecast)
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                if results.iloc[-1]['Red deer'] > self.redDeer_stocking_forecast:
                    number_to_subtract = random.randint(0,self.redDeer_stocking_forecast)
                    self.remove_herbivores(red_deer_agent, number_to_subtract)
                if results.iloc[-1]['Tamworth pigs'] > self.tamworthPig_stocking_forecast:
                    number_to_subtract = random.randint(0,self.tamworthPig_stocking_forecast)
                    self.remove_pig(tamworth_pig_agent,number_to_subtract,0,1)


            # Feb 2022: cull them all back to stocking values
            if self.schedule.time >= 193 and ((self.schedule.time % 12) + 1) == 12:
                if self.experiment_growth == True:
                    if self.schedule.time >= 193 and (self.schedule.time < (193 + self.duration)):
                        # collect my initial time
                        self.chance_reproduceSapling = self.exp_chance_reproduceSapling
                        if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                        self.chance_reproduceYoungScrub = self.exp_chance_reproduceYoungScrub
                        if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                        self.chance_regrowGrass = self.exp_chance_regrowGrass
                        if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    else: 
                        self.chance_reproduceSapling = self.original_chance_reproduceSapling
                        self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub
                        self.chance_regrowGrass = self.original_chance_regrowGrass
                if self.experiment_linear_growth == True and self.schedule.time >= 193:
                    noise = self.tree_reduction[self.schedule.time-193]
                    # set saplings
                    self.chance_reproduceSapling = self.original_chance_reproduceSapling * noise 
                    if self.chance_reproduceSapling > 1: self.chance_reproduceSapling = 1
                    if self.chance_reproduceSapling < 0: self.chance_reproduceSapling = 0
                    # young scrub
                    self.chance_reproduceYoungScrub = self.original_chance_reproduceYoungScrub * noise  
                    if self.chance_reproduceYoungScrub > 1: self.chance_reproduceYoungScrub = 1
                    if self.chance_reproduceYoungScrub < 0: self.chance_reproduceYoungScrub = 0
                    # grass
                    self.chance_regrowGrass = self.original_chance_regrowGrass * noise
                    if self.chance_regrowGrass > 1: self.chance_regrowGrass = 1
                    if self.chance_regrowGrass < 0: self.chance_regrowGrass = 0
                    
                
                results = self.datacollector.get_model_vars_dataframe()
                if results.iloc[-1]['Longhorn cattle'] > self.cattle_stocking_forecast:
                    number_to_subtract = int(-self.cattle_stocking_forecast + int(results.iloc[-1]['Longhorn cattle']))
                    self.remove_herbivores(longhorn_cattle_agent, number_to_subtract)
                if results.iloc[-1]['Fallow deer'] > self.fallowDeer_stocking_forecast:
                    number_to_subtract = int(-self.fallowDeer_stocking_forecast + int(results.iloc[-1]['Fallow deer']))
                    self.remove_herbivores(fallow_deer_agent, number_to_subtract)
                if results.iloc[-1]['Red deer'] > self.redDeer_stocking_forecast:
                    number_to_subtract = int(-self.redDeer_stocking_forecast + int(results.iloc[-1]['Red deer']))
                    self.remove_herbivores(red_deer_agent, number_to_subtract)
                if results.iloc[-1]['Tamworth pigs'] > self.tamworthPig_stocking_forecast:
                    number_to_subtract = int(-self.tamworthPig_stocking_forecast + int(results.iloc[-1]['Tamworth pigs']))
                    self.remove_pig(tamworth_pig_agent,number_to_subtract,0,0)  
        
        else:
            results = self.datacollector.get_model_vars_dataframe()
            print("time", self.schedule.time, "roe deer", results.iloc[-1]["Roe deer"], "scrub", results.iloc[-1]["Thorny Scrub"], "wood", results.iloc[-1]["Woodland"], "grass", results.iloc[-1]["Grassland"])

                  



    def run_model(self): 
        while self.running == True: 
            self.step()

        return self.datacollector.get_model_vars_dataframe()