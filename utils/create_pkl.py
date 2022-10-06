"""
Reads glove file, graph, and visible object map, and generates pickle files storing test episodes for each scene.

Returns
.pkl files inside test_val_split folder - contains test episodes for each scene of iThor
"""
import json
import h5py
import pickle
import random
random.seed(10)
import numpy as np
from ai2thor import controller, platform
import sys
sys.path.append('..')
from datasets.offline_controller_with_small_rotation import ThorAgentState
from datasets.constants import KITCHEN_OBJECT_CLASS_LIST, LIVING_ROOM_OBJECT_CLASS_LIST, BEDROOM_OBJECT_CLASS_LIST, BATHROOM_OBJECT_CLASS_LIST
from kg_prep.misc import ensuredirs
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--offline_dir', type=str, required=True)
args = parser.parse_args()

controller = controller.Controller(platform=platform.CloudRendering)
glove_embeddings = h5py.File("../data/thor_glove/glove_thorv3_300.hdf5", "r")

# Allowed rotations.
all_rotations = [0, 45, 90, 135, 180, 225, 270, 315]
# Allowed horizons.
all_horizons = [0, 30]
rooms = [['Kitchen',0,KITCHEN_OBJECT_CLASS_LIST],['Living_Room',200,LIVING_ROOM_OBJECT_CLASS_LIST],['Bedroom',300,BEDROOM_OBJECT_CLASS_LIST],['Bathroom',400,BATHROOM_OBJECT_CLASS_LIST]]
for k in range(len(rooms)):#len(rooms)
    room = rooms[k][0]
    room_ind = rooms[k][1]
    target_objs = rooms[k][2]
    all_episodes = []
    for i in tqdm(range(20,30), desc=room):    #for each scene
        for _ in range(200):    #run 200 times
            controller.reset('FloorPlan'+str(1+i+room_ind))
            graph = json.load(open("{}/{}/graph.json".format(args.offline_dir,'FloorPlan'+str(1+i+room_ind)), "r"))
            graph_positions = graph["nodes"]
            visible_objects = json.load(open("{}/{}/visible_object_map.json".format(args.offline_dir,'FloorPlan'+str(1+i+room_ind)), "r"))
            id_list = list(visible_objects.keys())
            type_lst = [item.split("|")[0] for item in id_list]
            while True:
                sampled_obj = random.choice(target_objs)
                if sampled_obj in type_lst:
                    indices = [i for i, x in enumerate(type_lst) if x == sampled_obj]
                    task_data = []
                    for idx in indices:
                        task_data.append(id_list[idx])
                    break
            episode = {}
            episode['room'] = room
            episode['scene'] = 'FloorPlan'+str(1+i+room_ind)
            episode['goal_object_type'] = sampled_obj
            episode['task_data'] = task_data
            
            sampled_pos = random.choice(graph_positions)
            x,z,rotation,horizon = sampled_pos['id'].split("|")
            y = 0.8696960806846619
            episode['state'] = ThorAgentState(float(x), y, float(z), float(rotation), float(horizon))
            episode['glove_embedding'] = np.array(glove_embeddings[sampled_obj])
            all_episodes.append(episode)

    random.shuffle(all_episodes)
    evaluated_episodes = all_episodes[:250]
    all_rooms = []
    from collections import Counter
    for episode in evaluated_episodes:
        all_rooms.append(episode['scene'])

    print(Counter(all_rooms))
    with open(ensuredirs('../test_val_split/{}_test.pkl'.format(room.lower())), 'wb') as f:
        pickle.dump(all_episodes, f)