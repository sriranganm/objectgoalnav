""" Contains the Episodes for Navigation. """
import random
import math

import torch

from datasets.constants import GOAL_SUCCESS_REWARD, STEP_PENALTY
from datasets.constants import DONE
from datasets.environment import Environment

from utils.net_util import gpuify, toFloatTensor
from utils.action_util import get_actions
from utils.net_util import gpuify
from .episode import Episode
from utils import flag_parser

import json
import pickle

c2p_prob = json.load(open("./data/c2p_prob.json"))
args = flag_parser.parse_arguments()

class BasicEpisode(Episode):
    """ Episode for Navigation. """

    def __init__(self, args, gpu_id, strict_done=False):
        super(BasicEpisode, self).__init__()

        self._env = None
        self.gpu_id = gpu_id
        self.strict_done = strict_done
        self.task_data = None
        self.glove_embedding = None
        self.actions = get_actions(args)
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self._last_action_embedding_idx = 0
        self.target_object = None
        self.prev_frame = None
        self.current_frame = None
        self.current_objs = None

        self.scene_states = []
        self.partial_reward = args.partial_reward
        self.seen_list = []
        self.seen_bb_size_dict = {}
        self.seen_bb_init_size_dict = {}
        self.seen_dist_dict = {}
        if args.eval:
            random.seed(args.seed)
        self.room = None

    @property
    def environment(self):
        return self._env

    @property
    def actions_list(self):
        return [{"action": a} for a in self.actions]

    def reset(self):
        self.done_count = 0
        self.duplicate_count = 0
        self._env.back_to_start()

    def state_for_agent(self):
        return self.environment.current_frame

    def objstate_for_agent(self):
        return self.environment.current_objs

    def current_agent_position(self):
        """ Get the current position of the agent in the scene. """
        return self.environment.current_agent_position

    def step(self, action_as_int):

        action = self.actions_list[action_as_int]

        if args.vis:
            print(action)

        if action["action"] != DONE:
            self.environment.step(action)
        else:
            self.done_count += 1

        reward, terminal, action_was_successful = self.judge(action)
        return reward, terminal, action_was_successful

    def judge(self, action):
        """ Judge the last event. """
        reward = STEP_PENALTY

        # Thresholding replaced with simple look up for efficiency.
        if self.environment.controller.state in self.scene_states:
            if action["action"] != DONE:
                if self.environment.last_action_success:
                    self.duplicate_count += 1
                else:
                    self.failed_action_count += 1
                # added partial reward
                if self.partial_reward == "sparse":
                    reward = self.get_partial_reward()
                elif self.partial_reward == "dense_bbox":
                    reward = self.get_partial_reward_dense_bbox()
                elif self.partial_reward == "dense_depth":
                    reward = self.get_partial_reward_dense_depth()
        else:
            self.scene_states.append(self.environment.controller.state)

        done = False

        if action["action"] == DONE:
            action_was_successful = False
            for id_ in self.task_data:
                if self.environment.object_is_visible(id_):
                    reward = GOAL_SUCCESS_REWARD
                    done = True
                    action_was_successful = True
                    if self.partial_reward is not None:
                        self.seen_list = []
                        self.seen_bb_size_dict = {}
                        self.seen_dist_dict = {}
                        reward += self.get_partial_reward()
                    break
            self.seen_list = []
            self.seen_bb_size_dict = {}
            self.seen_bb_init_size_dict = {}
            self.seen_dist_dict = {}

            if args.vis:
                print("Success:", action_was_successful)
        else:
            action_was_successful = self.environment.last_action_success

        return reward, done, action_was_successful

    # Set the target index.
    @property
    def target_object_index(self):
        """ Return the index which corresponds to the target object. """
        return self._target_object_index

    @target_object_index.setter
    def target_object_index(self, target_object_index):
        """ Set the target object by specifying the index. """
        self._target_object_index = gpuify(
            torch.LongTensor([target_object_index]), self.gpu_id
        )

    def get_partial_reward(self):
        """ get partial reward if parent object is seen for the first time"""
        reward = STEP_PENALTY
        reward_dict = {}
        if self.target_parents is not None:
            for parent_type in self.target_parents:
                parent_ids = self.environment.find_id(parent_type)
                for parent_id in parent_ids:
                    if self.environment.object_is_visible(parent_id) and parent_id not in self.seen_list:
                        reward_dict[parent_id] = self.target_parents[parent_type]
        if len(reward_dict) != 0:
            v = list(reward_dict.values())
            k = list(reward_dict.keys())
            reward = max(v)           #pick one with greatest reward if multiple in scene
            self.seen_list.append(k[v.index(reward)])
        return reward

    def get_partial_reward_dense_bbox(self):
        """ get partial reward if parent object is seen for the first time"""
        reward = STEP_PENALTY
        reward_dict = {}
        k_dict = {}
        if self.target_object is not None:
            target_ids = self.environment.find_id(self.target_object)
            target_bb_size = self.environment.get_object_bb_size(target_ids[0])
            if target_bb_size > self.seen_bb_size_dict.get(target_ids[0], 0.0):
                self.seen_bb_init_size_dict.setdefault(target_ids[0], target_bb_size)
                self.seen_bb_size_dict[target_ids[0]] = target_bb_size
                reward = GOAL_SUCCESS_REWARD*(1-math.sqrt(self.seen_bb_init_size_dict.get(target_ids[0])/target_bb_size))
                print("Target ", self.target_object, target_ids[0], target_bb_size, reward)
                return reward

        if self.target_parents is not None:
            for parent_type in self.target_parents:
                parent_ids = self.environment.find_id(parent_type)
                for parent_id in parent_ids:
                    parent_bb_size = self.environment.get_object_bb_size(parent_id)
                    if parent_id not in self.seen_list and parent_bb_size > self.seen_bb_size_dict.get(parent_id, 0.0):
                        self.seen_bb_init_size_dict.setdefault(parent_id, parent_bb_size)
                        self.seen_bb_size_dict[parent_id] = parent_bb_size
                        k_dict[parent_id] = 1- math.sqrt(self.seen_bb_init_size_dict.get(parent_id)/parent_bb_size)
                        reward_dict[parent_id] = self.target_parents[parent_type]/0.1
        if len(reward_dict) != 0:
            v = list(reward_dict.values())
            k = list(reward_dict.keys())
            reward = max(v)           #pick one with greatest reward if multiple in scene
            max_reward = max(v)
            best_parent_obj = k[v.index(reward)]
            k_factor = k_dict[best_parent_obj]
            if (self.environment.object_is_visible(best_parent_obj)):
                self.seen_list.append(best_parent_obj)
            else:
                reward = max_reward*k_factor
                print("Parent", best_parent_obj, reward)
        return reward

    def get_partial_reward_dense_depth(self):
        """ get partial reward if parent object is seen for the first time"""
        reward = STEP_PENALTY
        reward_dict = {}
        distance_dict = {}
        if self.target_object is not None:
            target_ids = self.environment.find_id(self.target_object)
            target_dist = self.environment.get_object_dist(target_ids[0])
            if target_dist < self.seen_dist_dict.get(target_ids[0], math.inf):
                reward = GOAL_SUCCESS_REWARD*max(((-0.15*(target_dist-1))+1), 0.0)
                self.seen_dist_dict[target_ids[0]] = target_dist
                print(self.target_object, target_ids[0], reward)
                return reward

        if self.target_parents is not None:
            for parent_type in self.target_parents:
                parent_ids = self.environment.find_id(parent_type)
                for parent_id in parent_ids:
                    parent_dist = self.environment.get_object_dist(parent_id)
                    if parent_id not in self.seen_list and parent_dist < self.seen_dist_dict.get(parent_id, math.inf):
                        distance_dict[parent_id] = parent_dist
                        reward_dict[parent_id] = self.target_parents[parent_type]/0.1
        if len(reward_dict) != 0:
            v = list(reward_dict.values())
            k = list(reward_dict.keys())
            reward = max(v)           #pick one with greatest reward if multiple in scene
            max_reward = max(v)
            best_parent_obj = k[v.index(reward)]
            dist = distance_dict[best_parent_obj]
            if (dist <= 1.0):
                self.seen_list.append(best_parent_obj)
            else:
                reward = reward*max(((-0.15*(dist-1))+1), 0.0)
                self.seen_dist_dict[best_parent_obj] = dist
                print(best_parent_obj, reward)
        return reward

    def _new_episode(
        self, args, scenes, possible_targets, targets=None, room = None, keep_obj=False, glove=None
    ):
        """ New navigation episode. """
        scene = random.choice(scenes)
        self.room = room
        if self._env is None:
            self._env = Environment(
                offline_data_dir=args.offline_data_dir,
                use_offline_controller=True,
                grid_size=0.25,
                images_file_name=args.images_file_name,
                local_executable_path=args.local_executable_path,
            )
            self._env.start(scene)
        else:
            self._env.reset(scene)

        # Randomize the start location.
        start_state = self._env.randomize_agent_location()
        objects = self._env.all_objects()

        visible_objects = [obj.split("|")[0] for obj in objects]
        intersection = [obj for obj in visible_objects if obj in targets]

        self.task_data = []

        idx = random.randint(0, len(intersection) - 1)
        goal_object_type = intersection[idx]
        self.target_object = goal_object_type

        for id_ in objects:
            type_ = id_.split("|")[0]
            if goal_object_type == type_:
                self.task_data.append(id_)

        child_object = self.task_data[0].split("|")[0]
        #print('room is ', self.room)
        try:
            self.target_parents = c2p_prob[self.room][child_object]
        except:
            self.target_parents = None

        if args.verbose:
            print("Scene", scene, "Navigating towards:", goal_object_type)

        self.glove_embedding = None
        self.glove_embedding = toFloatTensor(
            glove.glove_embeddings[goal_object_type][:], self.gpu_id
        )

    def new_episode(
        self,
        args,
        scenes,
        possible_targets=None,
        targets=None,
        rooms=None,
        keep_obj=False,
        glove=None,
    ):
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self.prev_frame = None
        self.current_frame = None
        self.current_objs = None
        self._new_episode(args, scenes, possible_targets, targets, rooms, keep_obj, glove)
