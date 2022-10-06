import h5py
import numpy as np
import re

embedding_model = {}
f = open('glove.840B.300d.txt', "r", encoding="utf8")
print("Loading glove...")
for line in f:
    values = line.split()
    word = ''.join(values[:-300])
    coefs = np.asarray(values[-300:], dtype='float32')
    embedding_model[word] = coefs
f.close()
print("Loaded glove!")

object_fn = "thor_v3_objects.txt"
with open(object_fn) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

object_glove_dict = {}
for object in lines:
    if object in embedding_model.keys():
        object_glove_dict[object] = embedding_model[object]
    else:
        sublist = re.findall('[a-zA-Z][^A-Z]*', object)
        glove_list = []
        for sub_obj in sublist:
            glove_list.append(embedding_model[sub_obj])
        glove_list = np.array(glove_list)
        object_glove_dict[object] = np.mean(glove_list,axis=0)

h = h5py.File('data/thor_glove/glove_thorv3_300.hdf5', "w")
for k, v in object_glove_dict.items():
    h.create_dataset(k, data=v)