import os
import json
import jsonlines
import h5py
import networkx as nx
import math
import numpy as np
import random

class ImageFeaturesDB(object):
    def __init__(self, img_ft_file, image_feat_size):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}
        with h5py.File(self.img_ft_file, 'r') as f:
            for key in f.keys():
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                self._feature_store[key] = ft 
        

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                self._feature_store[key] = ft
        return ft

class ImageFeaturesDB2(object):
    def __init__(self, img_ft_files, image_feat_size):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_files
        self._feature_stores = {}
        for name in img_ft_files:
            self._feature_stores[name] = {}
            with h5py.File(name, 'r') as f:
                for key in f.keys():
                    ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                    self._feature_stores[name][key] = ft 
        self.env_names = list(self._feature_stores.keys())
        print(self.env_names)
        

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        env_name = random.choice(self.env_names)
        if key in self._feature_stores[env_name]:
            ft = self._feature_stores[env_name][key]
        else:
            with h5py.File(env_name, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                self._feature_stores[env_name][key] = ft
        return ft

class ObjectFeatureDB2(object):
    def __init__(self, obj_ft_file, obj_feat_size):
        self.obj_feat_size = obj_feat_size
        self.obj_ft_file = obj_ft_file
        self._feature_store = {}
    
    def load_feature(self, scan, viewpoint, max_objects=None):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            obj_fts, obj_attrs = self._feature_store[key]
        else:
            with h5py.File(self.obj_ft_file, 'r') as f:
                obj_attrs = {}
                if key in f:
                    obj_fts = f[key][...][:, :self.obj_feat_size].astype(np.float32) 
                    for attr_key, attr_value in f[key].attrs.items():
                        obj_attrs[attr_key] = attr_value
                else:
                    obj_fts = np.zeros((0, self.obj_feat_size), dtype=np.float32)
            self._feature_store[key] = (obj_fts, obj_attrs)

        if max_objects is not None:
            obj_fts = obj_fts[:max_objects]
            # obj_attrs = {k: v[:max_objects] for k, v in obj_attrs.items()}
            obj_attrs['bboxs'] = obj_attrs['bboxs'][:max_objects]
        return obj_fts, obj_attrs
    
    def get_object_feature(
        self, scan, viewpoint, max_objects=None
    ):
        obj_fts, obj_attrs = self.load_feature(scan, viewpoint, max_objects=max_objects)
        obj_box_fts = np.zeros((len(obj_fts), 4), dtype=np.float32)
        feature_type = obj_attrs['feature_type']
        for k, bbox in enumerate(obj_attrs['bboxs']):
            x1, y1, x2, y2 = bbox
            if feature_type == 1:
                obj_box_fts[k, :] = [x1/1024, y1/512, x2/1024, y2/512]
            else:
                obj_box_fts[k, :] = [x1/3072, x2/512, y1/3072, y2/512]
        return obj_fts, obj_box_fts, feature_type

# def load_nav_graphs(connectivity_dir, scans):
#     ''' Load connectivity graph for each scan '''

#     def distance(pose1, pose2):
#         ''' Euclidean distance between two graph poses '''
#         return ((pose1['pose'][3]-pose2['pose'][3])**2\
#           + (pose1['pose'][7]-pose2['pose'][7])**2\
#           + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

#     graphs = {}
#     for scan in scans:
#         with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
#             G = nx.Graph()
#             positions = {}
#             data = json.load(f)
#             for i,item in enumerate(data):
#                 if item['included']:
#                     for j,conn in enumerate(item['unobstructed']):
#                         if conn and data[j]['included']:
#                             positions[item['image_id']] = np.array([item['pose'][3],
#                                     item['pose'][7], item['pose'][11]]);
#                             assert data[j]['unobstructed'][i], 'Graph should be undirected'
#                             G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
#             nx.set_node_attributes(G, values=positions, name='position')
#             graphs[scan] = G
#     return graphs

def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    scanvp_position = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
                scanvp = '%s_%s' % (scan, item['image_id'])
                if scanvp not in scanvp_position:
                    scanvp_position[scanvp] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]])
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs, scanvp_position

def new_simulator(connectivity_dir, scan_data_dir=None):
    import MatterSim

    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    return sim

def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading), math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)

def get_point_angle_feature(sim, angle_feat_size, baseViewId=0):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)

    for ix in range(36):
        if ix == 0:
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature

def get_all_point_angle_feature(sim, angle_feat_size):
    return [get_point_angle_feature(sim, angle_feat_size, baseViewId) for baseViewId in range(36)]

