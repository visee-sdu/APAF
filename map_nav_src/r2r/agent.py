import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict
import line_profiler

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from .agent_base import Seq2SeqAgent
from .eval_utils import cal_dtw

from models.graph_utils import GraphMap
from models.model import VLNBert, Critic
from models.ops import pad_tensors_wgrad
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)

# flag = True

def toggle_flag():
    global flag
    flag = not flag

class GMapNavAgent(Seq2SeqAgent):
    
    def _build_model(self):
        self.vln_bert = VLNBert(self.args).cuda()
        self.critic = Critic(self.args).cuda()
        # buffer
        self.scanvp_cands = {}

    def _language_variable(self, obs):
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]
        
        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask
        }

    def _panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens, batch_cand_vpids = [], []
        
        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)
            
            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts, 
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens, 
            'cand_vpids': batch_cand_vpids,
        }

    def _nav_gmap_variable(self, obs, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(obs)
        
        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []                
            for k in gmap.node_positions.keys():
                if self.args.act_visited_nodes:
                    if k == obs[i]['viewpoint']:
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:
                    if gmap.graph.visited(k):
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                if self.args.dataset == 'r2r' or self.args.dataset == 'rxr':
                    gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
                else:
                    gmap_visited_masks = [0] + [0] * len(visited_vpids) + [0] * len(unvisited_vpids)
                    for x in range(len(visited_vpids)):
                        if visited_vpids[x] == obs[i]['viewpoint']:
                            gmap_visited_masks[x+1] = 1
                            break       
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            
            gmap_object_embeds = [gmap.get_node_object_embed(vp) for vp in gmap_vpids[1:1+len(visited_vpids)]]
            for k in range(len(gmap_object_embeds)):
                gmap_img_embeds[k] += gmap_object_embeds[k] * self.args.object_weight
            
            if self.args.dataset == 'r2r'or self.args.dataset == 'rxr':
                gmap_action_embeds = [gmap.get_node_action_embed(vp) for vp in gmap_vpids[1+len(visited_vpids):]]
                for k in range(len(gmap_action_embeds)):
                    gmap_img_embeds[k+len(visited_vpids)] += gmap_action_embeds[k] * self.args.action_weight
            else:
                # gmap_action_embeds = [gmap.get_node_action_embed(vp) for vp in gmap_vpids[1:] if vp != obs[i]['viewpoint']]
                gmap_action_embeds = [gmap.get_node_action_embed(vp) if vp != obs[i]['viewpoint'] else None for vp in gmap_vpids[1:]]
                for k in range(len(gmap_action_embeds)):
                    if gmap_action_embeds[k] is not None:
                        gmap_img_embeds[k] += gmap_action_embeds[k] * self.args.action_weight
            
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )   # cuda

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i+1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds, 
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks, 
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
        }

    def _nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i], 
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp], 
                obs[i]['heading'], obs[i]['elevation']
            )                    
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': gen_seq_masks(view_lens+1),
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': [[None]+x for x in cand_vpids],
        }

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0    # Stop if arrived 
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                            dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                    + self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).cuda()

    def _teacher_action_r4r(
        self, obs, vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None
    ):
        """R4R is not the shortest path. The goal location can be visited nodes.
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if imitation_learning:
                    assert ob['viewpoint'] == ob['gt_path'][t]
                    if t == len(ob['gt_path']) - 1:
                        a[i] = 0    # stop
                    else:
                        goal_vp = ob['gt_path'][t + 1]
                        for j, vpid in enumerate(vpids[i]):
                            if goal_vp == vpid:
                                a[i] = j
                                break
                else:
                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = 0    # Stop if arrived 
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        for j, vpid in enumerate(vpids[i]):
                            if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                                if self.args.expert_policy == 'ndtw':
                                    dist = - cal_dtw(
                                        self.env.shortest_distances[scan], 
                                        sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][vpid][1:], 
                                        ob['gt_path'], 
                                        threshold=3.0
                                    )['nDTW']
                                elif self.args.expert_policy == 'spl':
                                    # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                                    dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                            + self.env.shortest_distances[scan][cur_vp][vpid]
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:            # None is the <stop> action
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    
    def _object_feature_variable(self, obs):
        batch_object_fts, batch_object_loc_fts = [], []
        batch_object_lens = []
        batch_feature_types = []
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            object_fts, object_loc_fts, feature_type = self.object_features.get_object_feature(scan, vp, self.args.max_objects)
            object_lens = len(object_fts)
            feature_type = torch.fill_(torch.zeros(len(object_fts)), feature_type).long()  # (n_objects)
            batch_object_fts.append(torch.from_numpy(object_fts))  # [(n_objects, dim_ft)]
            batch_object_loc_fts.append(torch.from_numpy(object_loc_fts))  # [(n_objects, 4)]
            batch_object_lens.append(object_lens)  # [n_objects]
            batch_feature_types.append(feature_type)  # [n_objects]
        
        # pad features to max_len
        batch_object_fts = pad_tensors(batch_object_fts).cuda()  # (batch_size, max_objects, dim_ft)
        batch_object_loc_fts = pad_tensors(batch_object_loc_fts).cuda()  # (batch_size, max_objects, 4)
        batch_feature_types = pad_tensors(batch_feature_types).cuda()  # (batch_size, max_objects)
        batch_object_lens = torch.LongTensor(batch_object_lens).cuda()  # (batch_size)
        return {
            'object_img_fts': batch_object_fts, 'object_loc_fts': batch_object_loc_fts, 'object_feature_types': batch_feature_types ,'object_lens': batch_object_lens
        }
    
    def _action_gmap_feature_variable(self, gmaps, obs):
        batch_action_fts, batch_action_lens = [], []
        for i, gmap in enumerate(gmaps):
            action_fts = []
            for vp in gmap.node_positions.keys():
                if self.args.dataset == 'r2r' or self.args.dataset == 'rxr':
                    if not gmap.graph.visited(vp):
                        action_fts.append(gmap.get_node_action_embed(vp))
                else:
                    if obs[i]['viewpoint'] != vp:
                        action_fts.append(gmap.get_node_action_embed(vp))
            if len(action_fts)!=0:
                action_fts = torch.stack(action_fts, 0)
            else:
                action_fts = torch.zeros(1, 768).cuda()
            batch_action_lens.append(len(action_fts))
            batch_action_fts.append(action_fts)
        batch_action_lens = torch.LongTensor(batch_action_lens).cuda()
        batch_action_fts = pad_tensors(batch_action_fts).cuda()
        batch_action_masks = gen_seq_masks(batch_action_lens).cuda()
        return {'action_fts': batch_action_fts, 'action_masks': batch_action_masks}
                
    
    def _calculate_heading_and_elevation(self, pointA, pointB, heading, evalation):
        x, y, z= pointA[0], pointA[1], pointA[2]
        x1, y1, z1 = pointB[0], pointB[1], pointB[2]
        
        delta_x = x1 - x
        delta_y = y1 - y
        delta_z = z1 - z

        theta_heading = int(np.degrees(np.arctan2(delta_x, delta_y)))
        rel_heading = (theta_heading - heading + 360) % 360
            
        d_xy = np.sqrt(delta_x**2 + delta_y**2)
        theta_elevation = int(np.degrees(np.arctan2(delta_z, d_xy)))
        rel_elevation = max(-30, min(theta_elevation - evalation, 29))
        return rel_heading, rel_elevation

    
    def _act_variable2(self, ob, gmap, i_vp): 
        batch_action_heading_ids = []
        batch_action_elevation_ids = []
        batch_action_lens = []
        nodes = []    
        if self.args.dataset == 'r4r':   
            for vp in gmap.node_positions.keys():
                if ob['viewpoint'] != vp:
                    path = gmap.graph.path(i_vp, vp)
                    action_heading_ids = []
                    action_elevation_ids = []
                    current_vp = i_vp
                    for node in path: 
                        viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], current_vp)][node]
                        heading = (viewidx % 12) * 30
                        elevation = (viewidx // 12 - 1) * 30
                        pointA = self.env.scanvp_position['%s_%s'%(ob['scan'], current_vp)]
                        pointB = self.env.scanvp_position['%s_%s'%(ob['scan'], node)]
                        rel_heading, rel_elevation = self._calculate_heading_and_elevation(pointA, pointB, heading, elevation)
                        action_heading_ids.append(rel_heading)
                        action_elevation_ids.append(rel_elevation + 30)
                        current_vp = node
                    batch_action_lens.append(len(action_heading_ids))
                    batch_action_heading_ids.append(torch.LongTensor(action_heading_ids))
                    batch_action_elevation_ids.append(torch.LongTensor(action_elevation_ids))
                    nodes.append(vp)
        else:
            for vp in gmap.node_positions.keys():
                if not gmap.graph.visited(vp):
                    path = gmap.graph.path(i_vp, vp)
                    action_heading_ids = []
                    action_elevation_ids = []
                    current_vp = i_vp
                    for node in path: 
                        viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], current_vp)][node]
                        heading = (viewidx % 12) * 30
                        elevation = (viewidx // 12 - 1) * 30
                        pointA = self.env.scanvp_position['%s_%s'%(ob['scan'], current_vp)]
                        pointB = self.env.scanvp_position['%s_%s'%(ob['scan'], node)]
                        rel_heading, rel_elevation = self._calculate_heading_and_elevation(pointA, pointB, heading, elevation)
                        action_heading_ids.append(rel_heading)
                        action_elevation_ids.append(rel_elevation + 30)
                        current_vp = node
                    batch_action_lens.append(len(action_heading_ids))
                    batch_action_heading_ids.append(torch.LongTensor(action_heading_ids))
                    batch_action_elevation_ids.append(torch.LongTensor(action_elevation_ids))
                    nodes.append(vp)
        if batch_action_heading_ids:
            batch_action_lens = torch.LongTensor(batch_action_lens).cuda()
            batch_action_heading_ids = pad_tensors(batch_action_heading_ids).cuda()
            batch_action_elevation_ids = pad_tensors(batch_action_elevation_ids).cuda()
        return {'action_heading_ids': batch_action_heading_ids, 'action_elevation_ids': batch_action_elevation_ids, 'action_lens': batch_action_lens}, nodes

    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        self._update_scanvp_cands(obs)

        batch_size = len(obs)
        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)
            

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
        } for ob in obs]

        # Language input: txt_ids, txt_masks
        language_inputs = self._language_variable(obs)
        
        txt_embeds = self.vln_bert('language', language_inputs)
    
        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # Init the logs
        masks = []
        entropys = []
        ml_loss = 0.     
        for t in range(self.args.max_action_len):
            
            
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # graph representation
            pano_inputs = self._panorama_feature_variable(obs)
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                            torch.sum(pano_masks, 1, keepdim=True)

            object_inputs = self._object_feature_variable(obs)
            object_embed, object_masks = self.vln_bert('object', object_inputs)
            avg_object_embeds = torch.sum(object_embed * object_masks.unsqueeze(2), 1) / \
                                torch.sum(object_masks, 1, keepdim=True)
            

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    gmap.update_node_object_embed(i_vp, avg_object_embeds[i])
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])
                    
                    act_inputs, nodes = self._act_variable2(obs[i], gmap, i_vp)
                    if len(nodes) > 0:
                        action_embeds, action_masks = self.vln_bert('action', act_inputs)
                        action_embeds = torch.sum(action_embeds * action_masks.unsqueeze(2), 1) / \
                                        torch.sum(action_masks, 1, keepdim=True)
                        for k, node in enumerate(nodes):
                            gmap.update_node_action_embed(node, action_embeds[k])

            action_inputs = self._action_gmap_feature_variable(gmaps, obs)
            action_embed = action_inputs['action_fts']
            action_mask = action_inputs['action_masks']

            txt_features = txt_embeds
            vision_features = torch.cat([pano_embeds, object_embed, action_embed], 1)
            vision_masks = torch.cat([pano_masks, object_masks, action_mask], 1)
            
            txt_step_ids = torch.full((batch_size, txt_embeds.shape[1]), t, dtype=torch.long).cuda()
            vision_step_ids = torch.full((batch_size, vision_features.shape[1]), t, dtype=torch.long).cuda()
            txt_type_ids = torch.full((batch_size, txt_embeds.shape[1]), 0, dtype=torch.long).cuda()
            vision_type_id1 = torch.full((batch_size, pano_embeds.shape[1]), 1, dtype=torch.long).cuda()
            vision_type_id2 = torch.full((batch_size, object_embed.shape[1]), 2, dtype=torch.long).cuda()
            vision_type_id3 = torch.full((batch_size, action_embed.shape[1]), 3, dtype=torch.long).cuda()
            vision_type_ids = torch.cat([vision_type_id1, vision_type_id2, vision_type_id3], 1)
            inputs = {'txt_features': txt_features, 'txt_masks': language_inputs['txt_masks'], 'vision_features': vision_features, 'vision_masks': vision_masks, \
                        'txt_step_ids': txt_step_ids, 'vision_step_ids': vision_step_ids, 'txt_type_ids': txt_type_ids, 'vision_type_ids': vision_type_ids}
            txt_features = self.vln_bert("cocoop", inputs)
            
            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps)
            nav_inputs.update(
                self._nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'], 
                    pano_inputs['view_lens'], pano_inputs['nav_types'],
                )
            )
            
            nav_inputs.update({
                'txt_embeds': txt_features,
                'txt_masks': language_inputs['txt_masks'],
            })
            nav_outs = self.vln_bert('navigation', nav_inputs)

            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['vp_cand_vpids']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits']
                nav_vpids = nav_inputs['gmap_vpids']

            nav_probs = torch.softmax(nav_logits, 1)
            
            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),
                    }
                                        
            if train_ml is not None:
                # Supervised training
                if self.args.dataset == 'r2r':
                    nav_targets = self._teacher_action_r4r(
                        obs, nav_vpids, ended, 
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        imitation_learning=(self.feedback=='teacher'), t=t, traj=traj
                    )
                else:
                    nav_targets = self._teacher_action_r4r(
                        obs, nav_vpids, ended, 
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        imitation_learning=(self.feedback=='teacher'), t=t, traj=traj
                    )
                
                ml_loss += self.criterion(nav_logits, nav_targets)
                
            # Determinate the next navigation viewpoint
            if self.feedback == 'teacher':
                a_t = nav_targets                 # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)        # student forcing - argmax
                a_t = a_t.detach() 
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach() 
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample': # in training
                # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []  
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])   

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf')}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))
                        
                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                            }

                # new observation and update graph
                obs = self.env._get_obs()
                self._update_scanvp_cands(obs)
                for i, ob in enumerate(obs):
                    if not ended[i]:
                        gmaps[i].update_graph(ob)

                ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

                # Early exit if all ended
                if ended.all():
                    break
                
                # global flag 
                # flag = True

        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            self.loss += ml_loss
            self.logs['IL_loss'].append(ml_loss.item())

        return traj
    

        