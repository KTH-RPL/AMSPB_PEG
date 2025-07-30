# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import math
import random
import matplotlib.pyplot as plt
import torch
import torch.distributions as D
import numpy as np
from typing import Dict, List


def get_seeds_list(n_seeds:int) -> List[int]:
    seeds = np.arange(0, n_seeds, dtype=int)
    return seeds.tolist()

def rectangular_ring_sampling(big_low, big_high, small_low, small_high, num_points):
    """
    Sample points in a rectangular ring area defined by two rectangles.
    The outer rectangle is defined by big_low and big_high, and the inner
    rectangle is defined by small_low and small_high.
    """
    big_dist = D.Uniform(big_low, big_high)

    points_per_batch = 100
    points = torch.empty((num_points,3),device=big_low.device)
    num_points_achieved = 0

    while num_points_achieved < num_points:
        outer_points = big_dist.sample((points_per_batch,))

        # Check if the points are outside the inner rectangle
        mask = (outer_points[:, 0] < small_low[0]) | (outer_points[:, 0] > small_high[0]) | \
               (outer_points[:, 1] < small_low[1]) | (outer_points[:, 1] > small_high[1]) | \
               (outer_points[:, 2] < small_low[2]) | (outer_points[:, 2] > small_high[2])

        valid_outer_points = outer_points[mask]

        num_points_new = np.min([(num_points - num_points_achieved), valid_outer_points.shape[0]])

        points[num_points_achieved:(num_points_achieved + num_points_new), :3] = valid_outer_points[:num_points_new]
        num_points_achieved += num_points_new
    
    return points


def min_separation_sampling(low:torch.Tensor, high:torch.Tensor, ref_points:torch.Tensor, min_separation:float, K:int=10):
    points_dist = D.Uniform(low, high)

    num_points = ref_points.shape[0]
    points = torch.empty((num_points,3),device=low.device)
    points_not_achieved_mask = torch.ones((num_points), device=points.device).bool()

    while points_not_achieved_mask.any():
        num_remaining = points_not_achieved_mask.sum()
        sampled_points = points_dist.sample((num_remaining, K))

        mask_separation = torch.norm(sampled_points - ref_points[points_not_achieved_mask, :].unsqueeze(1), dim=-1) > min_separation
        mask_valid_separation = torch.argmax(mask_separation.int(), dim=1) 
        mask_separation = mask_separation.any(dim=1)
        sampled_points = sampled_points[mask_separation,:]
        mask_add_points = torch.logical_and(points_not_achieved_mask, mask_separation)
        batch_idx = torch.arange(sampled_points.shape[0], device=sampled_points.device)
        points[mask_add_points, :] = sampled_points[batch_idx,mask_valid_separation[mask_separation],:]

        points_not_achieved_mask = torch.logical_and(points_not_achieved_mask, ~mask_add_points)
    # One can comment this check for performance
    final_check = torch.norm(points - ref_points, dim=-1) > min_separation
    assert final_check.all(), "Some points are too close to the reference points"

    return points


def policy_sampling(policy_pool: Dict[str, float], num_sampled: int) -> Dict[str, List[int]]:
    policies = policy_pool.keys()
    probs = np.array([policy_pool[p] for p in policies], dtype=float)
    total = probs.sum()
    if not np.isclose(total, 1.0):
        probs = probs / total  # normalize if needed
    cdf = np.cumsum(probs)

    result: Dict[str, List[int]] = {p: [] for p in policies}

    for i in range(num_sampled):
        threshold = (i + 0.5) / num_sampled
        for policy, cum_prob in zip(policies, cdf):
            if threshold <= cum_prob:
                result[policy].append(i)
                break
    
    return result