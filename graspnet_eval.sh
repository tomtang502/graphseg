#!/bin/bash
for scene_id in {0..189}; do
    echo "Running eval_graspnet with --scene_id $scene_id"
    python eval_graspnet.py --scene_id "$scene_id"
done
