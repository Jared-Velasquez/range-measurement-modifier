import argparse
import os
import sys
from typing import Dict
import numpy as np
from py_factor_graph.io.pyfg_file import (
    read_from_pyfg_file,
    save_to_pyfg_file
)
from py_factor_graph.modifiers import (
    split_single_robot_into_multi,
)

from py_factor_graph.factor_graph import FactorGraphData

from py_factor_graph.utils.name_utils import get_robot_char_from_number

ROBOTS = [8]

def get_pose_mapping(fg: FactorGraphData, new_fg: FactorGraphData) -> Dict[str, str]:
    multi_robot_num_poses = []
    for i in range(new_fg.num_robots):
        multi_robot_num_poses.append(new_fg.num_poses_by_robot_idx(i))
    assert fg.num_poses_by_robot_idx(0) == sum(multi_robot_num_poses)

    num_robots = new_fg.num_robots
    num_total_poses = fg.num_poses

    print("Num robots: " + str(num_robots))

    pose_chain_indices = np.linspace(
            start=0, stop=num_total_poses, num=num_robots + 1, dtype=int
        )
    
    robot_pose_chain_bounds = list(
            zip(pose_chain_indices[:-1], pose_chain_indices[1:])
        )

    print(robot_pose_chain_bounds)
    
    total_poses = 0
    pose_mapping = {}
    for robot_idx, (start, end) in enumerate(robot_pose_chain_bounds):
        print("Robot idx: " + str(robot_idx))
        print("Start-end: " + str(start) + " " + str(end))
        old_robot_char = get_robot_char_from_number(0)
        new_robot_char = get_robot_char_from_number(robot_idx)
        for pose_idx in range(end - start):
            # print(total_poses, pose_idx)
            pose_mapping[f"{old_robot_char}{total_poses}"] = f"{new_robot_char}{pose_idx}"
            total_poses += 1
    
    return pose_mapping


def add_range_measurements_with_multi_robots(fg: FactorGraphData, new_fg: FactorGraphData) -> FactorGraphData:
    save_to_pyfg_file(fg, os.path.join(f"single_drone_modified_{new_fg.num_robots}.pyfg"))
    print("fg num robots: " + str(fg.num_robots))
    pose_mapping = get_pose_mapping(fg, new_fg)
    # print(pose_mapping)
    for landmark in fg.landmark_variables:
        new_fg.add_landmark_variable(landmark)
    for range_measurement in fg.range_measurements:
        print("FG Num robots: " + str(fg.num_robots))
        new_range_measurement = range_measurement
        print(range_measurement.association)
        # print(pose_mapping[range_measurement.association[0]])
        if range_measurement.association[0] not in pose_mapping.keys():
            print(f"{range_measurement.association[0]} not in pose mapping")
            continue
        new_association = (pose_mapping[range_measurement.association[0]], range_measurement.association[1])
        print(new_association)
        new_range_measurement.association = new_association
        new_fg.add_range_measurement(new_range_measurement)
    return new_fg

def split_robot(args):
    dataset = args.dataset
    output_dir = args.output_dir

    if (not os.path.isdir(output_dir)):
        os.makedirs(output_dir)

    fg = read_from_pyfg_file(dataset)
    # inter_robot_range_model = RangeMeasurementModel(SMALLGRID3D_SE_SYNC_SENSING_HORIZON, noise, meas_prob)

    for num_robot in ROBOTS:
        new_fg = split_single_robot_into_multi(fg, num_robot)
        new_fg = add_range_measurements_with_multi_robots(fg, new_fg)
        save_to_pyfg_file(new_fg, os.path.join(output_dir, f"single_drone_{num_robot}_robot.pyfg"))


def main(args):
    parser = argparse.ArgumentParser(
        description="This script is used to generate synthetic multi-robot datasets with range measurements from an existing g2o file."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="g2o filepath to create synthetic datasets from"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="directory where evaluation results are saved",
    )

    args = parser.parse_args()
    split_robot(args)

if __name__ == "__main__":
    main(sys.argv[1:])