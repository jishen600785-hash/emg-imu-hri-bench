from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder
import xacro
import yaml


DEFAULT_PROJECT_ROOT = os.environ.get("EMG_HRI_PROJECT_ROOT", os.getcwd())


def _launch_setup(context):
    package_share = get_package_share_directory("sj_fitts_sim")
    project_root = LaunchConfiguration("project_root").perform(context)
    subject_id = LaunchConfiguration("subject_id")
    fold_index_value = 1
    evaluation_mode = True
    held_out_condition = "dynamic"
    model_path = os.path.join(
        project_root,
        "artifacts",
        "ml",
        "optimized",
        "improved_strict_smoothed_model.joblib",
    )
    dataset_path = os.path.join(
        project_root,
        "artifacts",
        "prepared",
        "window_dataset.npz",
    )
    split_manifest_path = os.path.join(
        package_share, "config", "sj_split_manifest_v4.json"
    )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Optimized SJ model not found: {model_path}")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Prepared SJ window dataset not found: {dataset_path}")
    if not os.path.isfile(split_manifest_path):
        raise FileNotFoundError(f"SJ split manifest not found: {split_manifest_path}")
    gui = LaunchConfiguration("gui")
    rviz = LaunchConfiguration("rviz")
    autopilot = LaunchConfiguration("autopilot")
    record_video = LaunchConfiguration("record_video")
    video_fps = LaunchConfiguration("video_fps")
    sequences_per_condition = LaunchConfiguration("sequences_per_condition")
    practice_enabled = LaunchConfiguration("practice_enabled")
    far_validation = (
        LaunchConfiguration("far_validation").perform(context).strip().lower()
        in {"1", "true", "yes", "on"}
    )

    controllers_file = os.path.join(package_share, "config", "ros2_controllers.yaml")
    initial_positions_file = os.path.join(package_share, "config", "initial_positions.yaml")
    xacro_file = os.path.join(package_share, "urdf", "ur5e_fitts.urdf.xacro")
    world_file = os.path.join(package_share, "worlds", "fitts_lab.sdf")
    task_config = os.path.join(package_share, "config", "fitts_task.yaml")
    servo_file = os.path.join(package_share, "config", "ur_servo_gazebo.yaml")
    configured_results_dir = LaunchConfiguration("results_dir").perform(context).strip()
    configured_video_dir = LaunchConfiguration("video_dir").perform(context).strip()
    v4_root = os.path.join(project_root, "ros2", "sj_v4_ws")
    results_dir = configured_results_dir or os.path.join(v4_root, "results")
    video_dir = configured_video_dir or os.path.join(v4_root, "videos")
    task_profile_overrides = {}
    if far_validation:
        task_profile_overrides = {
            "distances_m": [0.28],
            "widths_m": [0.025, 0.060],
            "practice_enabled": False,
            "randomize_conditions": False,
        }

    robot_description_xml = xacro.process_file(
        xacro_file,
        mappings={
            "controllers_file": controllers_file,
            "initial_positions_file": initial_positions_file,
        },
    ).toxml()
    robot_description = {"robot_description": robot_description_xml}

    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        .robot_description_semantic(Path("srdf") / "ur.srdf.xacro", {"name": "ur5e_fitts"})
        .to_moveit_configs()
    )
    moveit_parameters = moveit_config.to_dict()
    moveit_parameters.update(robot_description)
    with open(servo_file, "r", encoding="utf-8") as handle:
        servo_parameters = {"moveit_servo": yaml.safe_load(handle)}

    gz_launch = os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
    gazebo_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch),
        condition=IfCondition(gui),
        launch_arguments={"gz_args": f"-r -v 3 '{world_file}'"}.items(),
    )
    gazebo_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch),
        condition=UnlessCondition(gui),
        launch_arguments={"gz_args": f"-r -s -v 3 '{world_file}'"}.items(),
    )

    clock_and_pose_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="fitts_gz_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/world/fitts_world/set_pose@ros_gz_interfaces/srv/SetEntityPose",
            "/fitts/camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
        ],
    )
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": True}],
    )
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-world",
            "fitts_world",
            "-topic",
            "robot_description",
            "-name",
            "ur5e_fitts",
            "-allow_renaming",
            "false",
        ],
    )
    joint_state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )
    position_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=["forward_position_controller", "--controller-manager", "/controller_manager"],
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_parameters, {"use_sim_time": True, "publish_robot_description_semantic": True}],
    )
    servo_node = Node(
        package="moveit_servo",
        executable="servo_node",
        name="servo_node",
        output="screen",
        parameters=[moveit_parameters, servo_parameters, {"use_sim_time": True}],
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        condition=IfCondition(rviz),
        arguments=["-d", os.path.join(package_share, "config", "fitts.rviz")],
        parameters=[moveit_parameters, {"use_sim_time": True}],
    )

    common_overrides = {
        "project_root": project_root,
        "subject_id": subject_id,
        "use_sim_time": True,
    }
    replay = Node(
        package="sj_fitts_sim",
        executable="emg_replay",
        name="sj_emg_replay",
        output="screen",
        condition=IfCondition(autopilot),
        parameters=[
            task_config,
            common_overrides,
            {
                "condition": held_out_condition,
                "model_path": model_path,
                "dataset_path": dataset_path,
                "split_manifest_path": split_manifest_path,
                "evaluation_mode": evaluation_mode,
                "fold_index": fold_index_value,
            },
        ],
    )
    classifier = Node(
        package="sj_fitts_sim",
        executable="gesture_classifier",
        name="sj_gesture_classifier",
        output="screen",
        parameters=[
            task_config,
            common_overrides,
            {
                "model_path": model_path,
                "expected_condition": held_out_condition,
            },
        ],
    )
    gesture_servo = Node(
        package="sj_fitts_sim",
        executable="gesture_servo",
        name="gesture_servo",
        output="screen",
        parameters=[task_config, {"use_sim_time": True}],
    )
    fitts_task = Node(
        package="sj_fitts_sim",
        executable="fitts_task",
        name="fitts_task",
        output="screen",
        parameters=[
            task_config,
            {
                "subject_id": subject_id,
                "results_dir": results_dir,
                "sequences_per_condition": sequences_per_condition,
                "practice_enabled": ParameterValue(
                    practice_enabled, value_type=bool
                ),
                "autopilot": ParameterValue(autopilot, value_type=bool),
                "evaluation_mode": evaluation_mode,
                "fold_index": fold_index_value,
                "held_out_condition": held_out_condition,
                "model_path": model_path,
                "use_sim_time": True,
            },
            task_profile_overrides,
        ],
    )
    video_recorder = Node(
        package="sj_fitts_sim",
        executable="video_recorder",
        name="fitts_video_recorder",
        output="screen",
        condition=IfCondition(record_video),
        parameters=[
            {
                "video_dir": video_dir,
                "subject_id": subject_id,
                "evaluation_mode": evaluation_mode,
                "fold_index": fold_index_value,
                "held_out_condition": held_out_condition,
                "fps": video_fps,
                "post_roll_sec": 2.0,
                "camera_topic": "/fitts/camera/image",
                "use_sim_time": True,
            }
        ],
    )

    controllers_after_spawn = RegisterEventHandler(
        OnProcessExit(target_action=spawn_robot, on_exit=[joint_state_spawner, position_spawner])
    )
    return [
        gazebo_gui,
        gazebo_headless,
        clock_and_pose_bridge,
        robot_state_publisher,
        spawn_robot,
        controllers_after_spawn,
        move_group,
        servo_node,
        rviz_node,
        replay,
        classifier,
        gesture_servo,
        fitts_task,
        video_recorder,
    ]


def generate_launch_description():
    default_root = os.environ.get("SJ_PROJECT_ROOT", DEFAULT_PROJECT_ROOT)
    return LaunchDescription(
        [
            DeclareLaunchArgument("project_root", default_value=default_root),
            DeclareLaunchArgument("subject_id", default_value="1"),
            DeclareLaunchArgument("autopilot", default_value="true"),
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("record_video", default_value="true"),
            DeclareLaunchArgument("video_fps", default_value="15.0"),
            DeclareLaunchArgument("sequences_per_condition", default_value="1"),
            DeclareLaunchArgument("practice_enabled", default_value="true"),
            DeclareLaunchArgument("far_validation", default_value="false"),
            DeclareLaunchArgument("results_dir", default_value=""),
            DeclareLaunchArgument("video_dir", default_value=""),
            OpaqueFunction(function=_launch_setup),
        ]
    )
