from glob import glob
from setuptools import find_packages, setup


package_name = "sj_fitts_sim"


setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml") + glob("config/*.rviz")),
        (f"share/{package_name}/urdf", glob("urdf/*.xacro")),
        (f"share/{package_name}/worlds", glob("worlds/*.sdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ji Shen",
    maintainer_email="student@example.com",
    description="SJ EMG/IMU optimized-model Fitts' Law robotic-arm simulation",
    license="MIT",
    entry_points={
        "console_scripts": [
            "emg_replay = sj_fitts_sim.emg_replay_node:main",
            "gesture_classifier = sj_fitts_sim.classifier_node:main",
            "gesture_servo = sj_fitts_sim.gesture_servo_node:main",
            "fitts_task = sj_fitts_sim.fitts_task_node:main",
            "model_smoke_test = sj_fitts_sim.model_smoke_test:main",
            "video_recorder = sj_fitts_sim.video_recorder_node:main",
        ],
    },
)
