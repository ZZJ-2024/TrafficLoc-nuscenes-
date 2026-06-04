This README provides an overview of the ``Carla Intersection Dataset``, which includes the dataset structure, code for data collection, and preprocessing scripts, among other resources. ``Carla Intersection Dataset`` is collected in [CARLA Simulator](https://carla.readthedocs.io/en/latest/) and is first utilized in [TrafficLoc](https://tum-luk.github.io/projects/trafficloc/)

# Dataset Structure
- The file structure should be like the following:
```
carla_intersection_dataset
├── dataset_large_int_pose
│   ├── t1_int1_lidar.txt
│   ├── t1_int1_lidar_traj.txt
│   ├── t1_int1_seq3.txt
│   ├── t1_int1_seq3_traj.txt
│   ├── t1_int1_seq4.txt
│   ├── t1_int1_seq4_traj.txt
│   ├── ......
├── t1_int1
│   ├── t1_int1_lidar_road
│       ├── agg_t1_int1_lidar_road.ply
│       ├── lidar_0.ply
│       ├── lidar_1.ply
│       ├── ......
│   ├── t1_int1_seq3_rgb_png
│       ├── image_0.png
│       ├── image_1.png
│       ├── ......
│   ├── t1_int1_seq3_depth_png
│       ├── depth_0.png
│       ├── depth_1.png
│       ├── ......
│   ├── t1_int1_seq4_rgb_png
│       ├── image_0.png
│       ├── image_1.png
│       ├── ......
│   ├── t1_int1_seq4_depth_png
│       ├── depth_0.png
│       ├── depth_1.png
│       ├── ......
│   ├── t1_int1_seq5_rgb_png
│       ├── image_0.png
│       ├── image_1.png
│       ├── ......
│   ├── t1_int1_seq5_depth_png
│       ├── depth_0.png
│       ├── depth_1.png
│       ├── ......
│   ├── t1_int1_eval_records_camera.txt
│   ├── t1_int1_eval_trajectories.txt
│   ├── t1_int1_sensors.txt
│   ├── t1_int1_train_records_camera.txt
│   ├── t1_int1_train_trajectories.txt
├── t1_int2
│   ├── ...
├── t1_int3
│   ├── ...
├── ......
├── ......
├── template
```

- **`dataset_large_int_pose`**: contains the original camera poses and lidar poses in the CARLA Simulator used for data collection stage.
    - **`xxx_lidar.txt`**: lidar pose in format x, y, z, yaw, pitch, roll
    - **`xxx_lidar_traj.txt`**: lidar pose in format quaternion
    - **`xxx_seqX.txt`**: camera pose in format x, y, z, yaw, pitch, roll
    - **`xxx_seqX_traj.txt`**: camera pose in format quaternion

- **`t1_int1`**: contains the data for intersection t1_int1 (Intersection 1 from Town 1)
    - **`t1_int1_lidar_road`**: contains the collected LiDAR scan and the final point cloud of the intersection
    - **`t1_int1_seq3_rgb_png`**: contains the RGB images of seq3
    - **`t1_int1_seq3_depth_png`**: contains the depth images of seq3
    - **`t1_int1_seq4_rgb_png`**: contains the RGB images of seq4
    - **`t1_int1_seq4_depth_png`**: contains the depth images of seq4
    - **`t1_int1_seq5_rgb_png`**: contains the RGB images of seq5
    - **`t1_int1_seq5_depth_png`**: contains the depth images of seq5
    - **`xxx.txt`**: Ground Truth camera pose data used for training and evaluation in [kapture](https://github.com/naver/kapture) format

- **`template`**: a template empty folder which will be used to generate data with softlink. 

## P.S
- Only the first intersection of each world (tX_int1) are used for testing. Other intersections are used for training. Folder `seq3` contains the training images.
- The testing scenes from Town01 to Town07 (`t1_int1`, `t2_int1`, ... , `t7_int1`) contain both the `seq4` and `seq5`. `seq4` is used for Test Split **``Test_T1-T7``**, `seq5` is used for Test Split **``Test_T1-T7_hard``**.
- The `seq4` of `t10_int1` is used for Test Split **``Test_T10``**.




# Dataset Preparation
The [`transform_dataset.py`](./transform_dataset.py) is used to soft-link the training and testing data to a new folder (e.g. `demo_dataset`) with desired format. So that you can create the `demo_dataset` folder anywhere you want. (under your project folder or where you need to use it) 

Remember to modify the path of `source_folder` and `train_folder` before running.
```shell
python transform_data.py
```

After processing, you should get the dataset in a new format like following. This dataset format is used for [TrafficLoc](https://tum-luk.github.io/projects/trafficloc/) to do training and evaluation:
```
demo_dataset
├── t1_int1
│   ├── mapping
│       ├── points # empty folder for placeholder
│       ├── sensors
│           ├── depth_data  # depth images
│           ├── records_data  # RGB images
│           ├── records_camera.txt  # training image names
│           ├── sensors.txt  # camera parameters
│           ├── trajectories.txt  # training camera poses
│       ├── pcd_t1_int1_train_down.ply  # point cloud
│   ├── query
│       ├── points # empty folder for placeholder
│       ├── sensors
│           ├── depth_data  # depth images
│           ├── records_data  # RGB images
│           ├── records_camera.txt  # evaluation image names
│           ├── sensors.txt  # camera parameters
│           ├── trajectories.txt  # evaluation camera poses
│       ├── pcd_t1_int1_train_down.ply  # point cloud
├── t1_int2
│   ├── ...
├── t1_int3
│   ├── ...
├── ......
├── ......

```

# Dataset Collection
To enable users to collect custom datasets, we have also open-sourced the code used for data collection in the CARLA simulator, along with the specific parameters employed for generating the Carla Intersection Dataset.

All the secific parameters used for data collection are documented in a published [Notion page](https://boiled-stick-c46.notion.site/Dataset-intersection-carla-1a63d4cd7d05460d873e4d7c970ca7ca), which contains:

- Environmental Config
- Camera Config
- LiDAR Config
- Center Coorindate of Each Intersection
- Pose Collection Parameters
- Visualization of Each Intersection

## CARLA Python API
Our codes are based on the Python CARLA package. You can follow the [Quick start package installation](https://carla.readthedocs.io/en/latest/start_quickstart/) in the CARLA website to install the CARLA and the Python package.

We use the `CARLA 0.9.15 Linux version` on Ubuntu to collect the dataset.

## Collection Code
All the codes are located in the folder [collection_code](./collection_code/). When collecting data in CARLA Simulator, remember to run the simulator first, and then run the target codes.

- **`generate_location.py`**: generate the camera pose or LiDAR pose in format of x, y, z, yaw, pitch, roll.

- **`filter_lidar_collision.py`**: filter the lidar pose that are not located in the traffic road. For some corner cases it may need to manuallt set the filter boundary of lidar poses.

- **`transform_pose_to_traj.py`**: transform the pose file into trajectory file with quaternion format. (e.g. from `t1_int1_seq3.txt` to `t1_int1_seq3_traj.txt`)

- **`collect_sensor_data.py`**: collect RGB images, depth images or LiDAR scans in the CARLA Simulator using the pose files. Remember to change the corresponding sensor type when collecting data.

- **`lidar_acc.py`**: accumulate all LiDAR scans of one intersection into a point cloud and downsample the point cloud with resolution of 0.2m.

- **`lidar_crop.py`**: crop the point cloud of one intersection into a region of 100m x 100m x 50m. Remember to change the center coordinate for each intersection.

- **`generate_trajectory.py`**: generate the training and evaluation trajectory file (e.g. `t1_int1_train_trajectories.txt`, `t1_int1_train_records_camera.txt`, `t1_int1_sensors.txt`) from the pose files.

The recommended execution order is:
```shell
# generate camera pose and LiDAR pose 
python generate_location.py

# filter the LiDAR pose (need CARLA running)
python filter_lidar_collision.py

# transform the pose into quaternion format
python transform_pose_to_traj.py

# collect images and LiDAR data (need CARLA running)
python collect_sensor_data.py

# accumulate and crop LiDAR scans
python lidar_acc.py
python lidar_crop.py

# generate the training and evaluation trajectory file
python generate_trajectory.py

```