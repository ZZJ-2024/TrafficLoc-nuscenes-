import os

int_name_list = ['t1_int2', 't1_int3', 't1_int4', 't1_int5', 't1_int6', 't1_int7', 't1_int8', 't1_int9', 't1_int10',
                't2_int2', 't2_int3', 't2_int4', 't2_int5', 't2_int6', 't2_int7', 't2_int8', 't2_int9',
                't3_int2', 't3_int3', 't3_int4', 't3_int5', 't3_int6', 't3_int7', 't3_int8',
                't4_int2', 't4_int3', 't4_int4', 't4_int5', 't4_int6', 't4_int7', 't4_int8', 't4_int9', 't4_int10', 't4_int11', 't4_int12',
                't5_int2', 't5_int3', 't5_int4', 't5_int5', 't5_int6', 't5_int7', 't5_int8', 't5_int9', 't5_int10', 't5_int11', 't5_int12', 't5_int13', 't5_int14', 't5_int15',
                't6_int2', 't6_int3', 't6_int4', 't6_int5', 't6_int6', 't6_int7', 't6_int8', 't6_int9', 't6_int10', 't6_int11', 't6_int12',
                't7_int2', 't7_int3', 't7_int4', 't7_int5', 't7_int6', 't7_int7', 't7_int8'
                ]


root_folder = f"./demo_dataset"

voxel_size = 50
voxel_stride = 25
image_overlap_threshold = 0.3
str_image_overlap = str(image_overlap_threshold).replace(".", "")
voxel_overlap_threshold = 0.25
str_voxel_overlap = str(voxel_overlap_threshold).replace(".", "")

suffix_str = f"v{voxel_size}_s{voxel_stride}_io{str_image_overlap}_vo{str_voxel_overlap}"

output_path = os.path.join(root_folder, 'train_list', f'train_allscene_{suffix_str}.txt')

train_list = []

for int_name in int_name_list:
    print(f"processsing {int_name}")
    file_path = os.path.join(root_folder, 'train_list', f'train_{int_name}_{suffix_str}.txt')
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        train_list.extend(lines)
    
with open(output_path, 'w') as f:
    for line in train_list:
        f.write(line)
        
print(f"file is saved in {output_path}")