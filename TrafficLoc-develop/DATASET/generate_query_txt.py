import os

root_folder = "./demo_dataset"

voxel_size = 50
voxel_stride = 25
image_overlap_threshold = 0.3
str_image_overlap = str(image_overlap_threshold).replace(".", "")
voxel_overlap_threshold = 0.25
str_voxel_overlap = str(voxel_overlap_threshold).replace(".", "")

suffix_str = f"v{voxel_size}_s{voxel_stride}_io{str_image_overlap}_vo{str_voxel_overlap}"

test_list = []

# Test T1-T7
int_name_list = ['t1_int1', 't2_int1', 't3_int1', 't4_int1', 't5_int1', 't6_int1', 't7_int1']
output_path = os.path.join(root_folder, 'train_list', f'query_all_1to7_{suffix_str}.txt')

with open(output_path, 'w') as f:
    for int_name in int_name_list:
        save_str = f"{int_name}/train_list_{suffix_str}/query_{int_name}_{suffix_str}.npy\n"
        f.write(save_str)
        test_list.append(save_str)
        
print(f"Test split T1-T7 saved in {output_path}")

# Test T1-T7 hard
int_name_list = ['t1_int1', 't2_int1', 't3_int1', 't4_int1', 't5_int1', 't6_int1', 't7_int1']
output_path = os.path.join(root_folder, 'train_list', f'query_all_1to7_{suffix_str}_seq5.txt')

with open(output_path, 'w') as f:
    for int_name in int_name_list:
        save_str = f"{int_name}/train_list_{suffix_str}/seq5_query_{int_name}_{suffix_str}.npy\n"
        f.write(save_str)
        test_list.append(save_str)

print(f"Test split T1-T7_hard saved in {output_path}")

# Val (Test T1-T7 + Test T1-T7 hard + Test T10)
test_list.append(f"t10_int1/train_list_{suffix_str}/query_t10_int1_{suffix_str}.npy\n")
output_path = os.path.join(root_folder, 'train_list', f'query_all_3testset_{suffix_str}.txt')

with open(output_path, 'w') as f:
    for line in test_list:
        f.write(line)

print(f"Test split T10 saved in {output_path}")