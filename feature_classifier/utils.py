import os


def create_dirs(dir_list):
    for current_dir in dir_list:
        current_path = current_dir[0]
        if len(current_dir) > 1:
            for sub_dir_index in range(1, len(current_dir)):
                current_path = os.path.join(current_path, current_dir[sub_dir_index])
        if not os.path.exists(current_path):
            os.makedirs(current_path)