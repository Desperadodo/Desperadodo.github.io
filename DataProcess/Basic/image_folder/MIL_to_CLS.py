import os
import shutil


def make_and_clear_path(file_pack_path):
    if not os.path.exists(file_pack_path):
        os.makedirs(file_pack_path)

    """if os.path.exists(file_pack_path):
        # 如果目标路径存在原文件夹的话就先删除
        shutil.rmtree(file_pack_path)"""


def MIL_to_CLS(dataset_root, output_root):
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        output_root_data = os.path.join(output_root, split_name)
        make_and_clear_path(output_root_data)
        work_root = os.path.join(dataset_root, split_name)
        work_root_data = os.path.join(work_root, 'data')
        class_names = os.listdir(work_root_data)
        print(class_names)
        for class_name in class_names:
            source_root = os.path.join(work_root_data, class_name)
            print(source_root)
            source_root = os.path.abspath(source_root)
            target_root = os.path.join(output_root_data, class_name)
            print(target_root)
            target_root = os.path.abspath(target_root)

            shutil.copytree(source_root, target_root)


if __name__ == '__main__':
    current_all_root = r'D:\CellMix'
    all_output_root = r'D:\AMB_CLS'

    all_roots = os.listdir(current_all_root)
    for all_root in all_roots:
        input_root = os.path.join(current_all_root, all_root)
        output_root = input_root.replace('MIL', 'CLS')
        MIL_to_CLS(input_root,
                   output_root)