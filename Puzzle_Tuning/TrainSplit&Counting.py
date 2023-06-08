import os
import shutil


def make_and_clear_path(file_pack_path):
    if not os.path.exists(file_pack_path):
        os.makedirs(file_pack_path)

    """if os.path.exists(file_pack_path):
        # 如果目标路径存在原文件夹的话就先删除
        shutil.rmtree(file_pack_path)"""


def find_copy_all_files(root, target, suffix=None, i = 0,add_class = True):
    """
    Return a list of file paths ended with specific suffix
    """
    res = []
    if type(suffix) is tuple or type(suffix) is list:
        for root, _, files in os.walk(root):
            for f in files:
                if suffix is not None:
                    status = 0
                    for i in suffix:
                        if not f.endswith(i):
                            pass
                        else:
                            status = 1
                            break
                    if status == 0:
                        continue
                res.append(os.path.join(root, f))
                i = i+1
                if add_class:
                    # add class name before image name, preventing same name in one folder
                    img_name = f.split('.')[0]
                    class_name = os.path.split(root)[1]
                    new_img_name = class_name + '_' + img_name + '.jpg'
                    print(new_img_name)
                    shutil.copy(os.path.join(root, f), os.path.join(target, new_img_name))
                else:
                    shutil.copy(os.path.join(root, f), os.path.join(target, f))
        return i


    elif type(suffix) is str or suffix is None:
        for root, _, files in os.walk(root):
            for f in files:
                if suffix is not None and not f.endswith(suffix):
                    continue
                res.append(os.path.join(root, f))
                i = i+1
                if add_class:
                    # add class name before image name, preventing same name in one folder
                    img_name = f.split('.')[0]
                    class_name = os.path.split(root)[1]
                    new_img_name = class_name + '_' + img_name + '.jpg'
                    print(new_img_name)
                    shutil.copy(os.path.join(root, f), os.path.join(target, new_img_name))
                else:
                    shutil.copy(os.path.join(root, f), os.path.join(target, f))
        return i

    else:
        print('type of suffix is not legal :', type(suffix))
        return -1


def MIL_to_CLS(dataset_root, output_root, output_root_2, class_name):
    make_and_clear_path(output_root)

    train_root = os.path.join(dataset_root, 'train')
    num = find_copy_all_files(train_root, output_root, 'jpg', add_class=False)
    print(class_name + str(num))

    split_names = ['val', 'test']
    for split_name in split_names:
        output_root_data = os.path.join(output_root_2, split_name)
        make_and_clear_path(output_root_data)
        work_root = os.path.join(dataset_root, split_name)
        class_names = os.listdir(work_root)

        for class_name in class_names:
            source_root = os.path.join(work_root, class_name)

            source_root = os.path.abspath(source_root)
            target_root = os.path.join(output_root_data, class_name)

            target_root = os.path.abspath(target_root)

            shutil.copytree(source_root, target_root)


if __name__ == '__main__':
    input_root = r'D:\M2\PuzzleTuningDatasets#1_LiteM2\L_Scale'
    output_root = r'D:\M2\PuzzleTuningDatasets#1_Lite_trainM2\L_Scale'
    output_root_2 = r'D:\M2\PuzzleTuningDatasets#1_Lite_val_testM2\L_Scale'

    datasets = os.listdir(input_root)
    # datasets = ['CRC-TP']
    for dataset in datasets:
        input_root_data = os.path.join(input_root, dataset)
        output_root_data = os.path.join(output_root, dataset)
        output_root_data_2 = os.path.join(output_root_2, dataset)
        MIL_to_CLS(input_root_data,
               output_root_data, output_root_data_2, dataset)