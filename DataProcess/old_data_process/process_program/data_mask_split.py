'''
dataset divide script  ver： AUG 29th 22：00 official release

ref：https://zhuanlan.zhihu.com/p/199238910
'''
import os
import random
import shutil
from shutil import copy2
from multiprocessing import Pool, cpu_count


def del_file(filepath):
    """
    Delete all files or folders in a directory
    :param filepath: path of file
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        """if os.path.isfile(file_path):
            os.remove(file_path)"""
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)


def make_and_clear_path(file_pack_path):
    if not os.path.exists(file_pack_path):
        os.makedirs(file_pack_path)
    # del_file(file_pack_path)


def a_dataset_split(src_data_folder, class_name, train_scale, val_scale, test_scale, com_num=None):
    data_folder = os.path.join(src_data_folder, 'data')
    mask_folder = os.path.join(src_data_folder, 'mask')
    current_class_mask_path = os.path.join(mask_folder, class_name)
    current_class_data_path = os.path.join(data_folder, class_name)
    current_all_data_mask = os.listdir(current_class_mask_path)  # 用mask去shuffle， data跟着 没有相应的data就不要了

    # print(current_all_data_mask)
    current_data_mask_length = len(current_all_data_mask)
    current_data_mask_index_list = list(range(current_data_mask_length))
    random.shuffle(current_data_mask_index_list)

    train_data_folder = os.path.join(os.path.join(os.path.join(target_data_folder, 'train'), "data"), class_name)
    val_data_folder = os.path.join(os.path.join(os.path.join(target_data_folder, 'val'), "data"), class_name)
    test_data_folder = os.path.join(os.path.join(os.path.join(target_data_folder, 'test'), "data"), class_name)

    train_mask_folder = os.path.join(os.path.join(os.path.join(target_data_folder, 'train'), "mask"), class_name)
    val_mask_folder = os.path.join(os.path.join(os.path.join(target_data_folder, 'val'), "mask"), class_name)
    test_mask_folder = os.path.join(os.path.join(os.path.join(target_data_folder, 'test'), "mask"), class_name)

    make_and_clear_path(train_mask_folder)
    make_and_clear_path(train_data_folder)
    make_and_clear_path(val_mask_folder)
    make_and_clear_path(val_data_folder)
    make_and_clear_path(test_mask_folder)
    make_and_clear_path(test_data_folder)

    train_stop_flag = current_data_mask_length * train_scale
    val_stop_flag = current_data_mask_length * (train_scale + val_scale)
    all_stop_flag = current_data_mask_length * (train_scale + val_scale + test_scale)
    current_idx = 0
    train_num = 0
    val_num = 0
    test_num = 0
    for i in current_data_mask_index_list:
        src_mask_path = os.path.join(current_class_mask_path, current_all_data_mask[i])
        # Sometimes the name of mask won't match the name of data   the name shall be slightly changed
        img_name = (os.path.split(src_mask_path)[1].split('.')[0]).split('_')[-1] + '.jpg'
        src_data_path = os.path.join(current_class_data_path, img_name)
        src_mask_path_new = os.path.join(current_class_mask_path, img_name)
        os.rename(src_mask_path, src_mask_path_new)  # 这几行有点屎山 别骂

        # print("ok...")
        if current_idx < train_stop_flag:
            copy2(src_mask_path_new, train_mask_folder)

            if not os.path.exists(src_data_path):
                print('can\'t find this data:' + src_mask_path)
                continue
            else:
                copy2(src_data_path, train_data_folder)
                # print("{} copied to {}".format(src_img_path, train_folder))
            train_num = train_num + 1

        elif (current_idx >= train_stop_flag) and (current_idx < val_stop_flag):
            copy2(src_mask_path_new, val_mask_folder)
            if not os.path.exists(src_data_path):
                print('can\'t find this data:' + src_mask_path)
                continue
            else:
                copy2(src_data_path, val_data_folder)
                # print("{} copied to{}".format(src_img_path, val_folder))
            val_num = val_num + 1

        elif (current_idx >= val_stop_flag) and (current_idx <= all_stop_flag):
            copy2(src_mask_path_new, test_mask_folder)
            if not os.path.exists(src_data_path):
                print('can\'t find this data:' + src_mask_path)
                continue
            else:
                copy2(src_data_path, test_data_folder)
                # print("{} copied to {}".format(src_img_path, test_folder))
            test_num = test_num + 1

        current_idx = current_idx + 1

    print("*********************************{}*************************************".format(class_name) + '\n' +
          "{} class has been divided into {}:{}:{}, a total of {} images".format(class_name, train_scale, val_scale,
                                                                                 test_scale,
                                                                                 current_data_mask_length) + '\n' + "Train set{}: {} pics".format(
        train_mask_folder,
        train_num)
          + '\n' + "Validation set{}: {} pics".format(val_mask_folder, val_num) + '\n' + "Test set{}: {} pics".format(
        test_mask_folder, test_num)
          + '\n')

    if com_num is not None:
        print('processed class idx:', com_num)


def data_set_split(src_data_folder, target_data_folder, train_scale=0.7, val_scale=0.1, test_scale=0.2,
                   Parallel_processing=False):
    """
    Read source data folder, generate divided folders as 'train', 'val' and 'test'
    :param src_data_folder: source folder r'E:\A_bunch_of_data\Processed'
    :param target_data_folder: target folder E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/target_data
    :param train_scale: train set ratio
    :param val_scale: validation set ratio
    :param test_scale: test set ratio
    :param Parallel_processing: whether to process in parallel
    :return:
    """
    make_and_clear_path(target_data_folder)
    print("Begin dataset division")
    class_names = os.listdir(os.path.join(src_data_folder, 'mask'))
    # Create folder in the target directory
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        # Then create category folder under the split_path directory
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            #  os.makedirs(class_split_path)

    if Parallel_processing:
        # Create process pool
        tasks_num = len(class_names)
        process_pool = Pool(min(cpu_count() - 2, tasks_num))  # Number of parallels, leave at least 2 cores

        com_num = 0
        print("start processing" + str(tasks_num) + " files by multi-process")
        # Schedule tasks
        for class_name in class_names:
            # Pool.apply_async(target to be called,(parameter tuple passed to the target,))
            # Use free process to call the target during each loop
            com_num += 1
            args = (src_data_folder, class_name, train_scale, val_scale, test_scale, com_num)
            process_pool.apply_async(a_dataset_split, args)

        process_pool.close()  # Close the process pool, process pool will no longer receive new requests once it is closed.
        process_pool.join()  # Wait till all process in process pool finished, must be placed after the 'close' statement

    else:
        # Divide the dataset according to the proportion, and copy the data image
        # Traverse by category
        for class_name in class_names:
            a_dataset_split(src_data_folder, class_name, train_scale, val_scale, test_scale)


def k_fold_split(src_data_folder, target_data_folder='./kfold', k=5):
    """
    Read the source data folder, generate divided folders as 'train', 'val'.
    :param src_data_folder: organized imagenet format folders that need to be divided by k-folding
    :param target_data_folder: large target folder with k folders generated inside, k folders are in imagenet format with train and val inside
    :param k: the number of divided folds
    :return:
    """
    make_and_clear_path(target_data_folder)
    print("Begin dataset division")
    class_names = os.listdir(src_data_folder)  # Get category name

    # Divide the dataset for each category according to the proportion, and copy and distribute the data images
    for class_name in class_names:  # Classification traversal first

        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_class_data_names = os.listdir(current_class_data_path)

        current_data_length = len(current_class_data_names)
        random.shuffle(current_class_data_names)

        # Divide data
        split_num = current_data_length // k
        # Put a packet for evert split_num data, and if there are k+1 packets, the last packet can only have k-1 data at most
        temp_split_pack = [current_class_data_names[i:i + split_num] for i in range(0, current_data_length, split_num)]
        fold_name_pack = [temp_split_pack[i] for i in range(0, k)]  # Get the first k packets
        if len(temp_split_pack) > k:  # If it can’t be divided equally at the end, the last one will have one more pack, and put the contents into different packs in turn
            for pack_idx, name in enumerate(temp_split_pack[-1]):  # The extra pack have at most k-1 data
                fold_name_pack[pack_idx].append(name)

        print("{} class is divided into {} cross-validation, a total of {} images".format(class_name, k,
                                                                                          current_data_length))

        for p in range(1, k + 1):  # For each fold, start from 1
            # Folder
            train_folder = os.path.join(target_data_folder, 'fold_' + str(p), 'train', class_name)
            val_folder = os.path.join(target_data_folder, 'fold_' + str(p), 'val', class_name)
            os.makedirs(train_folder)
            os.makedirs(val_folder)

            pack_idx = p - 1  # Use the current fold of data as val set, and use the rest as train set

            # Copy divided data
            train_num = 0
            val_num = 0

            for j in range(k):
                if j == pack_idx:
                    for i in fold_name_pack[j]:
                        src_img_path = os.path.join(current_class_data_path, i)
                        copy2(src_img_path, val_folder)
                        val_num += 1
                        # print("{} has copied to {}".format(src_img_path, val_folder))
                else:
                    for i in fold_name_pack[j]:
                        src_img_path = os.path.join(current_class_data_path, i)
                        copy2(src_img_path, train_folder)
                        train_num += 1
                        # print("{} has copied to {}".format(src_img_path, train_folder))
            print("fold {}:  class:{}  train num: {}".format(p, class_name, train_num))
            print("fold {}:  class:{}  val num: {}".format(p, class_name, val_num))


if __name__ == '__main__':
    # src_data_folder = r'C:\Users\admin\Desktop\jpg_dataset'
    # target_data_folder = r'C:\Users\admin\Desktop\ZTY_dataset'
    # data_set_split(src_data_folder, target_data_folder, train_scale=0.7,val_scale=0.1, test_scale=0.2)

    # Now used to process imagenet_1k data for pre-training
    src_data_folder = r'/Users/munros/MIL/Processed/Cervical/SIPaKMeD'
    target_data_folder = r'/Users/munros/MIL/Split/SIPaKMeD_MIL'
    data_set_split(src_data_folder, target_data_folder, train_scale=0.7, val_scale=0.1, test_scale=0.2)
    # k_fold_split(src_data_folder, target_data_folder, k=5)
