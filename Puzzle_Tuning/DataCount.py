import os
import pandas as pd

root_path = 'D:\CPIA_VersionJournal\CPIA_MJ'
scales = ['L', "M", "S"]

for scale in scales:
    filepath = os.path.join(root_path, scale)
    dirs = os.listdir(filepath)
    name_dict = {}

    for dir in dirs:
        dir = os.path.join(filepath, dir)
        file_count = 0
        files = os.listdir(dir)
        for file in files:
            file_count += 1

        name_dict[os.path.split(dir)[1]] = file_count


    pd.DataFrame.from_dict(name_dict, orient='index', columns=['jpg number']).to_csv(os.path.join(root_path, 'CPIA_Journal_' + scale +' .csv'))