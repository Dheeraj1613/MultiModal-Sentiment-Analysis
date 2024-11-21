'''
extract facial features and save action unit columns to csv.
'''

import os
import pandas as pd
import argparse
import pickle

def getFilePaths(root):
    items = os.listdir(root)
    pathList=[]
    for item in items:
        full_path = os.path.join(root, item)
        if os.path.isfile(full_path):
            pathList.append(full_path)
        else:
            inner_paths = getFilePaths(full_path)
            for inner_path in inner_paths:
                pathList.append(inner_path)
    return pathList



parser = argparse.ArgumentParser()
parser.add_argument('-vid-folder', required=True)

args=parser.parse_args()

videoPaths = getFilePaths(args.vid_folder)

cols2select=['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']

for videoPath in videoPaths:
    print(videoPath)
    actor_name=videoPath.split(os.path.sep)[-2]
    video_name = videoPath.split(os.path.sep)[-1].split('.')[0]
    outs=os.path.join('./outs', actor_name, video_name)

    #extract facial features from current video and save into outs folder.
    command = f'./OpenFace/build/bin/FeatureExtraction -au_static -f {videoPath} -out_dir {outs}'
    os.system(command)

    #extract columns corresponding to action units from the csv file that contains all facial features.
    all_feat_file=os.path.join(outs, video_name+'.csv')
    pd_feats = pd.read_csv(all_feat_file)

    au_feats=pd_feats[cols2select]
    au_feat_save_folder=os.path.join('out_aus', actor_name)
    os.makedirs(au_feat_save_folder, exist_ok=True)
    au_feats.to_csv(os.path.join(au_feat_save_folder, video_name+'.csv'), sep=';', header=True, index=False)

# Function to recursively get paths of all CSV files in a folder and its subdirectories
def get_csv_files(folder_path):
    csv_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                csv_paths.append(csv_path)
                print(f"Found CSV file: {csv_path}")  # Debug statement
    return csv_paths

# Function to combine CSV files into a single pickle file
def combine_csv_to_pkl(folder_path, output_file="data.pkl"):
    csv_files = get_csv_files(folder_path)
    combined_data = pd.DataFrame()

    # Process each CSV file
    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")  # Print each file being processed
        try:
            # Attempt to read each CSV file
            df = pd.read_csv(csv_file, sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback encoding if utf-8 fails
            df = pd.read_csv(csv_file, sep=';', encoding='ISO-8859-1')
        combined_data = pd.concat([combined_data, df], ignore_index=True)

    # Save the combined DataFrame as a pickle file
    with open(output_file, "wb") as f:
        pickle.dump(combined_data, f)

    print(f"Data from all CSV files combined and saved as {output_file}")

# Usage
folder_path = "out_aus"  # Replace with the path to your folder containing CSV files
combine_csv_to_pkl(folder_path)