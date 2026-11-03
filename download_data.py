import os
import shutil
import kagglehub
import pandas as pd

def setup_directories():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    dirs_to_create = [
        os.path.join(data_dir, 'xray'),
        os.path.join(data_dir, 'respiratory_audio'),
        os.path.join(data_dir, 'cough_audio')
    ]
    
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
        print(f"Ensured directory exists: {d}")
        
    return data_dir

def download_datasets():
    data_dir = setup_directories()
    
    print("\n" + "="*50)
    print("1. Downloading ICBHI Respiratory Sound Database...")
    print("="*50)
    # Download ICBHI dataset
    icbhi_path = kagglehub.dataset_download("vbookshelf/respiratory-sound-database")
    print(f"Downloaded ICBHI to cache: {icbhi_path}")
    
    # Move files to our project structure
    target_audio_dir = os.path.join(data_dir, 'respiratory_audio')
    
    # Kaggle extracts it into a specific structure. The audio files are usually in 'Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files'
    # We will search for all .wav and .txt files and move them.
    print(f"Copying respiratory audio files to {target_audio_dir} ...")
    copied_audio = 0
    for root, dirs, files in os.walk(icbhi_path):
        for file in files:
            if file.endswith('.wav') or file.endswith('.txt'):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_audio_dir, file)
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                copied_audio += 1
    print(f"Successfully copied {copied_audio} audio/annotation files.")

    print("\n" + "="*50)
    print("2. Downloading NIH Chest X-ray14 Database...")
    print("WARNING: This is a ~42GB dataset. This may take a long time!")
    print("="*50)
    
    # We download the dataset path rather than loading pandas directly so we get the images too
    nih_path = kagglehub.dataset_download("nih-chest-xrays/data")
    print(f"Downloaded NIH X-rays to cache: {nih_path}")
    
    target_xray_dir = os.path.join(data_dir, 'xray')
    print(f"Copying NIH Chest X-ray files to {target_xray_dir} ...")
    
    # Copy Data_Entry_2017.csv
    csv_src = os.path.join(nih_path, "Data_Entry_2017.csv")
    if os.path.exists(csv_src):
        shutil.copy2(csv_src, os.path.join(target_xray_dir, "Data_Entry_2017.csv"))
        print("Copied Data_Entry_2017.csv")
        
    print("\nSetup complete! Note: To save disk space, we linked/copied files from the Kagglehub cache.")

if __name__ == "__main__":
    download_datasets()
