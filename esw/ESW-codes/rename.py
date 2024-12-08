import os
import os.path

def rename_files(folder_path):
    """
    Renames all files in the given folder, keeping the first 6 characters of the filename and the original file extension.
    
    Parameters:
    folder_path (str): The path to the folder containing the files to be renamed.
    """
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            base, ext = os.path.splitext(filename)
            new_filename = base[:19] + ext
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
            print(f"Renamed {filename} to {new_filename}")

# Example usage
folder_path = "/home/chaitu/Desktop/ESW-pics/final2/test/images"
rename_files(folder_path)
