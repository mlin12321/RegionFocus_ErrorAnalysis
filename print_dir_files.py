import os

def read_files_in_directory(directory):
    with open("error_notes.txt", 'r') as f:
        for file in os.listdir(directory):
            print(file, file=f)

read_files_in_directory('debug')