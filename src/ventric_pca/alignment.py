
import os 
from ventric_pca.utils import process_all_subjects
from ventric_pca.config import INPUT_DIRECTORY, OUTPUT_DIRECTORY


subject_list = os.listdir(INPUT_DIRECTORY)

subject_list = [subject for subject in subject_list if subject!=".DS_Store"]

process_all_subjects(subject_list, INPUT_DIRECTORY, OUTPUT_DIRECTORY)

