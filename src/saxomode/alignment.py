
import os 
from saxomode.utils import process_all_subjects

def main(
        input_directory=os.getcwd()+'/results/', 
        output_directory=os.getcwd()+'/Aligned_Models/'):
    """
    Align on all subjects to a common coordinate system.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    subject_list = os.listdir(input_directory)
    subject_list = [subject for subject in subject_list if subject != ".DS_Store"]

    process_all_subjects(subject_list, input_directory, output_directory,plot=True)

if __name__ == "__main__":
    main()
