"""run.py

Final Project - Computer Vision (22928)

:author: Ofir Paz
:version: 03.02.2024

This file is the main entry point for the project. It is responsible for
running the main program loop and handling the user's input.

To run this code, simply run this file. No arguments are required.

Required packaged to run the code:

    - torch
    - torchvision
    - numpy
    - opencv-python
    - sklearn
    - matplotlib
    - pickle
    - h5py
    - pandas
    - tqdm
"""


################################ Import section ################################

# Printing outside of the main function is bad practice,
# however, it is necessary in this case because the user
# needs to be informed of the time that the importing will take.
# pytorch is a heavy library and it takes a few seconds to import.

# Print the welcome message.
print("\nWelcome to the Openu CV project!\n")
print("Importing the required modules...")

from architecture import Net, ResidualBlock
from input_output import main_menu, print_done

############################# End of import section ############################


############################### Functions section ##############################

def main() -> None:
    """Main function.

    This function is the main entry point for the project. It is responsible for
    running the main program loop and handling the user's input.
    """

    # Inform the user that the importing is done.
    print_done()

    # Main program loop.
    while main_menu():
        pass

########################### End of functions section ###########################


################################# Main section #################################

if __name__ == "__main__":
    main()

############################## End of main section #############################