"""input_output.py

:author: Ofir Paz
:version: 10.02.2024

This file is responsible for the input and output of the project.
"""


################################ Import section ################################

import os
import re
import pandas as pd
from data_preperation import *
from architecture import Net, full_train_loop, save_model, load_model
from plots import *

############################# End of import section ############################

################################ Functions section #############################

############################ Menus section ###########################

def main_menu() -> bool:
    """Main menu function.

    This function is responsible for printing the main menu and getting the user's
    input.

    :return: True if the user wants to continue, False otherwise.
    """

    # Printing the main menu and getting the user's input.
    option = menu_handle("Train the model", "Create datasets", "Visualize data",
                       "Plot histograms", "Create submission file", "Exit")
    
    # Handling the user's input.
    match option:
        case 1:
            training_option()
        case 2:
            creating_datasets_option()
        case 3:
            visualize_data_option()
        case 4:
            plot_histograms_option()
        case 5:
            create_submission_file_option()
        case 6:
            print("Exiting...\n")

    return not (option == 6)


def visualzie_data_menu(data: dict) -> bool:
    """Visualize data menu function.

    This function is responsible for printing the visualize data menu and getting the user's
    input.

    :param data: The data.
    :return: True if the user wants to continue, False otherwise.
    """

    # Get the ploting option from the user.
    option = menu_handle("Plot a large image with bounding boxes",  
                         "Plot a cropped word", "Plot a cropped letter", "Back to main menu")
    
    # Printing new line.
    print()

    # Handling the user's input
    match option:
        case 1:
            idx = take_idx_input(0, len(data['images']) - 1, "large image")
            print("\nPlotting the large image with bounding boxes...")
            draw_boxes_on_image(data['images'][idx], data['charBBs'][idx], data['wordBBs'][idx])
            print_done()

        case 2:
            img_idx = take_idx_input(0, len(data['images']) - 1, "image containing the word")
            word_idx = take_idx_input(0, len(data['wordBBs'][img_idx]) - 1, "of the word in the image")
            print("\nPlotting the cropped word...")
            plot_cropped_bb(data, img_idx, word_idx, is_charBB=False)
            print_done()

        case 3:
            img_idx = take_idx_input(0, len(data['images']) - 1, "image containing the letter")
            char_idx = take_idx_input(0, len(data['charBBs'][img_idx]) - 1, "of the letter in the image")
            print("\nPlotting the cropped letter...")
            plot_cropped_bb(data, img_idx, char_idx, is_charBB=True)
            print_done()

        case 4:
            print("Returning to main menu...\n")

    return option != 4


def plot_histograms_menu(model: Net, train_outputs: tuple, val_outputs: tuple) -> bool:
    """Plot histograms menu function.

    This function is responsible for printing the plot histograms menu and getting the user's
    input.

    :param model: The model.
    :param train_outputs: The train outputs (train_probabilities, train_labels).
    :param val_outputs: The validation outputs (val_probabilities, val_labels).
    :return: True if the user wants to continue, False otherwise.
    """

    # Get the ploting option from the user.
    option = menu_handle("Plot the loss histogram", "Plot ROC graph", "Show the accuracy",
                         "Show the accuracy with smart algorithms", "Back to main menu")
    
    # Printing new line.
    print()

    # Handling the user's input
    match option:
        case 1:
            print("Plotting the loss histogram...")
            plot_loss_histogram(model)
            print_done()

        case 2:
            print("Plotting the ROC graph...")
            plot_roc_graph(*train_outputs, set_name='TRAIN')
            plot_roc_graph(*val_outputs, set_name='VALIDATION')
            print_done()

        case 3:
            print("Printing the accuracy...")
            print_accuracy(*train_outputs, set_name='TRAIN')
            print_accuracy(*val_outputs, set_name='VALIDATION')
            print_done()

        case 4:
            print("Printing the accuracy on the valid loader with smart algorithms...")
            print("First, load the unprocessed data...")
            train_data, valid_data = split_data(get_data_from_h5())
            print_done()
            print_all_smart_accuracy(train_data, train_outputs[0], set_name='TRAIN')
            print()  # New line.
            print_all_smart_accuracy(valid_data, val_outputs[0], set_name='VALIDATION')
            print()  # New line.
            print_done()

        case 5:
            print("Returning to main menu...\n")

    return option != 5

######################## End of menus section ########################

########################### Options section ##########################

def training_option() -> None:
    """Training option function.

    This function is responsible for handling the training option.
    
    :return: None.
    """

    # Check if the datasets are saved
    if not check_for_saved_datasets():
        creating_datasets_option()
    
    if check_for_saved_datasets():
        print("The datasets are ready for training.")
        print("Would you like to train the model now?")

        if take_y_n_input():
            print("Training the model... This process may take up to 10 minutes.")
            print("The model will be trained for 4 steps, each step will have " + \
                   "a different learning rate and weight decay.")
            print("The number of total epochs will be 22.\n")
            net = full_train_loop()
            print_done()

            print("Would you like to save the model?")
            if take_y_n_input():
                file_name = get_legal_file_name_input('models', 'pkl', must_exist=False)
                print(f"Saving the model in the models directory as {file_name}.pkl...")
                save_model(net, file_name)
                print_done()


def creating_datasets_option() -> None:
    """Creating datasets option function.

    This function is responsible for handling the creating datasets option.
    
    :return: None.
    """

    # Check if the datasets are saved
    if not check_for_saved_datasets():
        print("\nThe pytorch datasets have not been created yet.")
        print("Would you like to create them now? This process may take up " + \
               "to a minute depending on your machine.")
        
        if take_y_n_input():
            print("\nCreating the pytorch datasets...")
            get_datasets()
            print_done()

    else:
        print("\nThe pytorch datasets have already been created.")
        print("Would you like to recreate them?")

        if take_y_n_input():
            print("\nRecreating the pytorch datasets...")
            print("Deleting the saved datasets...")
            delete_saved_datasets()
            print_done()
            print("Creating the pytorch datasets...")
            get_datasets()
            print_done()


def visualize_data_option() -> None:
    """Visualize data option function.

    This function is responsible for handling the visualize data option.
    
    :return: None.
    """
    
    # Get the data from the h5 file.
    print("\nLoading the data...")
    data = get_data_from_h5()
    print_done()

    while visualzie_data_menu(data):
        pass


def plot_histograms_option() -> None:
    """Plot histograms option function.

    This function is responsible for handling the plot histograms option.
    
    :return: None.
    """

    # Check if the datasets are saved.
    if not check_for_saved_datasets():
        creating_datasets_option()

    if not check_for_saved_datasets():
        print("The datasets are not ready for plotting histograms.")
        return
    
    else:
        model = load_saved_model()

        if model:
            print("Making preprocessing for the plots...")
            train_loader, val_loader, _ = get_data_loaders(shuffle_train=False)
            train_outputs = model.get_probabilities(train_loader)
            val_outputs = model.get_probabilities(val_loader)
            print_done()

            while plot_histograms_menu(model, train_outputs, val_outputs):
                pass


def create_submission_file_option() -> None:
    """Create submission file option function.

    This function is responsible for handling the create submission file option.
    
    :return: None.
    """

    # Check if the datasets are saved.
    if not check_for_saved_datasets():
        creating_datasets_option()

    if not check_for_saved_datasets():
        print("The datasets are not ready for creating a submission file.")
        return
    
    else:
        model = load_saved_model()

        if model:
            print("Would you like to create the submission file?")
            print("The submission file will be created using the algorithm (1),")
            print("Which was dicussed about in the report. It was chosen since all the algorithms")
            print("Behave similarly, and this algorithm is the easiest to implement.")
            if take_y_n_input():
                file_name = get_legal_file_name_input('submissions', 'csv', must_exist=False)

                if file_name:
                    print("Creating the submission file... this process may take up to 1 minute(s).")
                    create_submission_file(model, file_name)
                    print_done()

                else:
                    print("Cancelling, returning to main menu...\n")

####################### End of options section #######################

###################### Utility functions section #####################

def menu_handle(*options) -> int:
    """Menu handle function.

    This function is responsible for printing the menu and getting the user's
    input.

    :param options: The options to display in the menu.
    :return user_input: The user's input. 
    """

    width = 60 # adjust this according to your screen size
    print("Please select an option from the menu:".center(width))
    print("-" * width)

    for i, option in enumerate(options):
        print(f"{i + 1}. {option}".center(width))

    print("-" * width, end="\n\n")

    # Get the user's input
    user_input = input("Enter your choice: ")

    # Validate the user's input
    try:
        # Convert the user's input to an integer
        user_input = int(user_input)

        # Check if the user's input is in a valid range
        if user_input < 1 or user_input > len(options):
            print(f"Invalid input! Input must be between 1 and {len(options)}, Please try again.\n")     
            user_input = menu_handle(*options)

    # Handle invalid input
    except ValueError:
        print("Invalid input! Input must be an integer. Please try again.\n")
        user_input = menu_handle(*options)

    finally: return user_input  # type: ignore


def create_submission_file(model: Net, file_name: str) -> None:
    """Create submission file function.

    This function is responsible for creating a submission file with the given
    model and smart algorithm (1).

    :param test_outputs: The test outputs.
    :param file_name: The name of the submission file.
    :return: None.
    """

    # Get the test data from the h5 file.
    test_data = get_data_from_h5(train=False)

    # Get the dataloader.
    _, _, test_loader = get_data_loaders()

    # Get the probabilities of the fonts.
    prob_predictions, _ = model.get_probabilities(test_loader, is_test=True)

    # Get the predictions.
    smart_predictions = make_smart_predictions1(prob_predictions, test_data['txts'])

    # Create a dataframe to hold the predictions.
    new_df = pd.DataFrame({'ind': range(len(smart_predictions)), 'font': smart_predictions})

    # Save the dataframe to a csv file.
    new_df.to_csv(PARENT_PATH + 'submissions\\' + file_name + '.csv', index=False)  


def load_saved_model() -> Net | None:
    """Load saved model function.
    
    This function is responsible for loading a saved model
    from a pickle file.

    :return: The loaded model, or None if the user wants to exit.
    """

    # Check if there are saved models (a saved model is a file that ends with .pt).
    if not any(file.endswith(".pkl") for file in os.listdir(PARENT_PATH + '\\models')):
        print("There are no saved models. Please train a model first.")
        print("This project comes with a pre-trained model. The program did not find " + \
              "this model as well, perhaps it was deleted?")
        print("if so, redownload the project.", end="\n\n")
        return None

    # Ask the user for the model name.
    print("Please enter the name of the model you would like to load.")

    # Present the user with the available models. 
    print("The available models are:\n")
    for file in os.listdir(PARENT_PATH + '\\models'):
        if file.endswith(".pkl"):
            print(f"-\t{file[:-len('.pkl')]}")

    # Print a new line.
    print()

    # Get the user's input.
    file_name = get_legal_file_name_input('models', 'pkl', must_exist=True)

    # Check if the user wants to exit.
    if not file_name:
        return None
    
    # Load the model.
    print(f"Loading the model from the models directory as {file_name}.pkl...")
    model = load_model(file_name)
    print_done()

    return model


def get_legal_file_name_input(parent_dir_name: str, end: str, must_exist: bool) -> str:
    """Get legal file name input function.

    This function is responsible for getting a legal file name input from the user.
    A legal file name contains only letters and numbers.

    :param parent_dir_name: The name of the parent directory of the file.
    :param end: The end of the file name.
    :param must_exist: Whether the file must exist, or must not exist.
    :return: The user's input.
    """

    print("A legal file name contains only letters and numbers.")
    print("The file name must", "exist." if must_exist else "be new.")

    while True:
        file_name = input("Please enter a legal file name, or \'exit\' to go back: ")
        print()  # New line.

        # Check if the file name is legal
        if re.search(r'[\\/:*?\'\"<>|.]', file_name) is not None:
            print("Invalid input! Please enter a legal file name.")
            print("A legal file name contains only legal characters for file names.")
            continue

        if must_exist:
            if file_name == 'exit':
                print("Exiting...", end="\n\n")
                return ''
            
            elif not os.path.exists(PARENT_PATH + f'\\{parent_dir_name}\\{file_name}.{end}'):
                print("The file does not exist! Please enter a name of an existing file.")
                continue
        
        else:
            if os.path.exists(PARENT_PATH + f'\\{parent_dir_name}\\{file_name}.{end}'):
                print("The file already exists! Please enter a new name.")
                continue

        return file_name


def take_y_n_input() -> bool:
    """Take y/n input function.

    This function is responsible for getting a yes or no input from the user.

    :return: The user's input.
    """

    while True:
        ans = input("\033[93m" + "(y/n) " + "\033[0m" + ": ")
        
        try:
            if ans.lower() == "y":
                print()  # New line.
                return True

            elif ans.lower() == "n":
                print()  # New line.
                return False
            
            else:
                raise ValueError
            
        except ValueError:
            print("Invalid input! Please enter 'y' or 'n'.")


def take_idx_input(down_bound: int, up_bound: int, context: str = '') -> int:
    """Take index input function.
    
    This function is responsible for getting an index input from the user.

    :param down_bound: The lower bound of the index.
    :param up_bound: The upper bound of the index.
    :param context: The context of the index.
    :return: The user's input.
    """

    # Get the index from the user.
    while True:
        idx = input(f"Please enter the index{f' of the {context}' if context else ''}: ")

        try:
            idx = int(idx)
            
            # Check if the index is in a valid range.
            if idx < down_bound or idx > up_bound:
                raise ValueError

            return idx

        except ValueError:
            print(f"Invalid index! The index should be between {down_bound} and {up_bound}.\n")


def print_done() -> None:
    """Print done function.

    This function is responsible for printing "Done!"
    in green coloring.

    :return: None.
    """

    print("\033[92m" + "Done!" + "\033[0m", end="\n\n")

################## End of utility functions section ##################

########################### End of functions section ###########################