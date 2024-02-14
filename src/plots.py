"""plots.py

:author: Ofir Paz
:version: 03.02.2024

This file is responsible for plotting the results of the project.
"""


################################ Import section ################################

import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data_preperation import crop_letter, FONT_DECODER, NUM_CLASSES
from architecture import *

############################# End of import section ############################


################################ Functions section #############################

def draw_boxes_on_image(image: np.ndarray, wordBB: np.ndarray, charBB: np.ndarray) -> None:
    """
    Draw bounding boxes on image.

    :param image: The image to draw the bounding boxes on.
    :param wordBB: The word bounding boxes assume shape: (#boxes, #points, (x,y)).
    :param charBB: The character bounding boxes assume shape: (#boxes, #points, (x,y)).
    :return: None.
    """

    # Create a figure.
    plt.figure(figsize=(10, 10))

    # Convert the image to int.
    image = image.astype(int)
    
    # Create an Axes object.
    ax = plt.subplot(1, 2, 1)
    ax.axis('off')
    ax.imshow(image)

    # Loop through the word bounding boxes and draw rectangles.
    for bb in wordBB:
        # Create a rectangle patch.
        rect = patches.Polygon(np.column_stack(bb.T),
                                closed=True, linewidth=2, edgecolor='g', facecolor='none')

        # Add the rectangle to the Axes.
        ax.add_patch(rect)

    ax = plt.subplot(1, 2, 2)
    ax.axis('off')
    ax.imshow(image)

    # Loop through the charecters bounding boxes and draw rectangles.
    for bb in charBB:
        # Create a rectangle patch.
        rect = patches.Polygon(np.column_stack(bb.T),
                                closed=True, linewidth=2, edgecolor='r', facecolor='none')

        # Add the rectangle to the Axes.
        ax.add_patch(rect)
        
    # Display the image with all bounding boxes.
    plt.show()


def plot_cropped_bb(data: dict, image_idx: int, idx: int, is_charBB: bool = True) -> None:
    """
    Plot a cropped image. The image is cropped according to the bounding box.
    The bounding box is either a word bounding box or a character bounding box.

    :param data: The data.
    :param image_idx: The index of the image containing the bounding boxes. Assuming valid index.
    :param idx: The index of the bounding box. Assuming valid index.
    :param is_charBB: A flag indicating if the bounding box is a character bounding box.
    :return: None.
    """

    # Get the big image.
    big_image = data['images'][image_idx]

    if is_charBB:
        
        # Get the char bounding box and the letter.
        bbox = data['charBBs'][image_idx][idx]
        txt = ''.join(data['txts'][image_idx])[idx]
        font = data['fonts'][image_idx][idx]
    
    else:
        
        # Get the word bounding box and the word.
        bbox = data['wordBBs'][image_idx][idx]
        txt = data['txts'][image_idx][idx]
        font = data['fonts'][image_idx][sum(len(data['txts'][image_idx][i]) for i in range(idx))]

    # Crop and rotate the image
    cropped_rotated_image = crop_letter(big_image, bbox)

    # Display the image
    plt.title(f'Cropped and Rotated Image.\ntext: \'{txt}\', font: {FONT_DECODER[font]} ({font})')
    plt.imshow(cropped_rotated_image)
    plt.show()


def plot_loss_histogram(model: Net) -> None:
    """
    Plot the loss histogram.

    :param model: The model.
    :return: None.
    """

    # Create a figure.
    plt.figure(figsize=(10, 5))

    # Plot the loss histogram.
    plt.plot(model.costs, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Histogram')
    plt.legend()
    plt.show()


def plot_roc_graph(prob_predictions, labels, set_name: str = '') -> None:
    """
    Plot the ROC graph on the given set.

    :param prob_predictions: The probabilities predictions of the model on some dataset.
    :param labels: The labels of the data.
    :param set_name: The name of the dataset to show the graph for.
    :return: None.
    """
    
    y_set_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))  # Binarize the output

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_set_bin[:, i], prob_predictions[:, i])  # type: ignore
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    plt.figure(figsize=(6, 6))
    for i in range(NUM_CLASSES):
        plt.plot(fpr[i], tpr[i], label=f'class: %d area = %0.2f)' %(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) {f"for {set_name}" if set_name else ""}')
    plt.legend(loc="lower right")
    plt.show()


def print_accuracy(prob_predictions, labels, set_name: str = '') -> None:
    """
    Print the accuracy of the model on the given set.

    :param prob_predictions: The probabilities predictions of the model on some dataset.
    :param labels: The labels of the data.
    :param set_name: The name of the dataset to show the graph for.
    :return: None.
    """

    # Get the number of correct predictions.
    num_correct_predictions = (prob_predictions.argmax(dim=1) == labels).sum().item()

    # Calculate the accuracy.
    acc = 100 * (num_correct_predictions / len(labels))  # type: ignore

    # Print the accuracy.
    print(f'The accuracy of the model {f"on {set_name}" if set_name else ""} is: {acc:.4f}')


def print_all_smart_accuracy(data: dict, prob_predictions, set_name: str = '') -> None:
    """
    Print the accuracy of the smart models on the given set.

    :param data: The original data.
    :param prob_predictions: The probabilities predictions of the model on some dataset.
    :param set_name: The name of the dataset to show the graph for.
    :return: None.
    """
    
    # Extract the labels.
    labels = torch.cat([torch.from_numpy(font).to(torch.long) for font in data['fonts']])

    # Extract the texts.
    txts = data['txts']

    # Extract the character bounding boxes.
    charBBs = data['charBBs']

    # Print the accuracy of the smart models.
    print_smart_accuracy(make_smart_predictions1(prob_predictions, txts), labels, 1, set_name)
    print_smart_accuracy(make_smart_predictions2(prob_predictions, txts), labels, 2, set_name)
    print_smart_accuracy(make_smart_predictions3(prob_predictions, txts, 1), labels, '3-1', set_name)
    print_smart_accuracy(make_smart_predictions3(prob_predictions, txts, 2), labels, '3-2', set_name)
    print_smart_accuracy(make_smart_predictions3(prob_predictions, txts, 3), labels, '3-3', set_name)
    print_smart_accuracy(make_smart_predictions4(prob_predictions, txts, charBBs), labels, 4, set_name)


def print_smart_accuracy(smart_predictions, labels, num_smart: str | float, 
                         set_name: str = '') -> None:
    """
    Print the accuracy of the smart model on the given set.

    :param smart_predictions: The smart predictions of the model on some dataset.
    :param labels: The labels of the data.
    :param num_smart: The number of the smart model.
    :param set_name: The name of the dataset.
    :return: None.
    """

    # Get the number of predictions.
    num_predictions = len(labels)
    
    # Get the number of correct predictions.
    num_correct_predictions = (smart_predictions == labels).sum().item()
    
    # Calculate the accuracy.
    acc = 100 * (num_correct_predictions / num_predictions)

    # Print the accuracy.
    print(f"[{set_name + ' ' if set_name else ''}SMART #{num_smart}] ACC: {acc:.4f}%")

############################ End of functions section ###########################