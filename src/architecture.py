"""architecture.py

:author: Ofir Paz
:version: 03.02.2024

This file is responsible for the architecture of the neural network,
as well as the training loop and the smart predictions.
"""


################################ Import section ################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
from data_preperation import get_data_loaders, PARENT_PATH, BATCH_SIZE, NUM_CLASSES

############################# End of import section ############################


################################ Globals section ###############################

# Set the device that will be used for training.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################ End of globals section ############################


################################ Classes section ###############################

class ResidualBlock(nn.Module):
    """ResidualBlock class.
    
    This class is responsible for creating a residual block for the ResNet
    architecture.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.to(DEVICE)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = torch.add(out, self.shortcut(residual))
        out = self.relu(out)
        
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 84, kernel_size=5, padding=2),
            nn.BatchNorm2d(84),
            nn.ReLU(inplace=True),
        )

        self.residual_blocks = nn.Sequential(
            ResidualBlock(84, 84),
            ResidualBlock(84, 114),
            nn.AvgPool2d(kernel_size=(2, 3), stride=(2, 3)),
            ResidualBlock(114, 114),
            ResidualBlock(114, 134, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(134 * 5 * 5, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Add dropout here
            nn.Linear(512, NUM_CLASSES)
        )
        
        self.epoch_counter = 0
        self.costs = []
        self.loss_func = nn.CrossEntropyLoss()
        self.set_optimizer()
        self.to(DEVICE)

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = self.residual_blocks(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def set_optimizer(self, lr=1e-4, weight_decay=1e-4):
        """
        Set the optimizer.

        :param lr: The learning rate.
        :param weight_decay: The weight decay.
        :return: None.
        """

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def fit(self, data_loader, num_epochs=10, lr=0.0025, \
            weight_decay=0.001, print_cost=False, print_stride=1):
        """
        Train the model.
        
        :param data_loader: The data loader.
        :param num_epochs: The number of epochs.
        :param lr: The learning rate.
        :param weight_decay: The weight decay.
        :param print_cost: A flag indicating if the cost should be printed.
        :param print_stride: The stride for printing the cost.
        :return: None.
        """
        
        self.train()
        self.set_optimizer(lr=lr, weight_decay=weight_decay)
        max_epoch = self.epoch_counter + num_epochs
        for epoch in range(num_epochs):
            self.epoch_counter += 1
            running_loss = 0.
            for i, mini_batch in enumerate(data_loader):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = mini_batch

                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                predicts = self(inputs)
                
                loss = self.loss_func(predicts, labels)
                loss.backward()
                self.optimizer.step()

                # Calc loss
                lloss = loss.item()
                running_loss += lloss * inputs.size(0)

                if print_cost and i % 4 == 0:
                    print(f"\r[epoch: {self.epoch_counter}/{max_epoch} MB: {i+1}] Loss: {lloss}", end='')

            epoch_loss = running_loss / (len(data_loader) * BATCH_SIZE)
            self.costs.append(epoch_loss)
            
            if print_cost and (epoch % print_stride == 0 or epoch == num_epochs - 1):
                print(f"\r[epoch: {self.epoch_counter}/{max_epoch}] Total Loss: {epoch_loss}")
        
        print()

    @torch.no_grad()
    def get_probabilities(self, data_loader: DataLoader, is_test: bool = False):
        """
        Get the probabilities of the model's predictions.

        :param data_loader: The data loader.
        :param is_test: A flag indicating if the data loader is for the test set.
        :return: The probabilities and the labels.
        """
        self.eval()

        # Create a tensor to hold the predictions and one for labels.
        prob_predictions = torch.zeros((len(data_loader.dataset), NUM_CLASSES))  # type: ignore
        labels_all = torch.zeros((len(data_loader.dataset))) if not is_test else None  # type: ignore

        for i, data in enumerate(data_loader):
            
            inputs, labels = None, None
            if not is_test:
                inputs, labels = data
                labels = labels.to(DEVICE)

            else:
                inputs = data
            
            inputs = inputs.to(DEVICE)
            outputs = self(inputs)
            
            # Add the predictions to the tensors
            start_index = i * BATCH_SIZE
            end_index = start_index + BATCH_SIZE if i < len(data_loader) - 1 else \
                len(data_loader.dataset)  # type: ignore
            
            prob_predictions[start_index:end_index] = F.softmax(outputs.data, dim=1) 
            
            if not is_test:
                
                labels_all[start_index:end_index] = labels  # type: ignore

        return prob_predictions, labels_all

    def plot_loss_histogram(self) -> None:
        """
        Plot the loss histogram.

        :return: None.
        """

        plt.plot(self.costs)
        plt.xlabel('Epoch')
        plt.ylabel('Train loss')
        plt.title('Loss Histogram')
        plt.show()

    def print_accuracy(self, data_loader, set_name: str) -> float:
        """
        Print the accuracy of the model on a given data loader.

        :param data_loader: The data loader.
        :param set_name: The name of the set.
        :return: The accuracy.
        """

        # Get the probabilities and labels.
        prob_predictions, labels = self.get_probabilities(data_loader)

        # Get the number of correct predictions.
        num_correct_predictions = (prob_predictions.argmax(dim=1) == labels).sum()

        # Calculate the accuracy.
        acc = 100 * (num_correct_predictions / len(labels))  # type: ignore

        print(f"{f'[{set_name}] ' if set_name else ''}ACC: {acc:.4f}%")

        return acc

############################ End of classes section ############################


############################### Functions section ##############################
        
def full_train_loop() -> Net:
    """Full train loop function.
    
    This function is responsible for running the full training loop.

    :return: None.
    """
    
    # Set the hyperparameters. These hyperparameters were chosen after
    # experimenting with different values.
    lr = 0.0025; weight_decay = 0.005; num_steps = 4; epochs_per_step = 7

    # Create an instance of the model.
    net = Net()

    # Get the data loaders.
    train_loader, valid_loader, _ = get_data_loaders()

    for step in range(num_steps):

        # Print the current step's hyperparameters.
        print(f'[STEP: {step + 1}]: lr: {lr}, weight_decay: {weight_decay}')
        
        # Train the model for a few epochs.
        net.fit(train_loader, epochs_per_step, lr, weight_decay, print_cost=True)

        # Print the accuracy of the model on the training and validation sets.
        net.print_accuracy(train_loader, 'TRAIN')
        net.print_accuracy(valid_loader, 'VALID')

        print()  # New line.

        # Update the learning rate and weight decay.
        lr /= 5
        weight_decay += 0.005

        # Decrease the number of epochs for the next steps.
        epochs_per_step -= 1

    return net


def make_smart_predictions1(font_probabilities, txts):
    """
    Make smart predictions based on the font probabilities and the text.
    We know that each txt contains only one font. Therefore, we can use the
    text to make better predictions.

    :param font_probabilities: The font probabilities.
    :param txts: The texts.
    :return: The smart predictions.
    """
    
    # Get the number of samples.
    num_samples = font_probabilities.shape[0]

    # Create a tensor to hold the predictions.
    predictions = torch.zeros(num_samples, dtype=torch.long)

    # Loop through the txts.
    i = 0  # The index of the current character.
    for image_txt in txts:
        for txt in image_txt:

            # Get the probabilities of the fonts in the txt.
            probs = font_probabilities[i : i + len(txt)]

            # Get the smart prediction.
            smart_prediction = torch.argmax(torch.sum(probs, dim=0))
            
            # Set the smart prediction.
            predictions[i : i + len(txt)] = smart_prediction

            # Update the index.
            i += len(txt)

    return predictions


def make_smart_predictions2(font_probabilities, txts):
    """
    Make smart predictions based on the font probabilities and the text.
    We know that each txt contains only one font. Therefore, we can use the
    text to make better predictions.

    :param font_probabilities: The predicted labels. 
    :param txts: The texts.
    :return: The smart predictions.
    """
    
    # Get the predicted labels.
    predicted_labels = torch.argmax(font_probabilities, dim=1)

    # Get the number of samples.
    num_samples = predicted_labels.shape[0]

    # Create a tensor to hold the smart predictions.
    smart_predictions = torch.zeros(num_samples, dtype=torch.long)

    # Loop through the txts.
    i = 0  # The index of the current character.
    for image_txt in txts:
        for txt in image_txt:

            # Get the probabilities of the fonts in the txt.
            word_predictions = predicted_labels[i : i + len(txt)]
            
            # Get the prediction that is made the most in the word.
            most_common_prediction = torch.argmax(torch.bincount(word_predictions))

            # Get the number of occurences of the most common prediction.
            num_occurences = torch.sum(word_predictions == most_common_prediction)


            if num_occurences == 1: 
                # If the most common prediction appears only once,
                # then we can't make a smart prediction.
                smart_predictions[i : i + len(txt)] = word_predictions

            else:
                # Set the smart prediction.
                smart_predictions[i : i + len(txt)] = most_common_prediction

            # Update the index.
            i += len(txt)

    return smart_predictions


def make_smart_predictions3(font_probabilities, txts, smart_fac: int = 1):
    """
    Make smart predictions based on the font probabilities and the text.
    We know that each txt contains only one font. Therefore, we can use the
    text to make better predictions.

    :param font_probabilities: The predicted labels. 
    :param txts: The texts.
    :param smart_fac: The smart factor. If the most common prediction appears only once,
                      then we can't make a smart prediction.
    :return: The smart predictions.
    """
    
    # Get the predicted labels.
    predicted_labels = torch.argmax(font_probabilities, dim=1)

    # Get the number of samples.
    num_samples = predicted_labels.shape[0]

    # Create a tensor to hold the smart predictions.
    smart_predictions = torch.zeros(num_samples, dtype=torch.long)

    # Loop through the txts.
    i = 0  # The index of the current character.
    for image_txt in txts:
        for txt in image_txt:

            # Get the probabilities of the fonts in the txt.
            word_predictions = predicted_labels[i : i + len(txt)]
            
            # Get the prediction that is made the most in the word.
            most_common_prediction = torch.argmax(torch.bincount(word_predictions))

            # Get the number of occurences of the most common prediction.
            num_occurences = torch.sum(word_predictions == most_common_prediction)


            if num_occurences <= smart_fac: 
                # If the most common prediction appears only once,
                # then make smart prediction #1.
                smart_predictions[i : i + len(txt)] = torch.argmax(torch.sum(font_probabilities[i : i + len(txt)], dim=0))

            else:
                # Set the smart prediction.
                smart_predictions[i : i + len(txt)] = most_common_prediction

            # Update the index.
            i += len(txt)

    return smart_predictions


def make_smart_predictions4(font_probabilities, txts, charBBs: list[np.ndarray]):
    """
    Make smart predictions based on the font probabilities and the text.
    We know that each txt contains only one font. Therefore, we can use the
    text to make better predictions.
    This smart prdiction is calculating the probabilty based on the weighted of the
    probabilities in respect to the size of the letter.

    :param font_probabilities: The predicted labels.
    :param txts: The texts.
    :param charBBs: The character bounding boxes assume shape: [#images, (#boxes, #points, (x,y))].
    :return: The smart predictions.
    """

    # Get the width and height of the bounding boxes
    widths = np.array([int(np.linalg.norm(bbox[1] - bbox[0])) or 1 for image_charBBs in charBBs for bbox in image_charBBs])
    heights = np.array([int(np.linalg.norm(bbox[3] - bbox[0])) or 1 for image_charBBs in charBBs for bbox in image_charBBs])

    # Calculate the size of the bounding boxes, desired shape is (#boxes, 1)
    sizes = (widths * heights).reshape(-1, 1)

    # Get the number of samples.
    num_samples = font_probabilities.shape[0]

    # Create a tensor to hold the predictions.
    predictions = torch.zeros(num_samples, dtype=torch.long)

    # Loop through the txts.
    i = 0  # The index of the current character.
    for image_txt in txts:
        for txt in image_txt:

            # Get the probabilities of the fonts in the txt.
            probs = font_probabilities[i : i + len(txt)]

            # Get the weighted probabilities.
            weighted_probs = probs * sizes[i : i + len(txt)]

            # Get the smart prediction.
            smart_prediction = torch.argmax(torch.sum(weighted_probs, dim=0))
            
            # Set the smart prediction.
            predictions[i : i + len(txt)] = smart_prediction

            # Update the index.
            i += len(txt)

    return predictions


def save_model(model: Net, name: str) -> None:
    """
    Save the model to a file using the pickle module.

    :param model: The model to save.
    :param name: The name of the file.
    :return: None
    """

    with open(PARENT_PATH + f'\\models\\{name}.pkl', 'wb') as f:
        pickle.dump(model, f)


def load_model(name: str) -> Net:
    """
    Load the model from a pickle file.

    :param name: The name of the file.
    :return: None
    """

    with open(PARENT_PATH + f'\\models\\{name}.pkl', 'rb') as f:
        model = pickle.load(f)

    return model

########################### End of functions section ###########################