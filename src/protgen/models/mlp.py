import torch
from torch import nn
import torch.nn.functional as F

####################################################################################################
########################################3 IMPORTS #####################################################
#####################################################################################################


class MLP(nn.Module):
    def __init__(
            self,
            vocabulary_size = 23, # BOS + EOS + PAD + 20AA
            embedding_size = 24, # Chosen based on ppt sent but as long as larger than 23 should be fine
            input_context_size = 3, #The look back window size, for the MLP it will look at the 3 previous 
            num_hidden = 128, #Neurons the MLP will have
            padding_id = 0, #Will be used when creating the embedding adn in forward
            step_size = 1 
            ):
        super().__init__()

        self.input_context_size = input_context_size 

        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_id) #Avoids numerical relation between inputs

        self.step_size = step_size

        self.padding_id = padding_id

        # We set out our variables and now need to create our network

        ###### Constants used for self.net #######################
        self.num_hidden = num_hidden
        self.vocabulary_size = vocabulary_size
        self.input_features = input_context_size*embedding_size
        ###################################################

        self.net = nn.Sequential(
            nn.Linear(self.input_features, self.num_hidden),
            nn.Tanh(), #
            nn.Linear(self.num_hidden, self.vocabulary_size)
        )

    def forward(self, x):
        ############################################################################################3
        #To do the forwared we first need the sequences in three to be able to predict the next value, additionally if we want to predict
        #the first aa in the sequence we will need to add padding before hand so that we get a sequence of 3 as the input, 
        #eg    [BOS, M, K, V, EOS]
        # For M we wel need PAD PAD BOS
        # For K we will need PAD BOS K
        ############################################################################################3

        padded = F.pad(input=x, pad =(self.input_context_size-1,0), mode='constant', value=self.padding_id) ## Adds Padding to the left

        windows_3s = padded.unfold(1, size = self.input_context_size, step = self.step_size)   # builds windoes
        #print(windows_3s)  check if its working properly

        embed = self.embedding(windows_3s)

        final = embed.flatten(start_dim=2)

        return self.net(final)

