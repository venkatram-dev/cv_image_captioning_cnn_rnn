import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        #print ('resnet modules except last',modules)
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        """
        sample size for each layers
        feature size after resnet torch.Size([10, 2048, 1, 1])
        features size afer view reshape torch.Size([10, 2048])
        features after embedding torch.Size([10, 256])
        """
        features = self.resnet(images)
        #print ('feature size after resnet',features.size())
        #print ('features',features)
        features = features.view(features.size(0), -1)
        #print ('features size afer view reshape',features.size())
        #print ('features.size(0)',features.size(0))
        #print ('features',features)
        features = self.embed(features)
        #print ('features after embedding',features.size())
        #print ('features',features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        #print ('embed_size',embed_size)
        #print ('hidden_size',hidden_size)
        #print ('vocab_size',vocab_size)
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers 
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        
        # As per https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html 
        # torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,        scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None)
        self.embed_layer = nn.Embedding(self.vocab_size, self.embed_size)
        # As per https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html lstm
        #input_size – The number of expected features in the input x
        #hidden_size – The number of features in the hidden state h
        #num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
    #bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
    #batch_first –If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: False
        
        self.lstm = nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, num_layers= self.num_layers, batch_first= True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.hidden_size,self.vocab_size)
    
    def forward(self, features, captions):
        """
        sample sizes for each layer   
        
        embed_size 256
        hidden_size 512
        vocab_size 8855
        features size torch.Size([10, 256])
        captions size torch.Size([10, 12])
        # above is 12 because in the batch selection, it has selected captions of length 12 
        # remove last dim /token corresponding to end
        reduced_captions size torch.Size([10, 11])
        embeddeded_captions size torch.Size([10, 11, 256])
        unsqueezed_features size torch.Size([10, 1, 256])
        # concatenate captions and features across dim 1
        embeddeded_captions_plus_features size torch.Size([10, 12, 256])
        embeddeded_captions_lstm size torch.Size([10, 12, 512])
        embeddeded_captions_dropout size torch.Size([10, 12, 512])
        embeddeded_captions_dropout_linear size torch.Size([10, 12, 8855])
        outputs.shape: torch.Size([10, 12, 8855])

        """
        
        #print ('features size',features.size())
        #print ('captions size',captions.size())
        
        #print ('captions',captions)
        
        # remove last dim /token corresponding to end
        reduced_captions = captions[:,:-1]
        
        #print ('reduced_captions size' , reduced_captions.size())
        #print ('reduced_captions',reduced_captions)
        
        embeddeded_captions = self.embed_layer(reduced_captions)
        
        #print ('embeddeded_captions size',embeddeded_captions.size())        
        #print ('embeddeded_captions ',embeddeded_captions)
            
        # add features
        
        unsqueezed_features = features.unsqueeze(1)
        
        #torch.set_printoptions(profile='full')
        
        #print ('unsqueezed_features size',unsqueezed_features.size())       
        #print ('unsqueezed_features ',unsqueezed_features)
        
        #torch.set_printoptions(profile='default')
        
        embeddeded_captions_plus_features = torch.cat((unsqueezed_features,embeddeded_captions),1)
                
        #print ('embeddeded_captions_plus_features size',embeddeded_captions_plus_features.size())        
        #print ('embeddeded_captions_plus_features',embeddeded_captions_plus_features)
        
       
        embeddeded_captions_lstm, (h, c) = self.lstm(embeddeded_captions_plus_features)
        #print ('embeddeded_captions_lstm size',embeddeded_captions_lstm.size())
               
        embeddeded_captions_dropout = self.dropout(embeddeded_captions_lstm)
       #print ('embeddeded_captions_dropout size',embeddeded_captions_dropout.size())
        
        embeddeded_captions_dropout_linear = self.fc(embeddeded_captions_dropout)
        #print ('embeddeded_captions_dropout_linear size',embeddeded_captions_dropout_linear.size())
        
        
        return embeddeded_captions_dropout_linear
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        """
        sample size for each layer - with 2 iterations
        
        inputs size torch.Size([1, 1, 512])
        inputs_after_lstm size torch.Size([1, 1, 512])
        inputs_after_dropout size torch.Size([1, 1, 512])
        inputs_after_linear size torch.Size([1, 1, 8855])
        inputs_after_linear_two_dim size torch.Size([1, 8855])
        max_pred_index tensor([ 0], device='cuda:0')
        max_pred_index size torch.Size([1])
        out_pred  0
        next input size torch.Size([1, 1])
        next input embedded size torch.Size([1, 1, 512])
        
        inputs size torch.Size([1, 1, 512])
        inputs_after_lstm size torch.Size([1, 1, 512])
        inputs_after_dropout size torch.Size([1, 1, 512])
        inputs_after_linear size torch.Size([1, 1, 8855])
        inputs_after_linear_two_dim size torch.Size([1, 8855])
        max_pred_index tensor([ 3], device='cuda:0')
        max_pred_index size torch.Size([1])
        out_pred  3
        next input size torch.Size([1, 1])
        next input embedded size torch.Size([1, 1, 512])
        
        """
        
        output_tensor_list = []
        
        for i in range(0,max_len):
        
            #print ('inputs size',inputs.size())
            
            inputs_after_lstm, states = self.lstm(inputs, states)
            #print ('inputs_after_lstm size',inputs_after_lstm.size())

            inputs_after_dropout = self.dropout(inputs_after_lstm)                        
            #print ('inputs_after_dropout size',inputs_after_dropout.size())

            inputs_after_linear = self.fc(inputs_after_dropout)
            #print ('inputs_after_linear size',inputs_after_linear.size())

            #reduce one dimension
            inputs_after_linear_two_dim  = inputs_after_linear.squeeze(1)
            #print ('inputs_after_linear_two_dim size',inputs_after_linear_two_dim.size())

            # get index position with max score
            # output is the index position with max score
            # this corresponds to 8854 indices as in the vocabulary - data_loader.dataset.vocab.idx2word 
            # for example , 0 means start, 3 means a, 857 means living
            max_pred_index = inputs_after_linear_two_dim.argmax(1)
            #print ('max_pred_index',max_pred_index)
            #print ('max_pred_index size',max_pred_index.size())

            #extract value from tensor
            max_pred_index_value = max_pred_index.item()
            out_pred = max_pred_index_value
            #print ('out_pred ',out_pred)

            #append index of the word 
            output_tensor_list.append(out_pred)
            
            #increase dimension again to pass as next input
            inputs = max_pred_index[None,:]

            #print ('next input',inputs)
            #print ('next input size',inputs.size())

            #pass through embedding layer
            inputs = self.embed_layer(inputs)

            #print ('next input embedded',inputs)
            #print ('next input embedded size',inputs.size())
            
            
        
        return output_tensor_list
        
        
