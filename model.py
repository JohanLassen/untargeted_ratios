import torch
from torch import nn



class dnn(nn.Module):

    def __init__(
        self, 
        in_features,
        out1,
        dropout):
        
        super(dnn, self).__init__()
        
        #self.norm = nn.LayerNorm(in_features)
        
        self.linear1 = nn.Sequential( 
            nn.Linear(
                in_features = in_features, 
                out_features = out1,
                bias = False
                )
            )
        
        
        out2 = int((out1*(out1-1))/2) // 2

        self.norm = nn.LayerNorm(int((out1*(out1-1))/2))

        self.linear2 = nn.Sequential( 
            nn.Linear(
                in_features = int((out1*(out1-1))/2), 
                out_features = out2,
                bias = False
                ),
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=out2),
            nn.Dropout(p=dropout) #0.1
            )
        
        
        self.linear3 = nn.Sequential( 
            nn.Linear(
                in_features = out2, 
                out_features = out2,
                bias = False
                ),
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=out2),
            nn.Dropout(p=dropout) #0.1
            )
        
        
        self.output = nn.Sequential( 
            nn.Linear(
                in_features = out2, 
                out_features = 1
                )
            )
        
    def ratio(self, x):
        for i in range(x.shape[1]-1):
            m = (x[:,i+1:]+1)/(x[:,i].view(-1,1)+1)
            if i == 0:
                new = m
            else:
                new = torch.cat((new, m), 1)
        return(new)

    def forward(self, x):
        x = self.linear1(x)
        x = self.ratio(x)
        x = self.norm(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.output(x)
        return x    
