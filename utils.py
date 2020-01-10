import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F 


class Model(nn.Module):
    def __init__(self, input_size=65):
        super(Model, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
    
    def predict(self,x):
        #Apply softmax to output. 
        pred = F.softmax(self.forward(x))
        ans = []
        #Pick the class with maximum weight
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)
    

def train_model(model, X_train, y_train, criterion, optimizer, epochs=50):
    #List to store losses
    losses = []

    model.cuda()
    X_train = X_train.cuda()
    y_train = y_train.cuda()

    for i in range(epochs):
        #Precit the output for Given input
        y_pred = model.forward(X_train)
        #Compute Cross entropy loss
        loss = criterion(y_pred,y_train)
        #Add loss to the list
        losses.append(loss.item())
        #Clear the previous gradients
        optimizer.zero_grad()
        #Compute gradients
        loss.backward()
        #Adjust weights
        optimizer.step()

        if i%10==0:
            print('Epoch {} completed with loss: {}'.format(i, loss.item()))
            
    return model.cpu(), losses


def adv_train(model, X_train, y_train, criterion, optimizer, target_explanation, epochs=50, beta = 0.00000008):
    #List to store losses
    losses = []

    model.cuda()

    for i in range(epochs):
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        target_explanation.cuda()
        X_train.requires_grad = True
        
        #Precit the output for Given input
        y_pred = model.forward(X_train)
        #Compute Cross entropy loss
        loss1 = criterion(y_pred,y_train)
        #Compute manipulation loss
        explanation = get_heatmaps_train(model, X_train)
        loss2 = F.mse_loss(explanation, target_explanation.cuda()).cuda()
        #total loss
        loss = (1-beta) * loss1 + (beta) * loss2
        #Add loss to the list
        losses.append(loss.item())
        #Clear the previous gradients
        optimizer.zero_grad()
        #Compute gradients
        loss.backward(retain_graph=False)
        #Adjust weights
        optimizer.step()

        if i%10==0:
            print('Epoch {} completed with loss: {}'.format(i, loss.item()))
            
    return model.cpu(), losses


def get_heatmaps_train(model, samples):
    samples.requires_grad = True
    output = model(samples)
    pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
    out_rel = torch.eye(output.shape[1])[pred].cuda()
    one_hot = torch.sum(output * out_rel)

    grad = torch.autograd.grad(one_hot, samples, create_graph=True)

    return torch.abs(torch.sum(grad[0], dim=0))


def get_heatmaps_test(model, samples):
    samples.requires_grad = True
    output = model(samples)
    pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
    out_rel = torch.eye(output.shape[1])[pred]
    one_hot = torch.sum(output * out_rel)

    grad = torch.autograd.grad(one_hot, samples, create_graph=True)

    return torch.abs(torch.sum(grad[0], dim=0))



 