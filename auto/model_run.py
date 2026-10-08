import matplotlib.pyplot as plt
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage
import torch 
import torch.nn as nn
from LinearModel import LinearRegressionModel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Since the model is linear, only K_P needs to be used; K_I and K_D add noise
#Feel free to use a low K_I (<0.25) and moderate K_D (<1)
#If you change these in this file, change them to the same values in 'run_template.py'
K_P = 1
K_I = 0
K_D = 0
 
STEPS = 550
 

#WRITE CODE HERE
'''
This file is to train the model and save it for comparision

A linear regresion model predicts the desired acceleration from the car's current and desired velocities

The first 2 loops generate data for the model to train and test off of, created using the normal PID loop across different desired velocities.

This is followed by the loop that trains and tests the model.

Finally, some graphs are outputted to see the model's metrics

Aftering running this file, please run 'run_template.py' for the comparison between this model and the actual PID controller
'''

#Stores each of these after every step in the PID loop
velocities = []
errors = []
train_times = []
test_times = []

#Desired velocities for the training and testing cars respectively
train_cars = np.random.randint(0, 25, size=(5)).tolist()
test_cars = [5, 12, 8, 20, 22]

#random set chosen for visualization purposes
num  = np.random.randint(0,len(train_cars))
random_set = []
random_velocities = [] 
random_times = []

x_train = []
y_train = []
x_test = []
y_test = []

#Filling the training set
for index, new_car in enumerate(train_cars):
    car = make_car(desired_v=new_car, dt=0.1)

    #reset for graphs
    train_times = []
    for step in range(STEPS):
        desired_accleration, error = calculate_desired_acceleration(car, K_P, K_I, K_D)
        throttle = acceleration_to_throttle_percentage(desired_accleration)

        x_train.append([car["v"], car["desired_v"]])
        y_train.append([desired_accleration])  

        if index == num:
            random_set.append([car["v"], car["desired_v"]])

        car["error_prev"] = error
        update(car, throttle)

        velocities.append(car["v"])
        if index == num:
                random_velocities.append(car["v"])

        errors.append(error)
        train_times.append(car["t"])
    if index == num:
        random_times = list(train_times)
    


#filling the testing set
for new_car in test_cars:
    car = make_car(desired_v=new_car, dt=0.1)
    for step in range(STEPS):
        desired_accleration, error = calculate_desired_acceleration(car, K_P, K_I, K_D)
        throttle = acceleration_to_throttle_percentage(desired_accleration)

        x_test.append([car["v"], car["desired_v"]])
        y_test.append([desired_accleration])  

        car["error_prev"] = error
        update(car, throttle)

        velocities.append(car["v"])
        errors.append(error)
        test_times.append(car["t"])

#convert to tensors for the model
x_train = torch.FloatTensor(x_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
x_test = torch.FloatTensor(x_test).to(device)
y_test = torch.FloatTensor(y_test).to(device)
random_set = torch.FloatTensor(random_set).to(device)

#standardize data for ADAM
mean, std = x_train.mean(0), x_train.std(0)
x_train = (x_train - mean) / std
x_test  = (x_test - mean) / std
random_set = (random_set - mean) / std

torch.manual_seed(42)

model = LinearRegressionModel()
model.to(device)

epochs = 150
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.5) 

epoch_count = []
train_losses = []
test_losses = []

#Model loop
for epoch in range(epochs):
    #training
    model.train()

    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    #testing
    model.eval()
    with torch.inference_mode():
        test_pred = model(x_test)
        test_loss = loss_fn(test_pred, y_test)

    
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        print(f"Epoch {epoch} | train {loss.item()} | test {test_loss.item()}")


#Linear Model MSE loss graph
plt.plot(epoch_count, train_losses, label="train")
plt.plot(epoch_count, test_losses, label="test")
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()


#Scatter of the accelerations and PID behavior from a random train car
try:
    predicted = model(random_set).detach().numpy().flatten()
except:
    predicted = model(random_set).detach().cpu().numpy().flatten()

fig, (plot1, plot2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

plot1.plot(train_times, predicted, label="acceleration")
plot1.set_title(f"Predicted Acceleration vs Time, Desired Velocity of {train_cars[num]}")
plot1.set_ylabel("predicted acceleration")
plot1.legend()

plot2.plot(random_times, random_velocities, color="red", label="velocity")
plot2.axhline(y=train_cars[num], color="black", linestyle="--", label="desired velocity")
plot2.set_title(f"Normal PID Plot, Velocity vs Time, Desired Velocity of {train_cars[num]}")
plot2.set_xlabel("time")
plot2.set_ylabel("velocity")
plot2.legend()

plt.show()

torch.save(model.state_dict(), "./auto")