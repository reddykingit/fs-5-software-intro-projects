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

K_P = 1
K_I = 0
K_D = 0
 
STEPS = 550


#WRITE CODE HERE
'''
RUN THIS ONLY AFTER 'model_run.py' HAS BEEN RUN

This file uses the linear regression model created from the other file and compares it with a normal PID loop

A similar loop from 'model_run.py' is used for the normal PID loop

The second loop is the model loop, where the desired acceleration is replace with a predicted acceleration from the velocity inputted into the model

'''
model = LinearRegressionModel()
model.load_state_dict(torch.load("./auto", weights_only=True))
model.to(device)

#set model to prediction mode
model.eval()

normal_velocities = []
normal_errors = []
normal_times = []

model_velocities = []
model_errors = []
model_times = []

compare_cars = [15]

x_train = []
y_train = []
x_test = []
y_test = []

#Normal PID loop
for new_car in compare_cars:
    car = make_car(desired_v=new_car, dt=0.1)
    for step in range(STEPS):
        desired_accleration, error = calculate_desired_acceleration(car, K_P, K_I, K_D)
        throttle = acceleration_to_throttle_percentage(desired_accleration)

        x_train.append([car["v"], car["desired_v"]])
        y_train.append([desired_accleration])  

        car["error_prev"] = error
        update(car, throttle)

        normal_velocities.append(car["v"])
        normal_errors.append(error)
        normal_times.append(car["t"])

#Prediction Model Loop
for new_car in compare_cars:
    car = make_car(desired_v=new_car, dt=0.1)
    for step in range(STEPS):
        x = torch.tensor([[car["v"], car["desired_v"]]], dtype=torch.float32, device=device)

        #no need for gradients
        with torch.inference_mode():
            acceleration = model(x).item()

        throttle = acceleration_to_throttle_percentage(acceleration)
        update(car, throttle)

        model_velocities.append(car["v"])
        model_times.append(car["t"])


#Comparison plot between the normal PID loop and predicted model loop
plt.plot(normal_times ,normal_velocities, label="Normal PID Run")
plt.plot(model_times, model_velocities, label="Model Predictions")
plt.xlabel("time")
plt.ylabel("velocity")
plt.legend()
plt.show()