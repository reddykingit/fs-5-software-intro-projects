# Software Onboarding Projects - Fall 2026 (fs-5 design cycle)

This repository holds the onboarding projects for the three software subsystems.

- [Simulation Team Onboarding Projects](sims)
- [Data Team Onboarding Projects](data)
- [Autonomous Team Onboarding Projects](auto)


# Autonomous Section

Install pytorch, numpy, and matplotlib before starting

## How to Run

### Step 1
First, train and save the model by running

```
python model_run.py
```
Two graphs should be shown

<img width="480" height="364" alt="image" src="https://github.com/user-attachments/assets/5e6bb203-12ea-4948-be2a-ba874d1f7f4c" />

First Graph Example

<img width="600" height="602" alt="image" src="https://github.com/user-attachments/assets/ce4206b6-0f43-4d93-8d2a-599a0fa3a32a" />

Second Graph Example

Closing off of the first shows the second

### Step 2
Then, to visualize the comparison between the model and PID controller run:

```
python run_template.py
```
An output graph should be shown

<img width="479" height="364" alt="image" src="https://github.com/user-attachments/assets/45f25196-a36e-45e6-bc64-3be2f544a38e" />


# Step 3
I'd recommend playing around with the data, here's some examples of what you can change in model_run.py

* K_P
* K_I (<0.25)
* K_D (<1)
* train_cars
* epochs
