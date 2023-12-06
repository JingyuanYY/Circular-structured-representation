import numpy as np
import matplotlib.pyplot as plt
import math
import torch

def Convert_polar_to_rectangular(theta, r):
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return x, y

def Convert_rectangular_to_polar(x, y):
    theta = torch.atan2(y, x)
    r = (x ** 2+y ** 2) ** 0.5
    # r = math.degrees(r) # radian to angle
    return theta, r

def Polar_coordinates(dist):
    # Emotion Circle order: 0-sad 1-fear 2-excitement 3-awe 4-contentment 5-amusement 6-anger 7-disgust
    # For calculation !!!
    # predict = np.array([0.6, 0, 0.3, 0, 0, 0.1, 0, 0])
    # dist = dist.cpu().detach().numpy() ######

    # Unit emotion vector
    theta_unit = np.array([0.125 * np.pi, 0.375 * np.pi, 0.625 * np.pi, 0.875 * np.pi,
                      1.125 * np.pi, 1.375 * np.pi, 1.625 * np.pi, 1.875 * np.pi])
    r_unit = np.ones(8)
    theta_unit = torch.Tensor(theta_unit)
    r_unit = torch.Tensor(r_unit)

    # Calculate individual emotion vector
    # weighted the unit emotion vector

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    theta_unit = theta_unit.to(device)
    r_unit = r_unit.to(device)

    theta = theta_unit
    r = dist * r_unit

    # turn polar to rectangular
    x, y = Convert_polar_to_rectangular(theta, r)
    # do vector sum in rectangular coordinate system
    x_composed = x.sum(1)
    y_composed = y.sum(1)
    # turn rectangular back to polar
    theta_composed, r_composed = Convert_rectangular_to_polar(x_composed, y_composed)
    theta_composed = theta_composed.cuda()
    r_composed = r_composed.cuda()
    return theta_composed, r_composed

#----------------VISUALIZE POLAR COORDINATE SYSTEM--------------------#
# # Generate emotion unit vector
# # Label and Predict order: 0-Amusement 1-Awe 2-Contentment 3-Excitement 4-Anger 5-Disgust 6-Fear 7-Sad
# weight = np.array([0.6, 0, 0.3, 0, 0, 0.1, 0, 0])
# N = 10
#
# # Emotion Circle order: 0-sad 1-fear 2-excitement 3-awe 4-contentment 5-amusement 6-anger 7-disgust
# # For visualization !!!
# emotions = ["Sad", "Fear", "Excitement", "Awe", "Contentment", "Amusement", "Anger", "Disgust"]
# theta_sad = np.linspace(0.125 * np.pi, 0.125 * np.pi, N, endpoint=False)
# r_sad = weight[7] * np.linspace(0, 1, N)
# theta_fea = np.linspace(0.375 * np.pi, 0.375 * np.pi, N, endpoint=False)
# r_fea = weight[6] * np.linspace(0, 1, N)
# theta_exc = np.linspace(0.625 * np.pi, 0.625 * np.pi, N, endpoint=False)
# r_exc = weight[3] * np.linspace(0, 1, N)
# theta_awe = np.linspace(0.875 * np.pi, 0.875 * np.pi, N, endpoint=False)
# r_awe = weight[1] * np.linspace(0, 1, N)
# theta_con = np.linspace(1.125 * np.pi, 1.125 * np.pi, N, endpoint=False)
# r_con = weight[2] * np.linspace(0, 1, N)
# theta_amu = np.linspace(1.375 * np.pi, 1.375 * np.pi, N, endpoint=False)
# r_amu = weight[0] * np.linspace(0, 1, N)
# theta_ang = np.linspace(1.625 * np.pi, 1.625 * np.pi, N, endpoint=False)
# r_ang = weight[4] * np.linspace(0, 1, N)
# theta_dis = np.linspace(1.875 * np.pi, 1.875 * np.pi, N, endpoint=False)
# r_dis = weight[5] * np.linspace(0, 1, N)
#
# ax = plt.subplot(111, projection='polar')
# ax.set_rgrids(np.arange(0, 1.0, 0.2))
# ax.set_rlim(0, 1)
# lines, labels = plt.thetagrids(range(22, 382, int(360/len(emotions))), (emotions))
# # ax.set_rlabel_position('90')
# # ax.bar(theta, r1, width=width, bottom=0.0, color ='red', alpha=0.5)
# # ax.plot(theta, r1, width=width, bottom=0.0, color ='Set2', alpha=0.5)
# plt.plot(theta_sad, r_sad, 'limegreen', lw=2)
# plt.plot(theta_fea, r_fea, 'gold', lw=2)
# plt.plot(theta_exc, r_exc, 'darkorange', lw=2)
# plt.plot(theta_awe, r_awe, 'red', lw=2)
# plt.plot(theta_con, r_con, 'purple', lw=2)
# plt.plot(theta_amu, r_amu, 'mediumblue', lw=2)
# plt.plot(theta_ang, r_ang, 'cornflowerblue', lw=2)
# plt.plot(theta_dis, r_dis, 'darkgreen', lw=2)
#
# plt.show()