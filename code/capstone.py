#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:20:00 2022
@author: Yimin Han
"""


import numpy as np
import modern_robotics as mr
import matplotlib.pyplot as plt
import logging

LOG_FILENAME = 'test.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

def NextState(config, u, dt, boundary=200):
    """
	currentConfig: Chassis phi, Chassis x, Chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, Gripper State
	control: Arm joint speed(thetadot), Wheel speed(wdot)
	dt: time step
	boundary: Maximum angular speed of the arm joints and the wheels
	"""
    r = 0.0475
    l = 0.47/2
    w = 0.3/2
    # vb_temp is the matrix for 4-wheel 
    vb_temp = np.array([
    [-1/(l+w),1/(l+w),1/(w+l),-1/(l+w)],
    [1,1,1,1],
    [-1,1,-1,1]    
    ])
    phi = config[0]
    x = config[1]
    y = config[2]
    for i in range(3,12): # update the configurations
        config[i] = config[i] + u[i-3] * dt
       
    dtheta = np.array([[u[5]],[u[6]],[u[7]],[u[8]]])*dt #
    Vb = np.dot(vb_temp,dtheta)*r/4
    Vb = Vb.T.flatten()
    Vb6 = np.array([0, 0, Vb[0], Vb[1],Vb[2],0])
    Tbb = mr.MatrixExp6(mr.VecTose3(Vb6))

    Ts = np.array([[np.cos(phi), -np.sin(phi), 0, x],[np.sin(phi), np.cos(phi), 0, y],[0, 0, 0, 0.0963],[0, 0, 0, 1]])
    Tb = np.dot(Ts,Tbb)
    config[0] = np.arccos(Tb[0][0])
    config[1] = Tb[0][3]
    config[2] = Tb[1][3]
    return config

def reshape(matrix, args = 'open'):
    if args == 'open':
        vec = np.array([matrix[0][0],matrix[0][1],matrix[0][2],matrix[1][0],matrix[1][1],matrix[1][2],matrix[2][0],matrix[2][1],matrix[2][2],matrix[0][3],matrix[1][3],matrix[2][3],0])
    elif args == 'closed':
        vec = np.array([matrix[0][0],matrix[0][1],matrix[0][2],matrix[1][0],matrix[1][1],matrix[1][2],matrix[2][0],matrix[2][1],matrix[2][2],matrix[0][3],matrix[1][3],matrix[2][3],1])

    return vec

def TrajectoryGenerator(Tse, TscI, TscF, TceG, TceST, k=1):
    """
	Tse: The initial configuration of the end-effector in the reference trajectory
	TscI: The cube's initial configuration
	TscF: The cube's desired final configuration
	TceG: The end-effector's configuration relative to the cube when it is grasping the cube
	TceST: The end-effector's standoff configuration above the cube, before and after grasping, relative to the cube
	k: The number of trajectory reference configurations per 0.01 seconds
	"""
    # seg1
    T1 = 10 # cost 10 seconds
    N = T1*k/0.01
    TseST = TscI @ TceST
    Traj1 = mr.CartesianTrajectory(Tse, TseST, T1, N, 3)
    temp = Traj1[1]
    Traj_total = np.array([temp[0][0],temp[0][1],temp[0][2],temp[1][0],temp[1][1],temp[1][2],temp[2][0],temp[2][1],temp[2][2],temp[0][3],temp[1][3],temp[2][3],0])
    for matrix in Traj1:
        temp = np.array([matrix[0][0],matrix[0][1],matrix[0][2],matrix[1][0],matrix[1][1],matrix[1][2],matrix[2][0],matrix[2][1],matrix[2][2],matrix[0][3],matrix[1][3],matrix[2][3],0])

        Traj_total = np.row_stack((Traj_total,temp))

    # seg2
    T2 = 10
    N = T2*k/0.01
    TseST = TscI @ TceST 
    TseG = TscI @ TceG
    Traj2 = mr.CartesianTrajectory(TseST, TseG, T2, N, 3)
    for matrix in Traj2:
        temp = reshape(matrix, 'open')

        Traj_total = np.row_stack((Traj_total,temp))

    #seg3: closed the gripper 1
    N=63
    temp[-1]=1
    for i in range(N):
        Traj_total = np.row_stack((Traj_total,temp))
    
    #seg4
    T4 = 10
    N = T4*k/0.01
    #TseST = TscI @ TceST 
    TseG = TscI @ TceG
    Traj4 = mr.CartesianTrajectory(TseG, TseST, T4, N, 3)
    for matrix in Traj4:
        temp = reshape(matrix,'closed')

        Traj_total = np.row_stack((Traj_total,temp))

    #seg5
    T5 = 10
    N = T5*k/0.01
    TseFST = TscF @ TceST 
    Traj5 = mr.CartesianTrajectory(TseST, TseFST, T5, N, 3)
    for matrix in Traj5:
        temp = reshape(matrix,'closed')
        Traj_total = np.row_stack((Traj_total,temp))

    #seg6
    T6 = 10
    N = T6*k/0.01
    #TseFST = TscF @ TceST 
    TseFG = TscF @ TceG
    Traj6 = mr.CartesianTrajectory(TseFST, TseFG, T6, N, 3)
    for matrix in Traj6:
        temp = reshape(matrix,'closed')
        Traj_total = np.row_stack((Traj_total,temp))

    #seg7 opening gripper 0
    N=63
    temp[-1]=0
    for i in range(N):
        Traj_total = np.row_stack((Traj_total,temp))
    
    #seg8 move up
    T8 = 10
    N = T8*k/0.01
    Traj8 = mr.CartesianTrajectory(TseFG, TseFST, T8, N, 3)
    for matrix in Traj8:
        temp = reshape(matrix,'open')
        Traj_total = np.row_stack((Traj_total,temp))

    return Traj_total

def FeedbackControl(X, Xd, Xdnext, integ, Kp, Ki, dt):
    '''
To compute V(t) = [Ad_X^(-1)*Xd]*Vd(t) + Kp*Xerr(t) + Ki*Integral[Xerr(t)]dt
    '''
    
    Ad = mr.Adjoint(np.dot(np.linalg.inv(X),Xd))
    Vd = mr.se3ToVec(mr.MatrixLog6(np.dot(np.linalg.inv(Xd),Xdnext))/dt)
    part1 = Ad @ Vd
   
    Xerr = mr.se3ToVec(mr.MatrixLog6(np.dot(np.linalg.inv(X),Xd)))
    part2 = Kp @ Xerr
    integ = integ + Xerr*dt # compute integration using recursion
    part3 = Ki @ integ
    Vt = part1 + part2 +part3
    
    return Vt, Xerr, integ

# compute je and u with pinv(je)&Vt
def computeControl(thetalist, Vt):
    T0e = mr.FKinBody(M0e,Blist,thetalist) #T0e(theta)
    Jbase = mr.Adjoint(np.dot(np.linalg.inv(T0e),np.linalg.inv(Tb0))) @ F6
    Jarm = mr.JacobianBody(Blist,thetalist)
    Je = np.append(Jbase, Jarm, axis = 1)
    u = np.dot(np.linalg.pinv(Je), Vt)
    return Je, u

def Trajtose3(traj):
    X = np.array([
        [traj[0],traj[1],traj[2],traj[9]],
        [traj[3],traj[4],traj[5],traj[10]],
        [traj[6],traj[7],traj[8],traj[11]],
        [0,0,0,1]
    ])
    return X

def ConfigtoTse(config):
    Blist = np.array([[0, 0, 1, 0, 0.033, 0],[0, -1,0, -0.5076, 0 ,0],[0, -1, 0, -0.3526, 0 ,0],[0, -1, 0, -0.2176, 0 ,0],[0, 0, 1, 0, 0, 0]]).T
    thetalist = config[3:8]
    phi = config[0]
    x=config[1]
    y=config[2]
    Tsb = np.array([[np.cos(phi), -np.sin(phi), 0, x],[np.sin(phi), np.cos(phi), 0, y],[0, 0, 1, 0.0963],[0, 0, 0, 1]])
    Tb0 = np.array([[1, 0, 0, 0.1662],[0, 1, 0, 0],[0, 0, 1, 0.0026],[0, 0, 0, 1]])
    T0e = mr.FKinBody(M0e,Blist,thetalist)
    X = np.dot(np.dot(Tsb,Tb0),T0e)
    return X

def writingData(theta_total, exp): 
    # Open a file for output
    # Please change the address below!!
    f = open("/mnt/d/NorthwesternUniversity/Courses/2022Fall/Robot Manipulation/HW/final_proj/final_config_exp="+str(exp)+".csv", "w") 

    for i in range(np.shape(theta_total)[0]):
        #change it according to the size of the theta_total
        output = " %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f\n" % (theta_total[i,0], theta_total[i,1], theta_total[i,2],theta_total[i,3],theta_total[i,4],theta_total[i,5],theta_total[i,6],theta_total[i,7],theta_total[i,8],theta_total[i,9],theta_total[i,10],theta_total[i,11],theta_total[i,12])
        f.write(output)
    
    # close file
    f.close()

def writingData2(theta_total): 
    # Open a file for output
    # Please change the address below!!
    f = open("/mnt/d/NorthwesternUniversity/Courses/2022Fall/Robot Manipulation/HW/final_proj/Traj.csv", "w") 

    for i in range(np.shape(theta_total)[0]):
        #change it according to the size of the theta_total
        output = " %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f\n" % (theta_total[i,0], theta_total[i,1], theta_total[i,2],theta_total[i,3],theta_total[i,4],theta_total[i,5],theta_total[i,6],theta_total[i,7],theta_total[i,8],theta_total[i,9],theta_total[i,10],theta_total[i,11],theta_total[i,12])
        f.write(output)
    
    # close file
    f.close()

# initial parameters

integ = 0
Blist = np.array([[0, 0, 1, 0, 0.033, 0],[0, -1,0, -0.5076, 0 ,0],[0, -1, 0, -0.3526, 0 ,0],[0, -1, 0, -0.2176, 0 ,0],[0, 0, 1, 0, 0, 0]]).T
r = 0.0475
l = 0.47/2
w = 0.3/2
vb_temp = np.array([
[-1/(l+w),1/(l+w),1/(w+l),-1/(l+w)],
[1,1,1,1],
[-1,1,-1,1]    
])
F = (vb_temp)*r/4
F6=np.array([[0]*4,[0]*4])
F6=np.append(F6,F,axis=0)
F6=np.append(F6,np.array([[0]*4]),axis=0)
Tsb = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.0963],[0,0,0,1]])
Tb0 = np.array([[1,0,0,0.1662],[0,1,0,0],[0,0,1,0.0026],[0,0,0,1]])
M0e = np.array([[1,0,0,0.033],[0,1,0,0],[0,0,1,0.6546],[0,0,0,1]])
height = 0.6546 - 0.025
k = 1
dt = 0.01
Kp = 20* np.identity(6)
Ki = 8* np.identity(6)


# initial config, thetalist and Tse
config = np.array([-np.pi/5,0.1,0.2,0,0,0.2,-1.6,0,0,0,0,0])
Tse = np.array([[0, 0, 1, 0],[0, 1, 0, 0],[-1, 0, 0, 0.5],[0, 0, 0, 1]])
# best and overshoot:
TscI = np.array([[1,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
TscF = np.array([[0,1,0,0],[-1,0,0,-1],[0,0,1,0],[0,0,0,1]])
# for new task:
# TscI = np.array([[1,0,0,1],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
# TscF = np.array([[0,1,0,1],[-1,0,0,-1],[0,0,1,0],[0,0,0,1]])

TceG = np.array([[-0.5, 0, 0.8660254, 0.015],[0, 1, 0, 0],[-0.8660254,0, -0.5,0.02],[0, 0, 0, 1]])
TceST = np.array([[0, 0, 1, -0.3],[0, 1, 0, 0],[-1, 0, 0, 0.5],[0, 0, 0, 1]])
thetalist = config[3:8]

Traj = TrajectoryGenerator(Tse, TscI, TscF, TceG, TceST, k=1)
writingData2(Traj)

X=Tse
config_total = np.append(config,0)
error = []
error1 = []
error2 = []
error3 = []
error4 = []
error5 = []
error6 = []
# loop to compute the actutal trajectory.
for i in range(np.shape(Traj)[0]-1):
    thetalist = np.array([config[3],config[4],config[5],config[6],config[7]])
    Xd = Trajtose3(Traj[i])
    Xdnext = Trajtose3(Traj[i+1])
    Vt, Xerr, integ = FeedbackControl(X, Xd, Xdnext, integ, Kp, Ki, dt)
    _, u = computeControl(thetalist, Vt)
    u = np.append(u[-5:],u[:4])
    new_config= NextState(config, u, dt, boundary=20)
    config = new_config
    X = ConfigtoTse(config)
    new_config = np.append(new_config, Traj[i][-1])
    config_total = np.row_stack((config_total, new_config))
    error1.append(Xerr[0])
    error2.append(Xerr[1])
    error3.append(Xerr[2])
    error4.append(Xerr[3])
    error5.append(Xerr[4])
    error6.append(Xerr[5])
    error.append([Xerr[0],Xerr[1],Xerr[2],Xerr[3],Xerr[4],Xerr[5]])

exp='test'
logging.debug('generating Xerr data file')
np.savetxt("/mnt/d/NorthwesternUniversity/Courses/2022Fall/Robot Manipulation/HW/final_proj/Error"+str(exp)+".csv", error, delimiter=",")

logging.debug('generating config csv file')
writingData(config_total,exp)

logging.debug('plotting error data')
print(len(error1))
t=np.arange(0,61.26,0.01)
plt.plot(t,error1,label='error1')
plt.plot(t,error2,label='error2')
plt.plot(t,error3,label='error3')
plt.plot(t,error4,label='error4')
plt.plot(t,error5,label='error5')
plt.plot(t,error6,label='error6')
plt.legend()
plt.title("Error Plot")
plt.show()

logging.debug("Done")
