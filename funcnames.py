import numpy as np


def SCB(theta):
    return (4-2.1*theta[0]**2+(1/3)*theta[0]**4)*theta[0]**2 + \
        theta[0]*theta[1]+(-4+4*theta[1]**2)*theta[1]**2


def Passino(theta):
    return 5*np.exp(-0.1 * ((theta[0]-15)**2+(theta[1]-20)**2)) - 2*np.exp(-0.08*((theta[0]-20)**2+(theta[1]-15)**2))+3*np.exp(-0.08*((theta[0]-25)**2+(theta[1]-10)**2))+2*np.exp(-0.1 * ((theta[0]-10)**2+(theta[1]-10)**2))-2*np.exp(-0.5 * ((theta[0]-5) ** 2+(theta[1]-10)**2))-4*np.exp(-0.1 * ((theta[0]-15)**2+(theta[1]-5) ** 2))-2*np.exp(-0.5 * ((theta[0]-8) ** 2+(theta[1]-25)**2))-2*np.exp(-0.5 * ((theta[0]-21)**2+(theta[1]-25)**2))+2*np.exp(-0.5 * ((theta[0]-25)**2+(theta[1]-16)**2))+2*np.exp(-0.5 * ((theta[0]-5) ** 2+(theta[1]-14)**2))


def Peaks(theta):
    return 3*(1-theta[0])**2*np.exp(-(theta[0]**2)-(theta[1]+1)**2)-10*(theta[0]/5-theta[0] ** 3-theta[1]**5)*np.exp(-theta[0]**2 - theta[1]**2)-1/3*np.exp(-(theta[0]+1)**2-theta[1]**2)


def DeJong_F1(theta):
    return np.sum(theta ** 2, axis=0)


def DeJong_F2(theta):
    return 100*(theta[0]**2-theta[1])**2+(1-theta[0])**2


def Rastrigin(theta):
    return np.sum(theta**2-(10*np.cos(2*np.pi*theta))+10, axis=0)


def Ackley(theta):
    return -20*np.exp(-0.2*np.sqrt((1/30)*sum(theta**2, 1)))-np.exp((1/30)*sum(np.cos(2*np.pi*theta)))+20+np.exp(1)


def Schaffer_F6(theta):
    return 0.5-(np.sin(np.sqrt(sum(theta ** 2))) ** 2-0.5)/((1+0.001*sum(theta ** 2)) ** 2)


def Schewefel(theta):
    return np.sum(theta*np.sin(np.sqrt(abs(theta))), axis=0)
