import numpy as np
from scipy.special import sici

def power(x,n):
    return np.power(x,n)

def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def sqrt(x):
    return np.sqrt(x)

def abs(x):
    return np.abs(x)

def log(x):
    return np.log(x)

def sign(x):
    return np.sign(x)

def Si(x):
    return sici(x)[0]

def Ci(x):
    return sici(x)[1]

def ArcCoth(x):
    return 0.5*log((x + 1)/(x - 1))

def Sinc(x):
    return sin(x)/x

def Tpsi(x):
    return np.where(x<1e-3, 3./2. - x*x/12. , 3./2.*sin(x/sqrt(3.))/(x/sqrt(3.)))

def dTpsi(x):
    return np.where(x<1e-3, -x/6. + x*x*x/180. , 3./2./x/x*(x*cos(x/sqrt(3)) - sqrt(3)*sin(x/sqrt(3))))

def Tphi(x):
    return np.where(x<1e-3, x*x/6 , 3./2.*(sin(x/sqrt(3.))/(x/sqrt(3.)) - cos(x/sqrt(3))))

def TB(x):
    return np.where(x<1e-3,x/2. - x*x*x/20. , 3./2./x/x*(6.*x*cos(x/sqrt(3)) + sqrt(3)*(x*x - 6)*sin(x/sqrt(3))))

def f_psi(u, v, x):
	return -0.015625*((u*x*cos((u*x)/sqrt(3))*(6*v*x*(9*pow(u,8)*pow(x,4) + pow(u,6)*pow(x,2)*(-264 + (-21 + 16*pow(v,2))*pow(x,2)) - 3*pow(u,4)*(-864 + 16*(-7 + 8*pow(v,2))*pow(x,2) + (-5 + 19*pow(v,2) + 4*pow(v,4))*pow(x,4)) - pow(u,2)*(2592 - 36*(-3 - 2*pow(v,2) + 16*pow(v,4))*pow(x,2) + (3 - 6*pow(v,2) - 33*pow(v,4) + 4*pow(v,6))*pow(x,4)) - 3*(-1 + pow(v,2))*(12*pow(x,2) + 3*pow(v,6)*pow(x,4) - 4*pow(v,4)*pow(x,2)*(6 + pow(x,2)) + pow(v,2)*(864 + 32*pow(x,2) + pow(x,4))))*cos((v*x)/sqrt(3)) + sqrt(3)*(9*pow(u,8)*pow(x,4)*(-6 + pow(v,2)*pow(x,2)) - 3*pow(u,2)*(-5184 + 72*(-3 + 6*pow(v,2) + 16*pow(v,4))*pow(x,2) + (-6 + 54*pow(v,2) + 126*pow(v,4) - 262*pow(v,6))*pow(x,4) + pow(v,2)*pow(-1 + pow(v,2),2)*(1 + 7*pow(v,2))*pow(x,6)) + pow(u,6)*(1584*pow(x,2) - 18*(-7 + 17*pow(v,2))*pow(x,4) - pow(v,2)*(21 + 23*pow(v,2))*pow(x,6)) + pow(u,4)*(-15552 + 2016*(-1 + 2*pow(v,2))*pow(x,2) - 6*(15 - 88*pow(v,2) + 45*pow(v,4))*pow(x,4) + pow(v,2)*(15 + 14*pow(v,2) + 35*pow(v,4))*pow(x,6)) + 36*(-1 + pow(v,2))*(6*pow(x,2) + pow(v,6)*pow(x,4) + 2*pow(v,4)*pow(x,2)*(-30 + pow(x,2)) + pow(v,2)*(432 + 16*pow(x,2) - pow(x,4))))*sin((v*x)/sqrt(3))) + sin((u*x)/sqrt(3))*(sqrt(3)*v*x*(3*pow(u,8)*pow(x,4)*(-36 + 7*pow(v,2)*pow(x,2)) + pow(u,6)*(3888*pow(x,2) + (180 - 834*pow(v,2))*pow(x,4) - pow(v,2)*(39 + 35*pow(v,2))*pow(x,6)) + pow(u,4)*(-5184 + 288*(-13 + 4*pow(v,2))*pow(x,2) + 6*(-18 + 47*pow(v,2) + 97*pow(v,4))*pow(x,4) + pow(v,2)*(15 - 14*pow(v,2) + 23*pow(v,4))*pow(x,6)) + 18*(-1 + pow(v,2))*(12*pow(x,2) + 3*pow(v,6)*pow(x,4) - 4*pow(v,4)*pow(x,2)*(6 + pow(x,2)) + pow(v,2)*(864 + 32*pow(x,2) + pow(x,4))) - 3*pow(u,2)*(3*pow(v,8)*pow(x,6) - pow(v,6)*pow(x,4)*(38 + 7*pow(x,2)) - 12*(144 + 2*pow(x,2) + pow(x,4)) + pow(v,4)*pow(x,2)*(1536 + 120*pow(x,2) + 5*pow(x,4)) - pow(v,2)*(-3456 + 720*pow(x,2) + 38*pow(x,4) + pow(x,6))))*cos((v*x)/sqrt(3)) - 18*(6*pow(u,8)*pow(x,4)*(-3 + pow(v,2)*pow(x,2)) + pow(u,6)*pow(x,2)*(648 - 6*(-5 + 26*pow(v,2))*pow(x,2) + pow(v,2)*(-11 + 24*pow(v,2))*pow(x,4)) - 2*pow(u,4)*(432 - 24*(-13 + 6*pow(v,2))*pow(x,2) + (9 - 47*pow(v,2) - 22*pow(v,4))*pow(x,4) + pow(v,2)*(-3 + pow(v,2) + 13*pow(v,4))*pow(x,6)) + 6*(-1 + pow(v,2))*(6*pow(x,2) + pow(v,6)*pow(x,4) + 2*pow(v,4)*pow(x,2)*(-30 + pow(x,2)) + pow(v,2)*(432 + 16*pow(x,2) - pow(x,4))) + pow(u,2)*(-4*pow(v,8)*pow(x,6) + pow(v,6)*pow(x,4)*(124 + 5*pow(x,2)) - 2*pow(v,4)*pow(x,2)*(288 + 65*pow(x,2)) + 6*(144 + 2*pow(x,2) + pow(x,4)) - pow(v,2)*(1728 - 264*pow(x,2) + 4*pow(x,4) + pow(x,6))))*sin((v*x)/sqrt(3)))))/(pow(u,3)*pow(v,3)*pow(x,6))

def f_B(u, v, x):
    return (9*(2*u*x*cos((u*x)/sqrt(3))*(6*v*x*(pow(u,6)*(15*pow(x,2) - pow(v,2)*pow(x,4)) + pow(u,4)*(-192 + 15*(-1 + 2*pow(v,2))*pow(x,2) + pow(v,2)*pow(x,4)) + pow(u,2)*(pow(v,6)*pow(x,4) - 3*(-64 + pow(x,2)) - 4*pow(v,2)*(-12 + pow(x,2)) - pow(v,4)*pow(x,2)*(33 + pow(x,2))) - 3*(-1 + pow(v,2))*(pow(x,2) + 4*pow(v,4)*pow(x,2) - pow(v,2)*(48 + pow(x,2))))*cos((v*x)/sqrt(3)) + sqrt(3)*(6*pow(u,6)*pow(x,2)*(-15 + 4*pow(v,2)*pow(x,2)) + pow(u,4)*(1152 + (90 - 384*pow(v,2))*pow(x,2) + pow(v,2)*(-27 + 5*pow(v,2))*pow(x,4)) - 2*pow(u,2)*(2*pow(v,6)*pow(x,4) - 9*(-64 + pow(x,2)) - 72*pow(v,2)*(-2 + pow(x,2)) - pow(v,4)*pow(x,2)*(141 + 2*pow(x,2))) - 3*(-1 + pow(v,2))*(-6*pow(x,2) + 3*pow(v,6)*pow(x,4) - 4*pow(v,4)*pow(x,2)*(12 + pow(x,2)) + pow(v,2)*(288 + 18*pow(x,2) + pow(x,4))))*sin((v*x)/sqrt(3))) + sin((u*x)/sqrt(3))*(2*sqrt(3)*v*x*(9*pow(u,8)*pow(x,4) + pow(u,6)*pow(x,2)*(-264 + (-15 + 7*pow(v,2))*pow(x,2)) + pow(u,4)*(288 - 6*(-43 + 18*pow(v,2))*pow(x,2) + (3 - 30*pow(v,2) - 11*pow(v,4))*pow(x,4)) + 18*(-1 + pow(v,2))*(pow(x,2) + 4*pow(v,4)*pow(x,2) - pow(v,2)*(48 + pow(x,2))) - 3*pow(u,2)*(96 - 8*pow(x,2) - pow(x,4) + 7*pow(v,6)*pow(x,4) - pow(v,4)*pow(x,2)*(116 + 9*pow(x,2)) + 3*pow(v,2)*(-64 + 20*pow(x,2) + pow(x,4))))*cos((v*x)/sqrt(3)) + (9*pow(u,8)*pow(x,4)*(-6 + pow(v,2)*pow(x,2)) + pow(u,6)*(1584*pow(x,2) + (90 - 432*pow(v,2))*pow(x,4) - 5*pow(v,2)*(3 + 7*pow(v,2))*pow(x,6)) + pow(u,4)*(-1728 + 36*(-43 + 36*pow(v,2))*pow(x,2) + 6*(-3 + 69*pow(v,2) + 20*pow(v,4))*pow(x,4) + pow(v,2)*(3 - 6*pow(v,2) + 35*pow(v,4))*pow(x,6)) + 18*(-1 + pow(v,2))*(-6*pow(x,2) + 3*pow(v,6)*pow(x,4) - 4*pow(v,4)*pow(x,2)*(12 + pow(x,2)) + pow(v,2)*(288 + 18*pow(x,2) + pow(x,4))) - 3*pow(u,2)*(3*pow(v,8)*pow(x,6) - pow(v,6)*pow(x,4)*(104 + 7*pow(x,2)) + 6*(-96 + 8*pow(x,2) + pow(x,4)) + pow(v,4)*pow(x,2)*(672 + 78*pow(x,2) + 5*pow(x,4)) - pow(v,2)*(-1152 + 312*pow(x,2) + 44*pow(x,4) + pow(x,6))))*sin((v*x)/sqrt(3)))))/(32.*pow(u,3)*pow(v,3)*pow(x,5))


def f_phi(u, v, x):
    return (-3*(2*v*x*cos((v*x)/sqrt(3))*(3*u*(-1 + pow(u,2) - pow(v,2))*x*(-36*(-1 + pow(v,2)) + 3*pow(u,4)*pow(x,2) + pow(u,2)*(-60 + (-3 + 11*pow(v,2))*pow(x,2)))*cos((u*x)/sqrt(3)) + sqrt(3)*(-108*(-1 + pow(v,4)) + 3*pow(u,6)*pow(x,2)*(-12 + pow(v,2)*pow(x,2)) + pow(u,4)*(36 + (54 - 36*pow(v,2))*pow(x,2) + pow(v,2)*(-3 + 5*pow(v,2))*pow(x,4)) + 6*pow(u,2)*(16*pow(v,4)*pow(x,2) - pow(v,2)*(-12 + pow(x,2)) - 3*(8 + pow(x,2))))*sin((u*x)/sqrt(3))) - (-(sqrt(3)*u*x*(3*pow(u,6)*pow(x,2)*(-6 + pow(v,2)*pow(x,2)) - 2*pow(u,4)*(-180 + 18*(-1 + 5*pow(v,2))*pow(x,2) + pow(v,2)*(3 + 8*pow(v,2))*pow(x,4)) + 36*(-1 + pow(v,2))*(-6 + pow(v,2)*(-6 + pow(x,2))) - 3*pow(u,2)*(-50*pow(v,4)*pow(x,2) + pow(v,6)*pow(x,4) + 6*(32 + pow(x,2)) - pow(v,2)*(-48 + 72*pow(x,2) + pow(x,4))))*cos((u*x)/sqrt(3))) + 18*(3*pow(u,6)*pow(x,2)*(-4 + pow(v,2)*pow(x,2)) - 2*pow(u,4)*(-6 + (-9 + 13*pow(v,2))*pow(x,2) + pow(v,2)*(2 + pow(v,2))*pow(x,4)) + 6*(-1 + pow(v,2))*(-6 + pow(v,2)*(-6 + pow(x,2))) + pow(u,2)*(38*pow(v,4)*pow(x,2) - pow(v,6)*pow(x,4) - 6*(8 + pow(x,2)) + pow(v,2)*(24 + 18*pow(x,2) + pow(x,4))))*sin((u*x)/sqrt(3)))*sin((v*x)/sqrt(3))))/(16.*pow(u,3)*pow(v,3)*pow(x,4))

def f_rho(u, v, x):
    return (3*(6*v*x*cos((v*x)/sqrt(3))*(-2*u*x*(3*pow(u,4)*pow(x,2) + 3*(-1 + pow(v,2))*(12 + (-1 + pow(v,2))*pow(x,2)) - 2*pow(u,2)*(-18 + (3 + 8*pow(v,2))*pow(x,2)))*cos((u*x)/sqrt(3)) - sqrt(3)*(pow(u,6)*pow(x,4) - 6*(-1 + pow(v,2))*(12 + (-1 + pow(v,2))*pow(x,2)) + pow(u,2)*(-72 + 4*(2 + 5*pow(v,2))*pow(x,2) + pow(-1 + pow(v,2),2)*pow(x,4)) - 2*pow(u,4)*(pow(x,2) + (1 + pow(v,2))*pow(x,4)))*sin((u*x)/sqrt(3))) + (-2*sqrt(3)*u*x*(3*pow(u,4)*pow(x,2)*(-6 + pow(v,2)*pow(x,2)) + 3*(-1 + pow(v,2))*(-72 + 6*(1 + pow(v,2))*pow(x,2) + pow(v,2)*(-1 + pow(v,2))*pow(x,4)) - 2*pow(u,2)*(108 - 6*(3 + 11*pow(v,2))*pow(x,2) + pow(v,2)*(3 + 7*pow(v,2))*pow(x,4)))*cos((u*x)/sqrt(3)) - 3*(pow(u,6)*pow(x,4)*(-6 + pow(v,2)*pow(x,2)) - 6*(-1 + pow(v,2))*(-72 + 6*(1 + pow(v,2))*pow(x,2) + pow(v,2)*(-1 + pow(v,2))*pow(x,4)) - 2*pow(u,4)*pow(x,2)*(-6 + (-6 + pow(v,2))*pow(x,2) + (pow(v,2) + pow(v,4))*pow(x,4)) + pow(u,2)*(432 - 48*(1 + 4*pow(v,2))*pow(x,2) + (-6 + 32*pow(v,2) + 46*pow(v,4))*pow(x,4) + pow(v,2)*pow(-1 + pow(v,2),2)*pow(x,6)))*sin((u*x)/sqrt(3)))*sin((v*x)/sqrt(3))))/(16.*pow(u,3)*pow(v,3)*pow(x,4))
    

def I_psi(u, v, x):
    return (3*sqrt(3)*(2*power(x,2)*(sqrt(3)*u*v*(power(-1 + power(u,2),2) - 2*(1 + power(u,2))*power(v,2) + power(v,4))*(-24*(-1 + u - v)*(1 + u - v)*(-1 + u + v)*(1 + u + v)*(5*power(u,4) + 3*power(v,2) - 3*power(v,4) - power(u,2)*(5 + 2*power(v,2))) + (u - v)*(u + v)*(9*power(u,8) + 3*power(-1 + power(v,2),3)*(-1 + 3*power(v,2)) - 2*power(u,6)*(15 + 7*power(v,2)) + 2*power(u,4)*(18 - 41*power(v,2) + 5*power(v,4)) - 2*power(u,2)*(9 - 57*power(v,2) + 41*power(v,4) + 7*power(v,6)))*power(x,2))*cos((u*x)/sqrt(3))*cos((v*x)/sqrt(3)) + x*(2*u*v*(-9*power(u,14) + power(u,12)*(42 + 89*power(v,2)) - power(u,10)*(66 + 328*power(v,2) + 301*power(v,4)) - 3*power(-1 + power(v,2),4)*(-3 - 2*power(v,2) + 8*power(v,4) + 3*power(v,6)) + power(u,8)*(33 + 290*power(v,2) + 492*power(v,4) + 485*power(v,6)) + power(u,6)*(3 - 26*power(v,2) - 1242*power(v,4) + 60*power(v,6) - 395*power(v,8)) + power(u,4)*(12 + 19*power(v,2) - 62*power(v,4) + 1238*power(v,6) - 482*power(v,8) + 139*power(v,10)) + power(u,2)*power(-1 + power(v,2),2)*(-24 + power(v,2)*(1 + power(v,2))*(-62 + 205*power(v,2) + power(v,4)))) + 3*power(power(-1 + power(u,2),2) - 2*(1 + power(u,2))*power(v,2) + power(v,4),2)*(3 + power(u,2) - 2*power(u,4) - 5*power(u,6) + 3*power(u,8) - (1 + 8*power(u,2) + 21*power(u,4))*power(v,2) + (-10 + 5*power(u,2) - 6*power(u,4))*power(v,4) + 5*power(v,6) + 3*power(v,8))*ArcCoth((2*u*v)/(-1 + power(u,2) + power(v,2))))*sin(x/sqrt(3))) - u*x*(-144*power(power(-1 + power(u,2),2) - 2*(1 + power(u,2))*power(v,2) + power(v,4),2)*(5*power(u,4) + 3*power(v,2) - 3*power(v,4) - power(u,2)*(5 + 2*power(v,2))) + 6*(3*power(u,2)*power(-1 + power(u,2),5)*(-1 + 3*power(u,2)) + power(-1 + power(u,2),3)*(-6 + 33*power(u,2) - 118*power(u,4) + 15*power(u,6))*power(v,2) - 4*(9 - 32*power(u,2) + 107*power(u,4) + 90*power(u,6) - 230*power(u,8) + 56*power(u,10))*power(v,4) + 2*(39 + 59*power(u,2) + 367*power(u,4) - 401*power(u,6) + 280*power(u,8))*power(v,6) - (72 + 481*power(u,2) + 252*power(u,4) + 615*power(u,6))*power(v,8) + (18 + 333*power(u,2) + 311*power(u,4))*power(v,10) + 2*(6 - 25*power(u,2))*power(v,12) - 6*power(v,14))*power(x,2) - power(u,2)*power(v,2)*power(power(-1 + power(u,2),2) - 2*(1 + power(u,2))*power(v,2) + power(v,4),2)*(3 + 9*power(u,4) + 12*power(v,2) - 3*power(v,4) - 2*power(u,2)*(6 + 19*power(v,2)))*power(x,4))*cos((u*x)/sqrt(3))*sin((v*x)/sqrt(3)) + sin((u*x)/sqrt(3))*(v*x*(144*power(power(-1 + power(u,2),2) - 2*(1 + power(u,2))*power(v,2) + power(v,4),2)*(power(u,4) + 3*power(v,2) - 3*power(v,4) + power(u,2)*(-1 + 2*power(v,2))) - 6*(18*power(u,14) - 3*power(v,2)*power(-1 + power(v,2),5)*(-1 + 3*power(v,2)) - 6*power(u,12)*(14 + 5*power(v,2)) + power(u,10)*(162 - 109*power(v,2) - 91*power(v,4)) + power(u,2)*power(-1 + power(v,2),3)*(-6 - 37*power(v,2) + 114*power(v,4) + 5*power(v,6)) + power(u,8)*(-168 + 321*power(v,2) + 76*power(v,4) + 295*power(v,6)) - 2*power(u,6)*(-51 + 91*power(v,2) + 379*power(v,4) - 353*power(v,6) + 150*power(v,8)) + 4*power(u,4)*(-9 - 4*power(v,2) + 95*power(v,4) + 74*power(v,6) - 184*power(v,8) + 28*power(v,10)))*power(x,2) + power(u,2)*power(v,2)*power(power(-1 + power(u,2),2) - 2*(1 + power(u,2))*power(v,2) + power(v,4),2)*(-3 + 3*power(u,4) + 12*power(v,2) - 9*power(v,4) + 2*power(u,2)*(-6 + 19*power(v,2)))*power(x,4))*cos((v*x)/sqrt(3)) - 2*sqrt(3)*(-1 + u - v)*(1 + u - v)*(-1 + u + v)*(1 + u + v)*(72*(-1 + u - v)*(1 + u - v)*(-1 + u + v)*(1 + u + v)*(power(u,4) + 3*power(v,2) - 3*power(v,4) + power(u,2)*(-1 + 2*power(v,2))) - 6*(-1 + u - v)*(1 + u - v)*(-1 + u + v)*(1 + u + v)*(-3 + 12*power(u,6) + power(v,2) + 2*power(v,4) + power(u,4)*(-8 + 26*power(v,2)) + power(u,2)*(-1 + 6*power(v,2) - 38*power(v,4)))*power(x,2) + power(u,2)*power(v,2)*(3*power(-1 + power(u,2),2)*(1 - 5*power(u,2) + 6*power(u,4)) + (-3 + 12*power(u,2) + 145*power(u,4) - 42*power(u,6))*power(v,2) + (-15 - 121*power(u,2) + 18*power(u,4))*power(v,4) + 9*(3 + 2*power(u,2))*power(v,6) - 12*power(v,8))*power(x,4))*sin((v*x)/sqrt(3))) + 3*power(power(-1 + power(u,2),2) - 2*(1 + power(u,2))*power(v,2) + power(v,4),2)*(3 + power(u,2) - 2*power(u,4) - 5*power(u,6) + 3*power(u,8) - (1 + 8*power(u,2) + 21*power(u,4))*power(v,2) + (-10 + 5*power(u,2) - 6*power(u,4))*power(v,4) + 5*power(v,6) + 3*power(v,8))*power(x,3)*((Ci(((1 + u - v)*x)/sqrt(3)) + Ci(((1 - u + v)*x)/sqrt(3)) - Ci(((-1 + u + v)*x)/sqrt(3)) - Ci(((1 + u + v)*x)/sqrt(3)))*sin(x/sqrt(3)) + cos(x/sqrt(3))*(-Si(((1 + u - v)*x)/sqrt(3)) - Si(((1 - u + v)*x)/sqrt(3)) + Si(((1 + u + v)*x)/sqrt(3)) + Si(x/sqrt(3) - (u*x)/sqrt(3) - (v*x)/sqrt(3))))))/(64.*power(u,3)*power(v,3)*power(power(-1 + power(u,2),2) - 2*(1 + power(u,2))*power(v,2) + power(v,4),2)*power(x,4))
    
   #The kernel function of the second-order eneregy density perturbation \delta^{(2)} corresponds to equation (7) in the paper.
   #I_psi, I_B, and I_phi are the kernel functions for psi, B, and phi in the article, respectively, while f_rho corresponds to S_rho in the text.
def I_delta(u, v, x):
    I1 = diPsi(u, v, x)
    I2 = I_psi(u, v, x)
    re1 = -2*(-x*I1 + f_phi(u, v, x)) # I_phi = f_phi - x*diPsi
    re2 = -2./3.*x*x*I2
    re3 = 2./3.*x*(6*I1 + x*I2 + f_B(u, v, x))  # I_B = 6*diPsi + x*I_psi + f_B 
    re4 = -2*x*I1
    return re1 + re2 + re3 + re4 + f_rho(u, v, x)