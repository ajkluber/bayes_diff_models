""" Bayesian estimates of F(Q) and D(Q)



Ingredients:
  1. Equilibrium trajectory Q(t)

Procedure: 
    1. Assume uniform prior distribution of equilibrium probabili



1. Read in coordinate. Given n_bins calculate bin_width and Nij transition
    counts.
2. Initialize -lnL with smoothening prior. g = -lnP, Rij = rate coefficient
    matrix.
3. Monte Carlo loop
    For step in n_steps:
        For attempt in n_attempts:
            Attempt a step in g 
            Attempt a step in R

        Updating MC step sizes delta_g, delta_R using scheme of Miller, Anon,
        Reinhardt in order to bring acceptance ratio to 0.5
            if acceptance ratio < 0.5 -> multiply step by 0.95
            if acceptance ratio > 0.5 -> multiply step by 1.05

4. Do another loop to determine the error bars
"""

import argparse
import os
import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt
import matplotlib

import hummer

global GAS_CONSTANT_KJ_MOL
GAS_CONSTANT_KJ_MOL = 0.0083145

def imshow_mask_zero(N):
    # Plot matrix but mask zeros.
    M = np.array(N,copy=True) 
    M[M == 0] = np.NaN
    jet = matplotlib.cm.jet
    temp = np.ma.array(M,mask=np.isnan(M))
    jet.set_bad('w',1.)
    plt.imshow(temp,cmap=jet,interpolation='none')
    plt.colorbar()
    plt.show()

def attempt_step_g(neg_lnL,g,g_step,Rij,Nij,P,beta,t_alpha,n_bins,g_accepts,g_attempts,gamma,deltaQ):
    """Attempt a monte carlo step in g, the free energy of bins"""

    which_bin = np.random.randint(n_bins)
    g_trial = np.array(g,copy=True)
    g_trial[which_bin] += np.sign(np.random.rand() - 0.5)*g_step[which_bin]

    P_trial = deltaQ*np.exp(-beta*g_trial)
    P_trial /= sum(P_trial)

    Rij_trial = hummer.calculate_detailed_balance_R(n_bins,Rij,P_trial)
    
    exp_tRij = linalg.expm(t_alpha*Rij_trial)
    neg_lnL_trial = hummer.calculate_logL_smoothed(n_bins,deltaQ,gamma,Nij,Rij_trial,exp_tRij,P_trial)
    
    if neg_lnL_trial < neg_lnL:
        # Accept the move if it increases the log Likelihood function.
        g = g_trial
        P = P_trial
        Rij = Rij_trial
        neg_lnL = neg_lnL_trial
        g_accepts[which_bin] += 1
    else:
        delta_lnL = neg_lnL_trial - neg_lnL
        if np.random.rand() <= np.exp(-delta_lnL):
            # Step accepted
            g = g_trial
            P = P_trial
            Rij = Rij_trial
            neg_lnL = neg_lnL_trial
            g_accepts[which_bin] += 1
        else:
            # Step rejected
            pass

    g_attempts[which_bin] += 1

    return neg_lnL,Rij,P,g

def attempt_step_Rij(neg_lnL,Rij,R_step,Nij,P,t_alpha,beta,n_bins,gamma,deltaQ,R_attempts,R_accepts):
    """Attempt a monte carlo step in Rij, the rate matrix"""

    # attempt a step in R
    which_bin = np.random.randint(n_bins - 1)
    Rij_trial = np.array(Rij,copy=True)
    Rij_trial[which_bin + 1,which_bin] += \
        np.sign(np.random.rand() - 0.5)*np.random.rand()*R_step[which_bin]

    Rij_trial = hummer.calculate_detailed_balance_R(n_bins,Rij_trial,P)

    exp_tRij = linalg.expm(t_alpha*Rij_trial)
    neg_lnL_trial = hummer.calculate_logL_smoothed(n_bins,deltaQ,gamma,Nij,Rij,exp_tRij,P)

    if neg_lnL_trial < neg_lnL:
        # Accept the move if it increases the log Likelihood function.
        Rij = Rij_trial
        neg_lnL = neg_lnL_trial
        R_accepts[which_bin] += 1
    else:
        delta_lnL = neg_lnL_trial - neg_lnL
        if np.random.rand() <= np.exp(-delta_lnL):
            # Step accepted
            Rij = Rij_trial
            neg_lnL = neg_lnL_trial
            R_accepts[which_bin] += 1
        else:
            # Step rejected
            pass
    R_attempts[which_bin] += 1

    return neg_lnL,Rij,P

def monte_carlo_optimization(beta,n_bins,lag_frames,t_alpha,gamma,Q,n_attempts,n_equil_steps,n_steps):

    #############################################################################
    # Initialize Nij,Rij,D,g
    #############################################################################
    n_frames = len(Q)
    bins = np.linspace(min(Q),max(Q),num=n_bins+1)
    deltaQ = bins[1] - bins[0]

    print "Estimating diffusion model using: lag_frames = %d  n_bins = %d " % (lag_frames,n_bins)
    if os.path.exists("Nij_%d_%d.dat" % (lag_frames,n_bins)):
        Nij = np.loadtxt("Nij_%d_%d.dat" % (lag_frames,n_bins))
        #shutil.move
    else:
        print "Count observed transitions between bins"
        Nij = hummer.count_transitions(n_bins,n_frames,bins,Q)
        np.savetxt("Nij_%d_%d.dat" % (lag_frames,n_bins),Nij)

    #imshow_mask_zero(Nij) 

    print "Initializing guess for P_i, R_ij"
    P = np.ones(n_bins,float)
    P /= sum(P)
    g = -np.log(P)

    Rij = np.zeros((n_bins,n_bins),float)
    for i in range(1,n_bins):
        #Rij[i,i - 1] = 100.*np.random.rand()
        Rij[i,i - 1] = 50.

    Rij = hummer.calculate_detailed_balance_R(n_bins,Rij,P)
    exp_tRij = linalg.expm(t_alpha*Rij)
    neg_lnL = hummer.calculate_logL_smoothed(n_bins,deltaQ,gamma,Nij,Rij,exp_tRij,P)

    #############################################################################
    # Monte Carlo optimization of g and Rij. Equilibration.
    #############################################################################
    a = 0.6729
    b = 0.0644
    r_ideal = 0.5

    g_accepts = np.zeros(n_bins,float)
    g_attempts = np.zeros(n_bins,float)

    R_accepts = np.zeros(n_bins - 1,float)
    R_attempts = np.zeros(n_bins - 1,float)

    #g_step = np.random.rand(n_bins)
    #R_step = np.random.rand(n_bins - 1)
    g_step = 0.1*np.ones(n_bins,float)
    R_step = np.ones(n_bins - 1,float)
    print "  Running %d equilibration steps:" % n_equil_steps
    g_equil = np.zeros((n_equil_steps,n_bins),float)
    R_equil = np.zeros((n_equil_steps,n_bins - 1),float)
    for n in range(n_equil_steps):
        #for x in range(n_attempts + np.random.randint(50)):
        for x in range(n_attempts):
            neg_lnL,Rij,P,g = attempt_step_g(neg_lnL,g,g_step,Rij,Nij,P,beta,t_alpha,n_bins,g_accepts,g_attempts,gamma,deltaQ)
            neg_lnL,Rij,P = attempt_step_Rij(neg_lnL,Rij,R_step,Nij,P,t_alpha,beta,n_bins,gamma,deltaQ,R_attempts,R_accepts)

        #if np.all(g_attempts):
        #    ratio_g = g_accepts/g_attempts
        #    g_step[ratio_g < 0.5] *= 0.95
        #    g_step[ratio_g > 0.5] *= 1.05
        #    #print "g_step ",ratio_g
        #    #print(g_attempts)
        #if np.all(R_attempts):
        #    ratio_R = R_accepts/R_attempts
        #    R_step[ratio_R < 0.5] *= 0.95
        #    R_step[ratio_R > 0.5] *= 1.05
        #    #print(ratio_R)
        #    #print "R_step ",R_step
        
        if (n % 10) == 0:
            print "   %5d  %20.4f" % (n,neg_lnL)
        g_equil[n,:] = g
        R_equil[n,:] = Rij[np.arange(n_bins - 1) + 1,np.arange(n_bins - 1)]

    np.savetxt("g_equil.dat",g_equil)
    np.savetxt("R_equil.dat",R_equil)
    os.chdir("..")
    raise SystemExit

    #############################################################################
    # Monte Carlo optimization of g and Rij. Production run.
    #############################################################################
    #g_step *= 0.5
    #R_step *= 0.5

    g_all = np.zeros((n_steps,n_bins),float)
    F_all = np.zeros((n_steps,n_bins),float)
    D_all = np.zeros((n_steps,n_bins-1),float)
    print "  Running %d production steps:" % n_steps
    for n in range(n_steps):
        #for x in range(n_attempts + np.random.randint(50)):
        for x in range(n_attempts):
            neg_lnL,Rij,P,g = attempt_step_g(neg_lnL,g,g_step,Rij,Nij,P,beta,t_alpha,n_bins,g_accepts,g_attempts,gamma,deltaQ)
            neg_lnL,Rij,P = attempt_step_Rij(neg_lnL,Rij,R_step,Nij,P,t_alpha,beta,n_bins,gamma,deltaQ,R_attempts,R_accepts)

        #if np.all(g_attempts):
        #    ratio_g = g_accepts/g_attempts
        #    g_step[ratio_g < 0.5] = g_step[ratio_g < 0.5]*0.95
        #    g_step[ratio_g > 0.5] = g_step[ratio_g > 0.5]*1.05
        #    #g_step += ((ratio_g - 0.5)**9)*g_step
        #    #g_step *= np.log(a*r_ideal + b)/np.log(a*ratio_g + b)
        #    #print(ratio_g)
        #    #print(g_attempts)
        #if np.all(R_attempts):
        #    ratio_R = R_accepts/R_attempts
        #    R_step[ratio_R < 0.5] = R_step[ratio_R < 0.5]*0.95
        #    R_step[ratio_R > 0.5] = R_step[ratio_R > 0.5]*1.05
        #    #R_step += ((ratio_R - 0.5)**9)*R_step
        #    #R_step *= np.log(a*r_ideal + b)/np.log(a*ratio_R + b)
        #    #print(ratio_R)
        
        if (n % 10) == 0:
            print "   %5d  %20.4f" % (n,neg_lnL)

        F_all[n,:] = -np.log(P/deltaQ)
        g_all[n,:] = g
        for i in range(n_bins-1):
            D_all[n,i] = (deltaQ**2)*Rij[i + 1,i]*np.sqrt(P[i]/P[i + 1])

    # Save 
    exp_tRij = linalg.expm(t_alpha*Rij)
    np.savetxt("expRij.dat",exp_tRij)
    np.savetxt("Rij.dat",Rij)
    np.savetxt("P.dat",P)
    np.savetxt("g_all.dat",g_all)
    np.savetxt("F_all.dat",F_all)
    np.savetxt("D_all.dat",D_all)
    np.savetxt("Qbins.dat",bins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--temp', type=float, required=True, help='Temperature in Kelvin')
    parser.add_argument('--coord', type=str, default="Qnrm.dat", help='Name of reaction coordinate timeseries. Default Qnrm.dat')
    parser.add_argument('--n_bins', type=int, default=25, help='Num bins along coordinate. Default 25')
    parser.add_argument('--lag_frames', type=int, default=50, help='Lagtime as # frames. Default 50')
    parser.add_argument('--delta_t', type=float, default=0.5, help='Timestep size. Default 0.5 ps')
    parser.add_argument('--gamma', type=float, default=0.001, help='Smoothing scale for D(x). Default 0.01')
    args = parser.parse_args()

    coord = args.coord              # Name of reaction coordinate timeseries
    T = args.temp                   # Temperature in Kelvin
    n_bins = args.n_bins            # Number of bins along coordinate
    lag_frames = args.lag_frames    # Timelag in # of frames
    delta_t = args.delta_t          # Timestep size in ps
    gamma = args.gamma              # Scale over which D(x) should be enforced to be smooth

    n_attempts = 200      # Number of times to attempt monte carlo moves per step
    n_equil_steps = 200  # Number of monte carlo steps to reach equilibration
    n_steps = 200        # Number of monte carlo steps for production run

    t_alpha = delta_t*lag_frames      # Lagtime in ps
    beta = 1./(GAS_CONSTANT_KJ_MOL*T) 

    # Assume Q has been normalized already
    Qfull = np.loadtxt("%s" % coord)
    
    Q = Qfull[::lag_frames]

    if not os.path.exists("lag_time_%d_bins_%d" % (lag_frames,n_bins)):
        os.mkdir("lag_time_%d_bins_%d" % (lag_frames,n_bins))
    os.chdir("lag_time_%d_bins_%d" % (lag_frames,n_bins))
    monte_carlo_optimization(beta,n_bins,lag_frames,t_alpha,gamma,Q,n_attempts,n_equil_steps,n_steps)
    os.chdir("..")
