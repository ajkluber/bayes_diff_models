""" Estimate free energy and diffusion coefficient along reaction coordinate

Description
-----------
    Estimate the free energy F(x) and diffusion coefficient D(x) for
Smoluchowski diffusion along a reaction coordinate x. The likelihood of
observing some given reaction coordinate dynamics can be calculated
as the product of propagator matrix elements.

The propagator of the dynamics, of fixed lagtime dt, is:
P(j, t + dt| i, t) = prob. of hopping from bin i to bin j in time dt


"""

import logging
import argparse
import os
import time
import numpy as np
from scipy import linalg

import pyximport
pyximport.install()
import util 
from plot_and_save import plot_figures,save_datafiles

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian estimation of 1D diffusion model.")
    parser.add_argument("--coord_file", 
                        type=str, 
                        required=True,
                        help="Name of reaction coordinate file.")
    parser.add_argument("--lag_frames", 
                        type=int, 
                        required=True, 
                        help="Number of frames to subsample.")
    parser.add_argument("--n_bins", 
                        type=int, 
                        required=True, 
                        help="Number of bins along reaction coordinate.")
    parser.add_argument("--gamma", 
                        type=float, 
                        required=True, 
                        help="Smootheness scale of D.")
    parser.add_argument("--dt", 
                        type=float, 
                        default=1, 
                        help="Timestep units per frame. Default: 1ps per frame.")
    parser.add_argument("--debug", 
                        action="store_true",
                        help="Output -lnL as opimization proceeds to monitor convergence.")
    parser.add_argument("--plotfigs", 
                        action="store_true",
                        help="Plot things")
    parser.add_argument("--no_display", 
                        action="store_true",
                        help="Plot things without display available (e.g. on compute node).")
    args = parser.parse_args()

    coord_file = args.coord_file
    coord_name = coord_file.split(".")[0]
    file_ext = coord_file.split(".")[-1]
    lag_frames = args.lag_frames
    dt = args.dt
    gamma = args.gamma
    n_bins = args.n_bins
    no_display = args.no_display
    plotfigs = args.plotfigs
    debug = args.debug

    t_alpha = lag_frames*dt
    n_attempts = n_bins*2

    run_directory = "%s_diff_model/lag_frames_%d_bins_%d/gamma_%.2e" \
                  % (coord_name,lag_frames,n_bins,gamma)
    logfilename = "%s/Bayes_FD.log" % run_directory

    if not os.path.exists(coord_file):
        raise IOError("Input reaction coordinate file %s does not exist!" % coord_file)

    if not os.path.exists(run_directory):
        os.makedirs(run_directory)

    logging.basicConfig(filename=logfilename,
                        filemode="w",
                        format="%(levelname)s:%(name)s:%(asctime)s: %(message)s",
                        datefmt="%H:%M:%S",
                        level=logging.DEBUG)

    logging.info("Bayesian estimation of 1D diffusion model")
    logging.info("Parameters:")
    logging.info(" coord_file = %s" % coord_file)
    logging.info(" lag_frames = %5d" % lag_frames)
    logging.info(" n_bins     = %5d" % n_bins)
    logging.info(" gamma      = %5.2e" % gamma)
    logging.info(" dt         = %5.2e" % dt)
    logging.info(" no-display = %s" % str(no_display))
    logging.info(" plotfigs   = %s" % str(plotfigs))

    os.chdir("%s_diff_model/lag_frames_%d_bins_%d" % (coord_name,lag_frames,n_bins))
    if os.path.exists("Nij.npy"):
        # Loading transition counts between bins.
        logging.info("Loading transition counts")
        Nij = np.load("Nij.npy")
        bins = np.load("bins.npy")
        bin_centers = 0.5*(bins[1:] + bins[:-1])
        dx = bins[1] - bins[0] 
    else:
        # If bins counts have not been created. Calculate them. 
        logging.info("Calculating transition counts")
        if file_ext == "npy": 
            xfull = np.load("../../%s" % coord_file)
        else:
            xfull = np.loadtxt("../../%s" % coord_file)
        x = xfull[::lag_frames]
        Nx,bins = np.histogram(x,bins=n_bins)
        n_frames = len(x)

        Nij_raw = util.count_transitions(n_bins,n_frames,bins,x)
        Nij = 0.5*(Nij_raw + Nij_raw.T)
        np.save("Nij.npy",Nij)
        np.save("bins.npy",bins)

        bin_centers = 0.5*(bins[1:] + bins[:-1])
        dx = bins[1] - bins[0] 

    # Empirical transfer matrix. Is this transposed?
    Nij_col_sum = np.sum(Nij,axis=1)
    Tij = np.zeros((n_bins,n_bins))
    for i in range(n_bins):
        if Nij_col_sum[i] == 0:
            pass
        else:
            Tij[i,:] = Nij[i,:]/Nij_col_sum[i]

    ########################################################################
    # Initialize F, D, lnL
    ########################################################################
    os.chdir("gamma_%.2e" % gamma)
    logging.info("Initializing F, D, and log-likelihood")
    F = np.ones(n_bins)
    D = np.ones(n_bins)

    F_step = 0.01*np.ones(n_bins)
    F_attempts = np.zeros(n_bins)
    F_accepts = np.zeros(n_bins)

    D_step = 0.01*np.ones(n_bins,float)
    D_attempts = np.zeros(n_bins,float)
    D_accepts = np.zeros(n_bins,float)

    Propagator = util.calculate_propagator(n_bins,F,D,dx,t_alpha)
    neg_lnL = util.calculate_logL_smoothed_FD(n_bins,Nij,Propagator,D,gamma)

    neg_lnL_all = [neg_lnL]
    neg_lnL_all = [neg_lnL]
    D_all = [D]
    F_all = [F]

    ########################################################################
    # Perform Metropolis-Hastings monte carlo 
    ########################################################################
    beta_MC_schedule = [40.,60.]
    #beta_MC_steps = [200,100]  # Default
    beta_MC_steps = [50,5] # Testing
    D_step_scale = [0.2,0.1]
    F_step_scale = [0.1,0.01]
    n_stages = len(beta_MC_schedule)    
    logging.info("Starting Monte Carlo optimization of Likelihood L")
    starttime = time.time()
    total_steps = 0
    for b in range(n_stages):
        # Annealing stage for MC acceptance ratio
        F_scale = F_step_scale[b]
        D_scale = D_step_scale[b]
        beta_MC = float(beta_MC_schedule[b])
        n_steps = beta_MC_steps[b]
        logging.info("Stage %3d of %3d: Beta = %.4e  Total steps = %d" % (b + 1,n_stages,beta_MC,n_steps*n_attempts))
        logging.info("  Step #      -log(L)")
        if debug:
            print "Stage %3d of %3d: Beta = %.4e  Total steps = %d" % (b + 1,n_stages,beta_MC,n_steps*n_attempts)
            print "  Step #      -log(L)"
        for n in range(n_steps):
            for i in range(n_attempts):
                neg_lnL,D = util.attempt_scaling_D(beta_MC,neg_lnL,t_alpha,D,F,Nij,n_bins,dx,gamma)
                neg_lnL,D = util.attempt_step_D(beta_MC,neg_lnL,D,F,D_step,t_alpha,Nij,n_bins,D_attempts,D_accepts,dx,gamma,D_scale)
                neg_lnL,F = util.attempt_step_F(beta_MC,neg_lnL,D,F,F_step,t_alpha,Nij,n_bins,F_attempts,F_accepts,dx,gamma,F_scale)
                neg_lnL_all.append(neg_lnL)
                D_all.append(D)
                F_all.append(F)

            logging.info("  %-10d  %-15.4f" % (n*n_attempts,neg_lnL))
            if debug:
                print "  %-10d  %-15.4f" % (n*n_attempts,neg_lnL)

        total_steps += n_steps*n_attempts
    runsecs = time.time() - starttime
    if debug:
        print "Took %.2f min for %d steps, %.2e steps per sec" % (runsecs/60.,total_steps,total_steps/runsecs)
    
    Propagator = util.calculate_propagator(n_bins,F,D,dx,t_alpha)

    save_datafiles(beta_MC_schedule,beta_MC_steps,Tij,Propagator,F,D,F_all,D_all,neg_lnL_all)

    if plotfigs:
        plot_figures(neg_lnL_all,F_all,D_all,F,D,
                    bin_centers,beta_MC_schedule,beta_MC_steps,
                    n_stages,n_attempts,gamma,coord_name,no_display,
                    Propagator,Tij)

    logging.info("Done")
    os.chdir("../../..")
