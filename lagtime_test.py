import os
import argparse
import logging
import numpy as np
from scipy import linalg

import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import misc.hummer as hummer

if __name__ == "__main__":
    """ Estimate lagtime and rate constants
    
        Calculate the empirical transfer matrix Tij from
    the hops between reactions coordinate bins. By 
    calculating the implied timescale as a function of 
    lagtime we can choose a lagtime where the implied 
    timescale plateaus. This also gives us a good guess
    for the rate matrix Rij. 
    """

    parser = argparse.ArgumentParser(description="Bayesian estimation of 1D diffusion model.")
    parser.add_argument("--coord_file", 
                        type=str, 
                        required=True,
                        help="Name of reaction coordinate file.")

    parser.add_argument("--n_bins", 
                        type=int, 
                        required=True, 
                        help="Number of bins along reaction coordinate.")

    parser.add_argument("--no_display", 
                        action="store_true",
                        help="Plot things without display available (e.g. on compute node).")

    args = parser.parse_args()

    coord_file = args.coord_file
    coord_name = coord_file.split(".")[0]
    file_ext = coord_file.split(".")[-1]
    n_bins = args.n_bins
    no_display = args.no_display

    lagtimes = [2,5,10,20,50,75,100,150,200,250,300,400]
    n_lags = len(lagtimes)

    run_directory = "%s_diff_model" % coord_name
    logfilename = "%s/implied_timescale.log" % run_directory

    if not os.path.exists(run_directory):
        os.makedirs(run_directory)

    logging.basicConfig(filename=logfilename,
                        filemode="w",
                        format="%(levelname)s:%(name)s:%(asctime)s: %(message)s",
                        datefmt="%H:%M:%S",
                        level=logging.DEBUG)

    if file_ext == "npy": 
        xfull = np.load("%s" % coord_file)
    else:
        xfull = np.loadtxt("%s" % coord_file)
        Nx,bins = np.histogram(xfull,bins=n_bins)

    os.chdir("%s_diff_model" % coord_name)
    
    all_vals = np.zeros((n_lags,n_bins))
    for tau in range(n_lags):
        lag_frames = lagtimes[tau]
        x = xfull[::lag_frames]
        n_frames = len(x)

        if not os.path.exists("lag_frames_%d_bins_%d" % (lag_frames,n_bins)):
            os.mkdir("lag_frames_%d_bins_%d" % (lag_frames,n_bins))
        os.chdir("lag_frames_%d_bins_%d" % (lag_frames,n_bins))

        print "Estimating diffusion model using: lag_frames = %d  n_bins = %d " % (lag_frames,n_bins)
        if os.path.exists("Nij.npy"):
            Nij = np.load("Nij.npy")
            bins = np.load("bins.npy")
        else:
            print "Count observed transitions between bins"
            Nij_raw = hummer.count_transitions(n_bins,n_frames,bins,x)
            Nij = 0.5*(Nij_raw + Nij_raw.T)
            np.save("Nij.npy",Nij)
            np.save("bins.npy",bins)

        # Empirical transfer matrix 
        Nij_col_sum = np.sum(Nij,axis=1)
        Tij = np.zeros((n_bins,n_bins))
        for i in range(n_bins):
            if Nij_col_sum[i] == 0:
                pass
            else:
                Tij[i,:] = Nij[i,:]/Nij_col_sum[i]

        #np.save("Tij.npy",Tij)
        S = np.linalg.eigvals(Tij)
        all_vals[tau,:] = S

        # Approximate rate matrix Rij from truncated eigenexpansion of Tij.
        #S,V = np.linalg.eig(Tij)
        #V_inv = np.linalg.inv(V)
        #Tij_approx = np.zeros(Tij.shape)
        #for i in range(5):
        #    if S[i] < 0:
        #        break
        #    else:
        #        Tij_approx += S[i]*np.outer(V[:,i],V_inv[i,:])
        #Rij = ((1./float(lag_frames))*linalg.logm(Tij_approx)).real
        #np.save("Rij_approx.npy",Rij)

        os.chdir("..")
    os.chdir("..")

    plt.plot(np.array(lagtimes),abs(all_vals))
    plt.xlabel("Lagtime (frames)")
    plt.ylabel("Eigenvalues")
    plt.title("Transition matrix eigenvalue spectra")
    if not no_display:
        plt.show()
