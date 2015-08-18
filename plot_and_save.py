import numpy as np

def plot_and_save(neg_lnL_all,F_all,D_all,F,D,
                bin_centers,beta_MC_schedule,beta_MC_steps,
                n_stages,n_attempts,gamma,coord_name,no_display,
                Propagator,Nij,save=True):

    if no_display:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    neg_lnL_all = np.array(neg_lnL_all)
    neg_lnL_all /= neg_lnL_all[0]
    D_all = np.array(D_all)
    F_all = np.array(F_all)

    # Empirical transfer matrix 
    Nij_col_sum = np.sum(Nij,axis=1)
    Tij = np.zeros((n_bins,n_bins))
    for i in range(n_bins):
        if Nij_col_sum[i] == 0:
            pass
        else:
            Tij[i,:] = Nij[i,:]/Nij_col_sum[i]


    logging.info("Saving:")
    logging.info("  annealing_schedule")
    logging.info("  F_final.dat D_final.dat")
    logging.info("  F_all.npy D_all.npy")
    logging.info("  neg_lnL_all.npy")
    logging.info("  Propagator.npy")
    logging.info("  Tij.npy")
    if save:
        with open("annealing_schedule","w") as fout:
            fout.write("#Beta  n_steps\n")
            for i in range(len(beta_MC_schedule)):
                fout.write("%.5f  %d\n" % (beta_MC_schedule[i],beta_MC_steps[i]))
            
        np.save("Tij.npy",Tij)
        np.save("Propagator.npy",Propagator)
        np.savetxt("F_final.dat",F)
        np.savetxt("D_final.dat",D)
        np.save("F_all.npy",F_all)
        np.save("D_all.npy",D_all)
        np.save("neg_lnL_all.npy",neg_lnL_all)

    logging.info("Plotting: (everything as png & pdf)")
    logging.info("  lnL")
    logging.info("  F_final  F_all")
    logging.info("  D_final  D_all")

    plt.figure()
    plt.pcolormesh(Tij)
    plt.colorbar()
    plt.xlabel("bin j")
    plt.ylabel("bin i")
    plt.title("Empirical Propagator $T_{ij}$")
    if save:
        plt.savefig("Tij.png")
        plt.savefig("Tij.pdf")

    plt.figure()
    plt.pcolormesh(Propagator)
    plt.colorbar()
    plt.xlabel("bin j")
    plt.ylabel("bin i")
    plt.title("Diffusive Model Propagator $P(j,dt|i,0)$")
    if save:
        plt.savefig("Propagator.png")
        plt.savefig("Propagator.pdf")

    plt.figure()
    plt.plot(neg_lnL_all)
    sum_steps = 0
    for i in range(n_stages):
        sum_steps += beta_MC_steps[i]*n_attempts
        plt.axvline(x=sum_steps,color='k')
        plt.text(sum_steps - 0.8*beta_MC_steps[i]*n_attempts,
                0.9*(max(neg_lnL_all) - min(neg_lnL_all)) + min(neg_lnL_all),
                "$\\beta=%.1e$" % beta_MC_schedule[i],fontsize=16)
    plt.title("Negative log likelihood",fontsize=16)
    if save:
        plt.savefig("lnL.png")
        plt.savefig("lnL.pdf")

    plt.figure()
    plt.plot(D_all)
    sum_steps = 0
    for i in range(n_stages):
        sum_steps += beta_MC_steps[i]*n_attempts
        plt.axvline(x=sum_steps,color='k')
    plt.xlabel("MC steps",fontsize=16)
    plt.title("Diffusion coefficient $\\gamma=%.2e$" % gamma,fontsize=16)
    if save:
        plt.savefig("D_all.png")
        plt.savefig("D_all.pdf")

    plt.figure()
    plt.plot(F_all)
    sum_steps = 0
    for i in range(n_stages):
        sum_steps += beta_MC_steps[i]*n_attempts
        plt.axvline(x=sum_steps,color='k')
    plt.xlabel("MC steps",fontsize=16)
    plt.title("Free energy $\\gamma=%.2e$" % gamma,fontsize=16)
    if save:
        plt.savefig("F_all.png")
        plt.savefig("F_all.pdf")

    plt.figure()
    plt.plot(bin_centers,F)
    plt.xlabel("%s" % coord_name,fontsize=16)
    plt.ylabel("F(%s) (k$_B$T)" % coord_name,fontsize=16)
    plt.title("Final Free energy $\\gamma=%.2e$" % gamma,fontsize=16)
    if save:
        plt.savefig("F_final.png")
        plt.savefig("F_final.pdf")

    plt.figure()
    plt.plot(bin_centers,D)
    plt.xlabel("%s" % coord_name,fontsize=16)
    plt.ylabel("D(%s)" % coord_name,fontsize=16)
    plt.title("Final Diffusion coefficient $\\gamma=%.2e$" % gamma,fontsize=16)
    if save:
        plt.savefig("D_final.png")
        plt.savefig("D_final.pdf")

    if not no_display:
        plt.show()
