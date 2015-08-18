import os
import argparse
import subprocess as sb 

def get_run_script(name,coord_file,lag_frames,n_bins,gamma,replicas=-1):
    if replicas == -1:
        script_string ="for y in {1..10..1}; do\n"
    else:
        script_string ="for y in %s; do\n" % replicas
    script_string +="""    destdir="replica_${y}/%s/iteration_0"
    cd ${destdir}
    temps=`cat long_temps`

    echo "  replica $y"
    for temp in $temps; do
        cd $temp
        python -m misc.Bayes_D --coord_file %s --lag_frames %d --n_bins %d --gamma %f
        cd ..
    done
    cd ../../..
done""" % (name,coord_file,lag_frames,n_bins,gamma)
    return script_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian estimation of 1D diffusion model.")
    parser.add_argument("--name",
                        type=str,
                        required=True,
                        help="Name.")
    parser.add_argument("--coord_file",
                        type=str,
                        required=True,
                        help="Reaction coordinate file.")
    parser.add_argument("--lag_frames",
                        type=int,
                        required=True,
                        help="Lag time in frames.")
    parser.add_argument("--n_bins",
                        type=int,
                        required=True,
                        help="Number of bins.")
    parser.add_argument("--gamma",
                        type=float,
                        required=True,
                        help="Smoothing parameter.")
    parser.add_argument("--nonnative_variances",
                        type=str,
                        nargs="+",
                        required=True,
                        help="Non-native variances. Must match number of processors.")
    parser.add_argument("--replicas",
                        type=int,
                        nargs="+",
                        help="Non-native variances. Must match number of processors.")
    parser.add_argument("--nompi",
                        action="store_true",
                        help="Don't use mpi.")

    args = parser.parse_args()
    name = args.name
    nompi = args.nompi
    nonnative_variances = args.nonnative_variances
    coord_file = args.coord_file
    lag_frames = args.lag_frames
    n_bins = args.n_bins
    gamma = args.gamma

    if args.replicas is not None:
        replicas = "" 
        for i in range(len(args.replicas)):
            replicas += "%d " % args.replicas[i]

    if not nompi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD   # MPI environment
        size = comm.Get_size()  # number of threads
        rank = comm.Get_rank()  # number of the current thread

        if not (len(nonnative_variances) == size):
            raise IOError("Number of processors %d must equal number of variances: %s" % (size,nonnative_variances.__str__()))
    
    if nompi:
        os.chdir("random_b2_%s" % nonnative_variances[0])  
    else:
        os.chdir("random_b2_%s" % nonnative_variances[rank])  
    
    with open("run_bayes.sh","w") as fout:
        if args.replicas is not None: 
            fout.write(get_run_script(name,coord_file,lag_frames,n_bins,gamma,replicas=replicas))
        else:
            fout.write(get_run_script(name,coord_file,lag_frames,n_bins,gamma))

    command = "bash run_bayes.sh"
    sb.call(command.split())

    os.chdir("..")



