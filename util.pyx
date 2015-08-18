import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def count_transitions(int n_bins, int n_steps, np.ndarray[np.double_t, ndim=1] bins, np.ndarray[np.double_t, ndim=1] Q):
    """ Count the number of transitions between bins """

    cdef np.ndarray[np.double_t, ndim=2, negative_indices=False,
                    mode='c'] N = np.zeros((n_bins,n_bins))

    cdef int i,j,t

    for t in range(n_steps):
        for i in range(n_bins+1):
            if (bins[i] < Q[t]) and (bins[i+1] >= Q[t]):
                if t == 0:
                    j = i
                else:
                    N[i,j] += 1.
                    j = i
                break
    return N

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_detailed_balance_R(int n_bins, np.ndarray[np.double_t, ndim=2] Rij, np.ndarray[np.double_t, ndim=1] P):
    """Calculates the diagonal and superdiagonal of rate matrix using detailed balance"""
    cdef int j

    for j in range(n_bins - 1):
        Rij[j,j + 1] = Rij[j + 1,j]*P[j + 1]/P[j]

    for j in range(n_bins):
        if j == 0:
            Rij[j,j] = -Rij[j + 1,j]
        elif j == (n_bins - 1):
            Rij[j,j] = -Rij[j - 1,j]
        else:
            Rij[j,j] = -(Rij[j - 1,j] + Rij[j + 1,j])
    return Rij

#@cython.boundscheck(False)
#@cython.wraparound(False)
#def calculate_detailed_balance_R(int n_bins, np.ndarray[np.double_t, ndim=2] Rij, np.ndarray[np.double_t, ndim=1] P):
#    """Constrain matrix to fulfill detailed balance. Allow all elements"""
#    cdef int i,j
#    cdef np.double_t row_sum = 0.
#
#    for j in range(n_bins - 1):
#        for i in range(j + 1,n_bins):
#            Rij[i,j] = Rij[j,i]*P[j]/P[i]
#        row_sum = 0.
#        for i in range(n_bins):
#            if (i != j):
#                row_sum += Rij[j,i]
#        Rij[j,j] = -row_sum
#    Rij[n_bins - 1,n_bins - 1] = -np.sum(Rij[n_bins - 1,:n_bins - 1])
#
#    return Rij

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_logL_smoothed(int n_bins, np.ndarray[np.double_t, ndim=2] Nij, \
                        np.ndarray[np.double_t, ndim=2] exp_tRij, \
                        np.ndarray[np.double_t, ndim=2] Rij, \
                        np.ndarray[np.double_t, ndim=1] P, \
                        np.double_t deltaQ, np.double_t gamma):
    """OLD function. NOT USED """

    cdef np.ndarray[np.double_t, ndim=1, negative_indices=False,
                    mode='c'] D = np.zeros(n_bins-1)

    cdef int i,j
    cdef np.double_t neg_lnL = 0.
    cdef np.double_t D2_sum = 0.

    # Calculate -lnL. Sum over all bins in the transition matrix
    for i in range(n_bins):
        for j in range(n_bins):
            neg_lnL += Nij[i,j]*np.log(exp_tRij[i,j])
    neg_lnL = -1*neg_lnL

    # Add the smoothening part to -lnL
    for i in range(n_bins - 1):
        D[i] = (deltaQ**2)*Rij[i + 1,i]*np.sqrt(P[i]/P[i + 1])

    for i in range(n_bins - 2):
        D2_sum += (D[i] - D[i + 1])**2

    neg_lnL = neg_lnL + (0.5/(gamma**2))*D2_sum

    return neg_lnL

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_logL_smoothed_FD(int n_bins, np.ndarray[np.double_t, ndim=2] Nij, \
                        np.ndarray[np.double_t, ndim=2] Propagator, \
                        np.ndarray[np.double_t, ndim=1] D, np.double_t gamma):

    cdef int i,j
    cdef np.double_t neg_lnL = 0.
    cdef np.double_t D2_sum = 0.

    # Calculate -lnL. Sum over all bins in the transition matrix
    for i in range(n_bins):
        for j in range(n_bins):
            # What about when Propagator[i,j] == 0? -> get nan, hmm.
            #neg_lnL += Nij[i,j]*np.log(Propagator[i,j])
            if Propagator[i,j] == 0:
                neg_lnL += Nij[i,j]*np.log(0.000001)
            else:
                neg_lnL += Nij[i,j]*np.log(Propagator[i,j])
    neg_lnL = -1*neg_lnL

    # Add the smoothening part to -lnL
    for i in range(n_bins - 1):
        D2_sum += (D[i] - D[i + 1])**2

    neg_lnL = neg_lnL + (0.5/(gamma**2))*D2_sum

    return neg_lnL

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_logL(int n_bins, np.ndarray[np.double_t, ndim=2] Nij, np.ndarray[np.double_t, ndim=2] exp_tRij):

    cdef int i,j
    cdef np.double_t neg_lnL = 0.

    # Calculate -lnL. Sum over all bins in the transition matrix
    for i in range(n_bins):
        for j in range(n_bins):
            neg_lnL += Nij[i,j]*np.log(exp_tRij[i,j])
    neg_lnL = -1*neg_lnL

    return neg_lnL

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_propagator(int n_bins, np.ndarray[np.double_t, ndim=1] F, np.ndarray[np.double_t, ndim=1] D,\
                            np.double_t dx, np.double_t t_alpha):
    """Calculate propagator using analytical expression from Bicout and Szabo"""

    cdef np.ndarray[np.double_t, ndim=2, negative_indices=False,
                    mode='c'] M = np.zeros((n_bins,n_bins))
    cdef np.ndarray[np.double_t, ndim=2, negative_indices=False,
                    mode='c'] V = np.zeros((n_bins,n_bins))
    cdef np.ndarray[np.double_t, ndim=2, negative_indices=False,
                    mode='c'] rootPeq_ratio = np.zeros((n_bins,n_bins))
    cdef np.ndarray[np.double_t,ndim=1,negative_indices=False,
                    mode='c'] S = np.zeros(n_bins)
    cdef int i

    for i in range(1,n_bins - 1):
        M[i,i] = -(omega(F,D,dx,i,i + 1) + omega(F,D,dx,i,i - 1))
        M[i,i + 1] = np.sqrt(omega(F,D,dx,i,i + 1)*omega(F,D,dx,i + 1,i))
        M[i + 1,i] = np.sqrt(omega(F,D,dx,i + 1,i)*omega(F,D,dx,i,i + 1))
        M[i,i - 1] = np.sqrt(omega(F,D,dx,i,i - 1)*omega(F,D,dx,i - 1,i))
        M[i - 1,i] = np.sqrt(omega(F,D,dx,i - 1,i)*omega(F,D,dx,i,i - 1))

    M[0,0] = -omega(F,D,dx,0,1)
    M[n_bins - 1,n_bins - 1] = -omega(F,D,dx,n_bins - 1,n_bins - 2)

    # Need to change coordinates back from the symmetrized coordinates
    # by multiplying rows by sqrt(P_eq(i)/P_eq(j)) 
    S,V = np.linalg.eig(M)
    rootPeq_ratio = np.outer(np.exp(-0.5*F),np.exp(0.5*F))
    Propagator = rootPeq_ratio*abs(np.dot(V,np.dot(np.diag(np.exp(S.real*t_alpha)),V.T)))
    return Propagator

@cython.boundscheck(False)
@cython.wraparound(False)
def omega(np.ndarray[np.double_t, ndim=1] F, np.ndarray[np.double_t, ndim=1] D, np.double_t dx, int i, int j):
    """Calculate matrix element of symmetrized rate matrix"""
    return 0.5*((D[i] + D[j])/(dx**2))*np.exp(-0.5*(F[j] - F[i]))

@cython.boundscheck(False)
@cython.wraparound(False)
def attempt_scaling_D(np.double_t beta_MC, np.double_t neg_lnL, np.double_t t_alpha,
                np.ndarray[np.double_t, ndim=1] D, np.ndarray[np.double_t, ndim=1] F,
                np.ndarray[np.double_t, ndim=2] Nij, int n_bins,
                np.double_t dx, np.double_t gamma):
    """Attempt a monte carlo scaling of D, the diffusion coefficients"""

    cdef np.ndarray[np.double_t, ndim=1, negative_indices=False,
                    mode='c'] D_trial = np.zeros(n_bins)
    cdef np.ndarray[np.double_t, ndim=2, negative_indices=False,
                    mode='c'] Propagator= np.zeros((n_bins,n_bins))
    cdef np.double_t scaling_constant, neg_lnL_trial, delta_lnL

    # Scale all components of D 
    scaling_constant = (1. + np.sign(np.random.rand() - 0.5)*(0.01 + np.random.normal(scale=0.05)))
    D_trial = np.array(D,copy=True)*scaling_constant

    Propagator = calculate_propagator(n_bins,F,D_trial,dx,t_alpha)
    neg_lnL_trial = calculate_logL_smoothed_FD(n_bins,Nij,Propagator,D_trial,gamma)

    if neg_lnL_trial < neg_lnL:
        # Accept the move if it increases the log Likelihood function.
        D = D_trial
        neg_lnL = neg_lnL_trial
    else:
        delta_lnL = neg_lnL_trial - neg_lnL
        if np.random.rand() <= np.exp(-beta_MC*delta_lnL):
            # Step accepted
            D = D_trial
            neg_lnL = neg_lnL_trial
        else:
            # Step rejected
            pass

    return neg_lnL,D

@cython.boundscheck(False)
@cython.wraparound(False)
def attempt_step_D(np.double_t beta_MC, np.double_t neg_lnL,
                np.ndarray[np.double_t, ndim=1] D, np.ndarray[np.double_t, ndim=1] F,
                np.ndarray[np.double_t, ndim=1] D_step, np.double_t t_alpha,
                np.ndarray[np.double_t, ndim=2] Nij, int n_bins,
                np.ndarray[np.double_t, ndim=1] D_attempts, np.ndarray[np.double_t, ndim=1] D_accepts,
                np.double_t dx, np.double_t gamma, np.double_t step_scale):
    """Attempt a monte carlo step in D, the diffusion coefficients"""

    cdef np.ndarray[np.double_t, ndim=1, negative_indices=False,
                    mode='c'] D_trial = np.zeros(n_bins)
    cdef np.ndarray[np.double_t, ndim=2, negative_indices=False,
                    mode='c'] Propagator= np.zeros((n_bins,n_bins))
    cdef np.double_t step, neg_lnL_trial, delta_lnL
    cdef int which_bin

    which_bin = np.random.randint(n_bins)
    D_trial = np.array(D,copy=True)
    #step = np.sign(np.random.rand() - 0.5)*D_step[which_bin]
    step = np.random.normal(scale=step_scale)
    if (D_trial[which_bin] + step) <= 0:
        # Step rejected. We don't let D be negative
        pass
    else:
        D_trial[which_bin] += step

        Propagator = calculate_propagator(n_bins,F,D_trial,dx,t_alpha)
        neg_lnL_trial = calculate_logL_smoothed_FD(n_bins,Nij,Propagator,D_trial,gamma)

        if neg_lnL_trial < neg_lnL:
            # Accept the move if it increases the log Likelihood function.
            D = D_trial
            neg_lnL = neg_lnL_trial
            D_accepts[which_bin] += 1
        else:
            delta_lnL = neg_lnL_trial - neg_lnL
            if np.random.rand() <= np.exp(-beta_MC*delta_lnL):
                # Step accepted
                D = D_trial
                neg_lnL = neg_lnL_trial
                D_accepts[which_bin] += 1
            else:
                # Step rejected
                pass

    D_attempts[which_bin] += 1

    return neg_lnL,D

@cython.boundscheck(False)
@cython.wraparound(False)
def attempt_step_F(np.double_t beta_MC, np.double_t neg_lnL,
                np.ndarray[np.double_t, ndim=1] D, np.ndarray[np.double_t, ndim=1] F,
                np.ndarray[np.double_t, ndim=1] F_step, np.double_t t_alpha,
                np.ndarray[np.double_t, ndim=2] Nij, int n_bins,
                np.ndarray[np.double_t, ndim=1] F_attempts, np.ndarray[np.double_t, ndim=1] F_accepts,
                np.double_t dx, np.double_t gamma, np.double_t step_scale):
    """Attempt a monte carlo step in F, the diffusion coefficients"""

    cdef np.ndarray[np.double_t, ndim=1, negative_indices=False,
                    mode='c'] F_trial = np.zeros(n_bins)

    cdef np.ndarray[np.double_t, ndim=2, negative_indices=False,
                    mode='c'] Propagator= np.zeros((n_bins,n_bins))

    cdef np.double_t step, neg_lnL_trial, delta_lnL
    cdef int which_bin

    which_bin = np.random.randint(n_bins)
    F_trial = np.array(F,copy=True)
    #step = np.sign(np.random.rand() - 0.5)*F_step[which_bin]
    step = np.random.normal(scale=step_scale)
    F_trial[which_bin] += step

    Propagator = calculate_propagator(n_bins,F_trial,D,dx,t_alpha)
    neg_lnL_trial = calculate_logL_smoothed_FD(n_bins,Nij,Propagator,D,gamma)

    if neg_lnL_trial < neg_lnL:
        # Accept the move if it increases the log Likelihood function.
        F = F_trial
        neg_lnL = neg_lnL_trial
        F_accepts[which_bin] += 1
    else:
        delta_lnL = neg_lnL_trial - neg_lnL
        if np.random.rand() <= np.exp(-beta_MC*delta_lnL):
            # Step accepted
            F = F_trial
            neg_lnL = neg_lnL_trial
            F_accepts[which_bin] += 1
        else:
            # Step rejected
            pass

    F_attempts[which_bin] += 1

    return neg_lnL,F

