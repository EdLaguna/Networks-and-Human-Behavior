# This file contains extra functions specifically writen for Econ 46 - financial networks chapter
# Created: Jan 29th 2020, Eduardo Laguna Muggenburg
# Last Modified: Feb #1st 2020, Eduardo Laguna Muggenburg

# Housekeeping: Importing packages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets       
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pl   
from scipy.stats import norm


def correlated_uniforms(N,rho=0):
    """ This function generates N uniform random variables (0,1)
        that may or may not be correlated, default is independent; when rho=0"""
    # check the correlation has acceptable value
    if rho>1 or rho<-1:
        print("Error, correlation coefficient must be between -1 and 1")
    # Generate correlation matrix, could be modified for more exotic covariance structure    
    corr_matrix=rho*np.ones((N,N))+(1-rho)*np.identity(N)
    
    # adjustment to generate normal random variables
    corr_matrix=np.sin(np.pi*corr_matrix/6)

    if isPSD(corr_matrix)!=True:
        corr_matrix=nearPD(corr_matrix)
    
    # generate multivariate normal with specified covariance
    random_normal = np.random.multivariate_normal(np.zeros(N),corr_matrix)
    # return cumulative probability of the normal, which is a uniform(0,1) random variable.
    return norm.cdf(random_normal)

def porfolio_returns(invested_amount,prob_success,R,rho):
    """ This function simulates the outcome of investing vector invested_amount
        in a simple portfolio that pays R*amount with probability p and zero otherwise.
        
        All entries of the vector are invested in loteries with the same expected value,
        the correlation between lotteries can be adjusted with rho parameter.
       
           invested_amount = array with quantity invested by each bank.
           prob_success: probability of succesful outcome.
           R: (default=1) Gross return rate.
           rho: (default=0) correlation between lotteries.
    """
    n_investing_bks = len(invested_amount)
    u = correlated_uniforms(n_investing_bks,rho).reshape(n_investing_bks,1)
    A = np.multiply(invested_amount,(u< prob_success))
    return A

def plot_weighted_edges(G): 
    # draw nodes and labels
    pos=nx.circular_layout(G) 
    insolvent_nodes = []
    solvent_nodes = []
    for node in G.nodes():
        if G.nodes[node]['insolvent']==True:
            insolvent_nodes = insolvent_nodes +[node]
        else:
            solvent_nodes = solvent_nodes +[node]
    nx.draw_networkx_nodes(G,pos,nodelist=solvent_nodes,node_color='crimson',node_size=300)
    nx.draw_networkx_nodes(G,pos,nodelist=insolvent_nodes,node_color='grey',node_size=310)
   
    nl = nx.draw_networkx_labels(G,font_color='white', font_weight='bold',
                              pos=pos,with_labels = True,node_size=300)
    # Consider all possible weights, to adjust width.
    weights = []
    for (n1,n2,df) in G.edges(data=True):
        weights.append(df['weight']) 
 
    unique_weights = list(set(weights))
    # Plot all edges of each weight with a given width
    for weight in unique_weights:
        weighted_edges = [(n1,n2) for (n1,n2,edge_attribute) in G.edges(data=True) if edge_attribute['weight']==weight]
        width = weight*len(G.nodes())*2.0/sum(weights)
        nx.draw_networkx_edges(G,pos,edgelist=weighted_edges,width=width,connectionstyle='arc3,rad=0.2')
    plt.axis('off')

# Plotting bank runs for small N
def plotting_bank_run_example(N,D_limit,Deposits,inv_share=1,prob_success=.7,R=2,rho=0):
    if N>15:
        print("Those are too many banks to get nice plots out of this. Try the function bank_cascades on its own.")
        return
    dict_fails = {}
    dict_plots = {}
    for k in range(N):
        dict_fails[k], dict_plots[k] = bank_cascades(N=N,k=k,
                                                     Deposits=Deposits,
                                                     inv_share=inv_share,
                                                     D_limit=D_limit,
                                                     prob_success=prob_success,
                                                     R=R,
                                                     rho=rho,plot_net=True)
    
    if N<5:
        fig, axes = plt.subplots(1, 4, figsize=(17, 4))
        i = 0 
        for degree in range(N):  
            if degree in dict_fails.keys():
                plt.sca(axes[i]) 
                plot_weighted_edges(dict_plots[degree])
                default_rate = 1.0*dict_fails[degree]/N
                axes[i].title.set_text('Degree = {K}, Insolvent Banks = {P} \n Default Rate = {DR}'.format(
                            K=degree, P=dict_fails[degree],DR=default_rate))
            else:
                plt.sca(axes[i])
                plt.axis("off")
            i = i+1

    
        plt.show()
    else:    
        num_plots = len(dict_plots.keys())
        rows = int(np.ceil(num_plots/4))
        fig, axes = plt.subplots(rows, 4, figsize=(17, rows*4))
        i = 0 
        j=0

        for degree in range(rows*4):    
            if degree in dict_fails.keys():
                plt.sca(axes[j][i]) # calls the subplot
                plot_weighted_edges(dict_plots[degree])
                default_rate = 1.0*dict_fails[degree]/N
                axes[j][i].title.set_text('Degree = {K}, Insolvent Banks = {P} \n Default Rate = {DR}'.format(
                            K=degree, P=dict_fails[degree],DR=default_rate))
            else:
                plt.sca(axes[j][i])
                plt.axis("off")
            i = i+1
            if i==4:
                i=0
                j = j+1
    
        plt.show()

def bank_cascades(N,k,Deposits,D_limit,inv_share,prob_success,R,rho,plot_net=False):
    """ This function simulates the model described in notebook 4.
        
        N: number of banks
        k: expected number of connections in the lending network.
        Deposits: amount of money received by each bank.
        inv_share: share of cash that bank invests.
        D_limit: amount of money that bank lends to the entire network.
        prob_success: probability of succesful outcome.
        R: Gross return rate of investment opportunities.
        rho: correlation between investment lotteries.
        plot_net: (default=False) boolean to control plotting outcome.
    """
    
    # Generae debt network from a directed ER model with expected degree equal to k
    # Base network:
    G = nx.erdos_renyi_graph(n=N,p=k/(N-1),directed=True)
    A = nx.adjacency_matrix(G).todense()
    DN = A/np.maximum(A.sum(axis=1),1)
    
    # Debt * graph = weighted graph
    Debts = np.multiply(DN,D_limit)

    # Debt network. Entry i,j is the money that goes from i to j
    G1 = nx.convert_matrix.from_numpy_matrix(Debts,create_using=nx.MultiDiGraph,
                                             parallel_edges=True)
    
    # Amount of money bank holds after borrowing and lending.
    # Summing the values of debt network across a row give the total of money lent by i.
    # Summing the values of debt network along a column give the total of money borrowed by i.
    total_lent = Debts.sum(axis=1)
    total_borrowed = Debts.sum(axis=0).T
    net_assets = Deposits - total_lent + total_borrowed
    # Amount to be invested vs to saved as cash
    investment = np.multiply(inv_share,net_assets)
    cash_shares = 1-inv_share
    cash_reserves = (1-inv_share)*net_assets
    # Investment stage
    returns = porfolio_returns(investment,prob_success,R,rho)
    # paytime & cascades
    # Outcome assuming everyone pays debts.
    outcome = returns + cash_reserves  +total_lent - total_borrowed +.000000000001
    # For plotting purposes we will keep a label for insolvent nodes. Here we set all as false.
    nx.set_node_attributes(G1, False, 'insolvent')
    
    # Now we iterate and see if even with this assumption, a bank becomes insolvent
    # if it does then we have to cancel their payments and see if other banks are affected
    # until we find the equilibrium
    outcome0 = np.zeros((N,1))
    while (outcome0!=outcome).any():
        outcome0 = outcome
        j =0 
        for insolvent_bank in outcome<0:
            # If a bank is insolvent, it goes bankrupt and pays nothing
            if insolvent_bank==True:
                Debts[:,j]=np.zeros((N,1))
                # update label
                G1.nodes[j]['insolvent'] = True
            j = j+1
        # Compute values after the bankruptcies
        outcome = np.multiply(returns + cash_reserves +Debts.sum(axis=1) - Debts.sum(axis=0).T+.000000000001,
                              outcome0>=0)+np.multiply(outcome0, outcome0<0)
    tot_insolvent = len([n for n,a in G1.nodes(data=True) if a['insolvent']==True])

    if plot_net==True:
        return  tot_insolvent, G1
    else:
        return tot_insolvent

def simulating_bank_runs(num_simulations,N,Deposits,inv_share,D_limit,prob_success,R,rho):
    # start with empty dictionary to save results later
    dict_fails = {}
    # Loop all possible degrees
    for k in range(N):
        # Empty dictionary for simulations for degree k
        dict_sims = 0
        for n in range(num_simulations):
            # Simulate bank cascade
            dict_sims = dict_sims+1.0*bank_cascades(N,k,Deposits,D_limit,inv_share,prob_success,R,rho)/N
        # Save avg failure rate for each degree
        dict_fails[k] = 1.0*dict_sims/num_simulations
     
    # transform dictionary keys and values into pairs for plotting
    x, y = zip(*sorted(dict_fails.items()))
    plt.figure(figsize=(17,8))
    plt.bar(x,y,color='crimson')
    plt.xlabel("Expected number of cross-bank lending relationships",fontsize=12)
    plt.ylabel("Default Rates",fontsize=12)
    plt.show()  
    
# Functions to make sure the covariance matrix in the uniform simulation is valid (from stackoverflow):    
def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearPD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n) 
# W is the matrix used for the norm (assumed to be Identity matrix here)
# the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk

def isPSD(A, tol=1e-8):
    E = np.linalg.eigvalsh(A)
    return np.all(E > -tol)
