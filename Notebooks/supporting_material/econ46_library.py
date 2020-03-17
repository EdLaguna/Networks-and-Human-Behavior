# This file contains a handful of functions specifically writen for Econ 46
# Created: May 27th 2019, Eduardo Laguna Muggenburg
# Last Modified: Feb 18th 2020, Eduardo Laguna Muggenburg

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
import random as rn

# Reading edgelists
def read_network_from_file(file_name):
    """ This function takes a file name (and path) and returns a network
        created using csv or xls edge lists or loads a network from a pajek file.
        
        There are many other potential formats in which a network could be saved and
        therefore loaded (e.g. adjacency matrix). NetworkX provides a lot of functionalities
        which can be found by doing a quick search on the web. 
    """
    # First we determine the file  format using the file extension
    if file_name.lower().endswith('.csv'):
        try:
            G = nx.from_pandas_edgelist(pd.read_csv(file_name))
            return G
        except FileNotFoundError:
            print('Could not find the file. Make sure the path and file extension are correct.')
        except:
            print('Error: are you sure the file contains an edgelist?')
    if file_name.lower().endswith(('.xls','.xlsx')):
        try:
            G=nx.from_pandas_edgelist(pd.read_excel(file_name))
            return G
        except FileNotFoundError:
            print('Could not find the file. Make sure the path and file extension are correct.')
        except:
            print('Error: are you sure the file contains an edgelist?')
    if file_name.lower().endswith(('.net','.pajek')):
        try:
            G=nx.read_pajek(file_name)
            return G
        except FileNotFoundError:
            print('Could not find the file. Make sure the path and file extension are correct.')
    

def network_selector(network,N,p=0,m=0,path=''):
    """ This function provides a simple generator of random networks.
        It is used in the interactive notebooks but could be used to 
        quickly generate a randfom network.
    """
    if network=='Erdos-Renyi':
        G = nx.erdos_renyi_graph(N,p,seed=65489, directed=False)
    elif network=='Barabasi Albert':
        G = nx.barabasi_albert_graph(N,m,seed=65489)
    elif network=='Powerlaw Cluster':
        G = nx.powerlaw_cluster_graph(N,m,p,seed=65489)
    elif network=='Watts-Strogatz':
        if m<2:
            m=2
        G = nx.watts_strogatz_graph(N,k=m,p=p,seed=65489)
    elif network=='Newman-Watts-Strogatz':
        G = nx.newman_watts_strogatz_graph(N,k=m,p=p,seed=65489)
    elif network=='Wheel Network':
        G = nx.wheel_graph(N)
    elif network=='Regular Network':
        G = nx.random_regular_graph(d=m,n=N) 
    elif network=='Dutch High School':
        G = read_network_from_file('../Data/dutch_hs/dutch_class_t0.csv')
    elif network=='From File':
        G = read_network_from_file(path)
    return G


def layout_selection(layout,G,seed=8769):
    """ This is a utility function for selecting plotting layouts according to
        some canned algorithms.
    """
    if layout == 'Circle':
        pos = nx.circular_layout(G)
    elif layout == 'Kamada Kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'Spring':
        pos = nx.spring_layout(G,seed=seed)
    elif layout == 'Random':
        pos = nx.random_layout(G)
    return pos

def centrality_dictionaries(G,Attribute,partition_label=''):
    """ This function allows you to vary the size of the nodes
        so that they are proportional to some centrality measures. 
    """
    if Attribute=='Degree Centrality':
        k = 100
        return dict(G.degree), k
    elif Attribute=='Clustering Centrality':
        k = 1000
        return dict(nx.clustering(G)), k
    elif Attribute=='Betweenness Centrality':
        k= 1000
        return dict(nx.betweenness_centrality(G)), k
    elif Attribute=='Eigenvector Centrality':
        k= 1000
        return dict(nx.eigenvector_centrality(G)), k
    elif Attribute=='Pagerank Centrality':
        k= 1000
        return (nx.pagerank(G)), k
    elif Attribute=='Diffusion Centrality':
        k= 200
        return diffusion_centrality(G), k    
    elif Attribute=='Relative Homophily':
        k= 1000
        node_sizes, _, Hb, IHb = homophily_stats(G,type_id=partition_label) 
        print('Relative Homophily by type:\n',Hb)
        return node_sizes, k
    elif Attribute=='Inbreeding Homophily':
        k= 1000
        _, node_sizes, Hb, IHb = homophily_stats(G,type_id=partition_label)
        print('Inbreeding Homophily by type:\n',IHb)
        return node_sizes, k

# Plotting networks
def plot_simple_graph(G,Attribute='',fig_dim=8,node_base_size=500, layout='Spring',G_nodes=[]):     
    """ This function creates most of the network plots presented in the notebooks.
        
        G_nodes allows to fix one version of the network when determining the position of
        nodes while drawing only the edges that appear in network G. This is useful when
        plotting a network through different periods or when plotting different types of edges.
    """    
    # Set node position based on layout algorithm:
    if G_nodes ==[]:
        G_nodes = G
    pos = layout_selection(layout,G_nodes)
        
    # Attributes to determine  node size
    node_list = list(G.nodes()) 
    if Attribute!='':
        node_sizes, k =  centrality_dictionaries(G,Attribute)
        node_size=[max(400,500+node_sizes[i] * k) for i in node_list]
    else:
        node_size = node_base_size
    
    # Actual plotting:
    f=plt.figure(figsize=(fig_dim, fig_dim))

    ec = nx.draw_networkx_edges(G, pos, alpha=1)
    nc = nx.draw_networkx_nodes(G,  node_color='crimson',nodelist=node_list,alpha=1,font_color='white', 
                                font_weight='bold',pos=pos,with_labels = True,node_size=node_size)
    nl = nx.draw_networkx_labels(G,font_color='white', font_weight='bold',
                              pos=pos,with_labels = True,node_size=500)
    plt.axis('off')
    plt.tight_layout()       
    plt.show()

def simple_random_block_graph(n1=5,n2=5,n3=5,n4=0,n5=0,
                     n6=0,n7=0,n8=0,P=[],seed=65489): 
    """ This function wraps NetworkX stochastic block model
        with a simple random generator for the within and across group link probabilities.
    """   
    np.random.seed(seed)
    if P==[]:
        U = np.random.uniform(0,.1,size=(8,8))
        P = (U + U.T)/2+np.diag(np.diag(np.random.uniform(.6,.9,size=(8,8))))
        
    n=[n1, n2, n3,n4, n5,n6,n7,n8]
    G = nx.stochastic_block_model(sizes=n, p=P,seed=seed)
    
    return G , 'block'    

def plot_block_graph(G, partition_label, Attribute='',layout='Spring',
                     G_nodes=[],type_description='',node_base_size=500,fig_dim=8,seed=65489): 

    # Set node position based on layout algorithm:
    if G_nodes ==[]:
        G_nodes = G
    pos = layout_selection(layout,G_nodes)
    
    # Attributes to determine  node size
    node_list = list(G.nodes()) 
    if Attribute!='':
        node_sizes, k =  centrality_dictionaries(G,Attribute,partition_label)
    else:
        node_size = node_base_size

    
    # Plotting
    f=plt.figure(figsize=(fig_dim, fig_dim))
    
    ec = nx.draw_networkx_edges(G, pos, alpha=1)
       
    # Assign different color to each group and plot each partition separately:
    partition = nx.get_node_attributes(G,partition_label)
    partitions = list(set(partition.values()))
    partitions.sort()
    # This is a colorblind friendly palette
    color_list = [ '#d55e00', '#0072b2','#f0e442','#009e73',
                 '#e69f00','#cc79a7','#56b4e9','#000000'] 

    color_idx=0
    for block in partitions:
        node_sublist = [node for node in node_list if partition[node]==block]
        if Attribute!='':
            node_size=[max(400,500+node_sizes[i] * k) for i in node_sublist]
        nc = nx.draw_networkx_nodes(G, pos, nodelist=node_sublist, node_color=color_list[color_idx], 
                        font_color='black', font_weight='bold', node_size=node_size)
        color_idx = color_idx+1
    # Add node labels in  white font on top of all.
    nl = nx.draw_networkx(G, node_color='none',alpha=1,font_color='white', font_weight='bold',
                              pos=pos,with_labels = True,node_size=500)
    
    # Create color-coded legend 
    ax = plt.gca()
    
    for color, label in zip(color_list,partitions):
        ax.plot([0],[0],
                color=color,
                label=label)
               
    plt.axis('off')
    
    # Shrink axis's height by 10% on the bottom to fit the legend below
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width*0.9, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5,title=type_description)

    plt.tight_layout()       
    plt.show() 

    
# Dynamic Network Formation Models
def g_uniform(N,m):
    """ This function simulaltes a random netowrk fomration model in which nodes enter the network one by one
        and form m links with existing nodes at random.    
    """
    # Start with m nodes all friends with each other to initialize the process.
    G_u = nx.complete_graph(m)

    # add each subsequent node 
    for node in range(m,N):
        G_u.add_node(node)
        # randomly select m existing nodes to create links
        for node_j in rn.sample(list(range(node)), m):
            G_u.add_edge(node,node_j)
    
    return G_u

def g_hybrid(N,m,a):
    """ This function simulaltes a random netowrk fomration model in which nodes enter the network one by one
        and form m links with existing nodes at a fraction (a) of the links are created uniformly at random
        while teh rest (1-a) are created with probability proportional to existing nodes. 
    """
    # Start with m nodes all friends with each other to initialize the process.
    G_h = nx.complete_graph(m)

    # add each subsequent node 
    for node in range(m,N):
        G_h.add_node(node)
        m_random = int(np.ceil(a*m))
        m_pa = m - m_random
        # randomly select m_random existing nodes to create links
        for node_j in rn.sample(list(range(node)), m_random):
            G_h.add_edge(node,node_j)
        # add m_pa more edges to existing nodes with probability proportional to degree
        node_sublist = list(range(4))
        degree_sublist = list(dict(G_h.degree(list(range(4)))).values())
        repeated_list = [item for item, count in zip(node_sublist, degree_sublist) for i in range(count)]
        for node_j in rn.sample(repeated_list, m_pa):
            G_h.add_edge(node,node_j)
    
    return G_h

def g_friends_of_friends(N,m,a):
    """ This function simulaltes a random netowrk fomration model in which nodes enter the network one by one
        and form m links with existing nodes at a fraction (a) of the links are created uniformly at random
        while the rest (1-a) are created by connecting to friends of the firstly created links.     
    """
    # Start with m nodes all friends with each other to initialize the process.
    G_fof = nx.complete_graph(m)

    # add each subsequent node 
    for node in range(m,N):
        G_fof.add_node(node)
        m_random = int(np.ceil(a*m))
        m_fof = m - m_random
        # randomly select m_random existing nodes to create links
        friends_of_friends = [] 
        for node_j in rn.sample(list(range(node)), m_random):
            G_fof.add_edge(node,node_j)
            friends_of_friends = friends_of_friends+[n for n in G_fof.neighbors(node_j)]
        for node_j in rn.sample(friends_of_friends, m_fof):
            G_fof.add_edge(node,node_j)
    
    return G_fof
    


# Random networks
def plot_random_graph(n=10,p=.12,Attribute='',layout='Circle',types=False,n1=0,p1=.1,p2=.1,px=0): 
    """ This function wraps the plotting functions from above to generate a quick comparison
        between a random graph and a two-group random graph to study homophily in the intercative
        notebook.
    """
    global edgelist_h
    if types == False:
        G = nx.erdos_renyi_graph(n, p, seed=65489, directed=False)
        plot_simple_graph(G,Attribute,layout=layout,G_nodes=[])
        edgelist_h = nx.to_pandas_edgelist(G)
    if types == True:
        P = [[p1,px,0,0,0,0,0,0],[px,p2,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
        G, block = simple_random_block_graph(n1=n1,n2=n-n1,n3=0,n4=0,n5=0,
                     n6=0,n7=0,n8=0,                     
                     P=P)
        plot_block_graph(G, partition_label=block,fig_dim=8,Attribute=Attribute,layout=layout,seed=65489)
        edgelist_h = nx.to_pandas_edgelist(G)
        
def homophily_stats(G,type_id='block'):
    """ This function computes the extend to which nodes of a given type tend to associate with others
        of the same type beyond what would be expected if meetings happened at random.
        See class notes on homophily.
    """
    
    N   = len(G.nodes())
    Hi  = {}
    IHi = {}
    Hb  = {}
    IHb = {}
    
    partition = nx.get_node_attributes(G,type_id)
    for block in set(partition.values()):
        node_sublist=[node for node in G.nodes() if partition[node]==block]
        # group-level homophily stats
        wi = len(node_sublist)/N # share of type "block" out of all nodes
        Gs = G.subgraph(node_sublist)
        si = sum(dict(Gs.degree).values())
        di = sum(dict(G.degree(node_sublist)).values())
        Hb[block]  = si/di  if di != 0 else np.nan
        IHb[block] = (Hb[block]-wi)/(1-wi)
        
        # individual-level homophily stats
        with np.errstate(divide='ignore'):
            for node in node_sublist:
                Hi[node]  = Gs.degree[node]/G.degree[node] if G.degree[node] != 0 else np.nan
                IHi[node] = (Hi[node]-wi)/(1-wi)
 
    return Hi, IHi, Hb, IHb

def diffusion_centrality(G,p=[],T=[]):
    """ This function computes the Diffusion Centrality from Banerjee et.al. (2013).
         It measures the expected number of information receivers for each node.
         
         If no probability of transmision (p) is specified, it selects 1/eigv1 where eigv1 is the 
         largest eigenvalue of the adjacency matrix.
         
         If the number of periods in which the information will be allowed to travel (T)
         is not specified, it uses the diameter of the largest component.
    """
        
    # get node list to keep order consistent
    node_list = list(G.nodes())
    node_list.sort()
    
    # Initialize Dictionary
    DC =  {}
    # Get adjacency Matrix
    A = nx.to_numpy_matrix(G,nodelist=node_list)
    if p == []:
        p = 1/max(np.linalg.eigvals(A).real)
    if T == []:
        T = nx.diameter(max(nx.connected_component_subgraphs(G), key=len))
    pA = p*A
    for t in range(1,T):
        pA = pA+np.linalg.matrix_power(p*A, t+1)
    dc = pA.sum(axis=1)
    for idx, node in enumerate(node_list):
        DC[node]=dc[idx]
    return DC
       
        
# Degree distribution analysis
def plot_centrality_dist(G,Attribute='Degree Centrality'):
    """ This function plots the historgam and cumulative distribution of 
        selected centrality measures
    """
    
    # Transform values from dictionary into a list of numbers.
    vals, _ = centrality_dictionaries(G,Attribute)
    #vals2 = vals.values()
    # Get unique values to generate bins that make sense for the range of the distribution.
    n_bins = len(set(vals))
    
    # Create figure
    fig = plt.figure(figsize=(16, 5))

    ax1 = fig.add_subplot(1, 2, 1) # Create first subplot
    pd.Series(vals).hist(bins=n_bins,density=False) # transform list into pandas series to use histogram function.
    ax1.title.set_text(Attribute)
    
    ax2 = fig.add_subplot(1, 2, 2)
    pd.Series(vals).hist(bins=n_bins,cumulative=1,density=True, histtype='step')
    ax2.title.set_text('Cumulative ' + Attribute)

    plt.show()

#  Contagion  Processes
def contagion(G,initial,T=5,p=1,spreading='simple',threshold=0,share=0,vaccinated=[],random_seed=876120):
    """ This function models contagion processes in which a list of node ids is provided as
        initially "infected". Once a node is infected it cannot recover.
       
        A variety of situations can be modeled:
            Simple: infected node infects neighbors with a fixed probability p.
            Threshold: a node becomes infected if at least "threshold" of its neighbors are already infected.
            Share: a node becomes infected if at least "share" fraction of its neighbors are already infected.
    """
    # Checking consistency of parameters
    v_set=set(vaccinated)
    if any(x in v_set  for x in initial):
        print("Infection-seeding nodes should differ from vaccinated nodes.")
        return 
    
    if p<0 or p>1:
        print("Probability of transmission most be in [0,1]")
    if share<0 or share>1:
        print("Share threshold most be in  [0,1]")
    
    # Set up internal variables:
    np.random.seed(random_seed)    
    node_list =  list(G.nodes())
    node_list.sort()
    N = len(node_list)
    A = nx.to_numpy_matrix(G,nodelist=node_list)
    exposed = np.zeros((T,N))
    
    
    for i in initial:
        exposed[0,i] =1
     
    j = 1
    fig=plt.figure(figsize=(12, T*5))
    
    for t in range(T):
        
        if t>0:
            degree= A.sum(axis=0)
            pos_deg = degree==0
            if spreading=='simple':
                U = np.random.uniform(0,1,size=(N,N))
                U_symm = (U + U.T)/2
                E = U_symm>(1-p)
                B = np.multiply(A,E)
                exposed[t,:] = (np.dot(B,exposed[t-1,:]).astype(int)+exposed[t-1,:])>0 
            elif spreading=='threshold':
                exposed[t,:] = (((np.dot(A,exposed[t-1,:]))>=threshold).astype(int)+exposed[t-1,:])>0
            elif spreading=='share':
                exposed[t,:] = ((np.dot(A,exposed[t-1,:])>=(degree*share+pos_deg)).astype(int)+exposed[t-1,:])>0
            for i in vaccinated:
                exposed[t,i] =0
        
        exposed_nodes = [node for node in node_list if exposed[t,node]==1]
        clean_nodes   = [node for node in node_list if exposed[t,node]==0]
        #actual = count_share_sick_dyads(G,exposed[t,:]) 

        ax = fig.add_subplot(T, 2, j)
        clust_data = [['Contagion: {TYPE}'.format(TYPE=spreading)],
                      ['Spreading Probability: {T}'.format(T=p)],
                      ['Threshold: {T}'.format(T=threshold)],
                      ['Time Period: {T}'.format(T=t)],
                      ['Infected nodes: {IN}'.format(IN=len(exposed_nodes))],
                      ['Suceptible nodes: {IN}'.format(IN=len(clean_nodes))],
                     ]
        tab = ax.table(cellText=clust_data,cellLoc='left',loc='center')
        for key, cell in tab.get_celld().items():
            cell.set_linewidth(0)
        tab.set_fontsize(14)
        ax.axis('off')
        
        
        ax = fig.add_subplot(T, 2, j+1)
        #ax.title.set_text('Period: {S}'.format(S=t))
        

        
        
        # drawing nodes and edges separately so we can capture collection for colobar
        pos0 = nx.circular_layout(G)
        ec = nx.draw_networkx_edges(G, pos0, alpha=1)
        nc = nx.draw_networkx_nodes(G, pos0, nodelist=exposed_nodes,node_color=('#d55e00'), 
                            with_labels=True,font_color='black', font_weight='bold', node_size=500)
        nn = nx.draw_networkx_nodes(G, pos0, nodelist=clean_nodes,node_color=('#009e73'), 
                            with_labels=True, font_color='black', font_weight='bold', node_size=500)
        nl = nx.draw_networkx(G, node_color='none',alpha=1,font_color='white', font_weight='bold',
                              pos=pos0,with_labels = True,node_size=500)


        red_patch = mpatches.Patch(color='#d55e00', label='Infected')
        plt.legend(bbox_to_anchor=(1, 1),handles=[red_patch])
        
        
        plt.axis('off')
        ax2= plt.gca()
        ax2.collections[0].set_edgecolor("#000000")


        
        #pd.Series(sim[:,t]).hist(bins=50,cumulative=-1,density=True, histtype='step')
        #plt.axvline(actual, color='r', linestyle='dashed', linewidth=1.5)
        #ax.annotate('p-value: \n{P}'.format(P=(sim[:,t]>=actual).sum()/n_sim), 
        #        xy=(actual+.01, .21),
        #        xytext=(actual+.01, .21),
        #       fontsize=12)
    

        plt.tight_layout()

        
        j = j+2
    plt.show()
    
    
# Financial Networks model
def finance_network_eq(Q,p,S,D1,beta=.1,case='worst',tol=.01,max_iter = 1000):
    
    # Checking dimensions are internally consistent
    N, m = Q.shape
    m1, _ = p.shape
    ns, ms = S.shape
    nd, md = D1.shape
    if m!=m1 or N!=ns or ns!=ms or nd!=md or N!=md:
        print('Dimension of inputed matrices is not consistent with the model.\nPlease check!')
        return
    
    # Start assuming no one is in bankruptcy, then the values would be:
    if case == 'best':
        V1 = np.linalg.inv(np.identity(N)-S)@(Q@p+(D1.sum(axis=1)-D1.sum(axis=0)).reshape(N,1))
    # For worst case asume everyone is in bankruptcy (V<0)
    if case == 'worst': 
        V1 = -1*np.ones((N,1)) 
    
    # compute share of j's debt that is owed to i:
    
    with np.errstate(divide='ignore', invalid='ignore'): #this omitts error  when dividing by zero and substitutrs infinity with zero as well
        sh_D_ij = np.nan_to_num(np.divide(D1,D1.sum(axis=0)))

    # Set V0 to very different level to force loop
    V0 = -99*np.ones((N,1))

    # Initialize values
    insolvent = np.zeros((N,N)) 
    j=0
    #D1 = D.copy()
    # Value function iteration until convergence
    while np.linalg.norm(V0-V1)>tol and j<max_iter:
        V0 = V1.copy()
        
        solvency_status = np.multiply(Q@p+D1.sum(axis=1).reshape(N,1)+S@np.maximum(0,V0),1-beta*(V0<0))
        bankrupt_debt_pay = np.maximum(0,np.multiply(sh_D_ij,solvency_status.T))   
        insolvent = np.maximum(insolvent,(D1 > bankrupt_debt_pay).astype(int))
        D1 = np.minimum(D1,bankrupt_debt_pay)
        # Actual value when accounting for bankruptcies:
        V1 = np.multiply(Q@p+(D1.sum(axis=1)-D1.sum(axis=0).T).reshape(N,1)+S@np.maximum(0,V0),1-beta*(V0<0)) 
        j = j+1
        
    insolvent_institutions = (insolvent.sum(axis=0)>0).T
    
    if j==max_iter:
        print('Algorithm ended after {J} steps without converging with the specified tolerance level: {Tol}'.format(
            J=j,Tol=tol))
        
    return V1, insolvent_institutions
    
# DeGroot's Model
def beliefs_at_time(T,P0,t):
    """
    This function computes the vector of beliefs at time t for a learning network represented
    by trust network T and initial belief vector P0.
    """
    T = np.asarray(T)
    P0 = np.transpose(np.asarray(P0))
    n, m = T.shape
    N = P0.size
    if (n!=m) or (m!=N):
        print("Trust matrix should be squared and number of agents should be consistent in T and P0.")
        return
    return np.linalg.matrix_power(T,t).dot(P0)
     
def time_to_convergence(T,P0,final_beliefs=False,tolerance=0,max_iter=10000):
    """
    This function calculates the number of periods that it takes for opinions to stop changing in the DeGroot Model.
    Optionally, one can also get the final belief profile.
    """
    T = np.asarray(T)
    P0 = np.transpose(np.asarray(P0))
    n, m = T.shape
    N = P0.size
    if (n!=m) or (m!=N):
        print("Trust matrix should be squared and number of agents should be consistent in T and P0.")
        return
    t = 1
    N = P0.size
    P1 = P0
    P0 =  T.dot(P1)
    while (t<max_iter) and (np.linalg.norm(P0-P1)>tolerance):
        P1 = P0
        P0 = T.dot(P1)
        t = t+1

     
    if final_beliefs == True:
        return t, P0
    else:
        return t

# This function is only to generate the nice plots above
# It does not really perform anything substantial in relation to the learning model
   
def plot_trust_network(TT,p): 
    G = nx.convert_matrix.from_numpy_matrix(TT,create_using=nx.MultiDiGraph,
                                             parallel_edges=True)
    plt.figure(figsize=(5,5))    
    pos=nx.circular_layout(G) 
    # draw nodes and labels
    nx.draw_networkx_nodes(G,pos,cmap=plt.get_cmap('viridis'),vmin=0, vmax=1, node_color=p,node_size=300)
   
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
    cmap = plt.cm.get_cmap('viridis')
    colors = cmap(np.arange(cmap.N))
    plt.show()
    plt.figure(figsize=(5,5))    

    plt.imshow([colors], extent=[0, 1, 0, .1])
    plt.yticks([0,.1], " ")
    plt.xlabel("Belief/Opinion (Pi)")
    plt.show()
    
    
#---------------------------------#
# INTERACTIVE INTERFACE FUNCTIONS # 
#---------------------------------#

# The code below creates the interactive features seen in the notebooks
# It mostly consists on creating buttons to allow for interactive parameter inputs
# It uses some of the functions from above although some very specific functions 
# are also coded for particular functionalities of the interface


# Dutch School Interactive:

def dutch_interactive(Period='t=0',Gen=False,Alcohol='None',layout='Kamada Kawai',Attribute=''):
    
    Gender= np.loadtxt('../Data/dutch_hs/classroom_demographics.dat')
    Gender = np.where(Gender==1, 'Female', Gender)
    Gender = np.where(Gender=='2.0', 'Male', Gender)

    m = 8
    ns = 500
    if Period == 't=0':
        G_nodes = nx.from_pandas_edgelist(pd.read_csv('../Data/dutch_hs/dutch_class_t3.csv'))
        G = nx.from_pandas_edgelist(pd.read_csv('../Data/dutch_hs/dutch_class_t0.csv'))
    elif Period == 't=1':
        G_nodes = nx.from_pandas_edgelist(pd.read_csv('../Data/dutch_hs/dutch_class_t3.csv'))
        G = nx.from_pandas_edgelist(pd.read_csv('../Data/dutch_hs/dutch_class_t1.csv'))
    elif Period == 't=2':
        G_nodes = nx.from_pandas_edgelist(pd.read_csv('../Data/dutch_hs/dutch_class_t3.csv'))
        G = nx.from_pandas_edgelist(pd.read_csv('../Data/dutch_hs/dutch_class_t2.csv'))
    elif Period == 't=3':
        G_nodes = nx.from_pandas_edgelist(pd.read_csv('../Data/dutch_hs/dutch_class_t3.csv'))
        G = nx.from_pandas_edgelist(pd.read_csv('../Data/dutch_hs/dutch_class_t3.csv'))
    
    if Gen == 'Reported Gender':
        feml = {}
        for idx, val in enumerate([i for i in Gender]):
            feml[idx] = val
        nx.set_node_attributes(G,feml,name='gender')
        plot_block_graph(G,'gender', Attribute,
                            layout=layout,G_nodes=G_nodes,
                            type_description='Gender')
    else:
         plot_simple_graph(G,Attribute,fig_dim=m,node_base_size=ns,
                             layout=layout,G_nodes=G_nodes)
           
    
dutch_nets = widgets.ToggleButtons(
    options=['t=0', 't=1', 't=2', 't=3'],
    description='Period:',
    disabled=False,
    button_style='', 
    tooltips=['High School Classroom in the Netherlands', 
              'Any Relationship in Indian Village 1 from Diffusion of Microfinance'],
)


dutch_types = widgets.ToggleButtons(
    options=['None','Reported Gender'],
    description='Types',
    disabled=False,
    button_style='', 
    tooltips=['High School Classroom in the Netherlands', 
              'Any Relationship in Indian Village 1 from Diffusion of Microfinance'],
)


dutch_lays = widgets.ToggleButtons(
    options=['Kamada Kawai','Circle','Spring','Random'],
    value='Kamada Kawai',
    description='Layout:',
    disabled=False,
    button_style='', 
    tooltips=['Circular Arrangement', 
              'Algorithm','Spring','Random'],
)

dutch_atts = widgets.Dropdown(
    options=['','Degree Centrality','Clustering Centrality',
                         'Betweenness Centrality','Eigenvector Centrality'], #,'Diffusion Centrality'],
    value='',
    description='Attribute:',
    disabled=False,
    tooltips=['Circular Arrangement', 
              'Algorithm','Spring','Random'],
)
 

def dutch_update_attributes(*args):
    if dutch_types.value=='None':
        dutch_atts.options=['','Degree Centrality','Clustering Centrality',
                         'Betweenness Centrality','Eigenvector Centrality'] #,'Diffusion Centrality']
        dutch_atts.value = ''
    else:
        dutch_atts.options=['','Degree Centrality','Clustering Centrality',
                         'Betweenness Centrality','Eigenvector Centrality'] #,'Diffusion Centrality',
              #'Relative Homophily','Inbreeding Homophily']

dutch_types.observe(dutch_update_attributes)

dutch_y= widgets.interactive_output(dutch_interactive, {'Period': dutch_nets, 'Gen': dutch_types,'Attribute':dutch_atts,
                                                     'layout':dutch_lays})

                                                        
                                                        
                                                        
# Network Comparison plots

def hist_attribute_distribution(vals1,vals2,title,network1,network2,ax):    
    n_bins = len(set(vals1).union(set(vals2)))
    pd.Series(vals1).hist(bins=n_bins,cumulative=1,density=True,color= '#d55e00', histtype='step',label=network1, linewidth=3)
    pd.Series(vals2).hist(bins=n_bins,cumulative=1,density=True,color= '#0072b2', histtype='step',label=network2, linewidth=3)
    pl.title(title,fontsize=20)
    
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width*0.9, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5,fontsize=12)
    
    
def compare_net_plot(G1,G2,network1,network2,layout):    
    # Create 3x2 sub plots
    gs = gridspec.GridSpec(3, 2, height_ratios=[3, 1, 1])

    pl.figure(figsize=(16,16))
    ax = pl.subplot(gs[0, 0]) # row 0, col 0
    
    pos =layout_selection(layout,G1)

    node_list = list(G1.nodes())
    node_list.sort()
    ec = nx.draw_networkx_edges(G1, pos, alpha=1)
    nc = nx.draw_networkx_nodes(G1,  node_color='#d55e00',nodelist=node_list,alpha=1,font_color='white', 
                                font_weight='bold',pos=pos,with_labels = True,node_size=500)
    nl = nx.draw_networkx_labels(G1,font_color='white',nodelist=node_list, font_weight='bold',
                              pos=pos,with_labels = True,node_size=500)    
    pl.axis('off')
    pl.title(network1,fontsize=20)


    ax = pl.subplot(gs[0, 1]) # row 0, col 1

    pos =layout_selection(layout,G2)
        
    node_list = list(G2.nodes())
    node_list.sort()
    ec = nx.draw_networkx_edges(G2, pos, alpha=1)
    nc = nx.draw_networkx_nodes(G2,  node_color='#0072b2',nodelist=node_list,alpha=1,font_color='white', 
                                font_weight='bold',pos=pos,with_labels = True,node_size=500)
    nl = nx.draw_networkx_labels(G2,font_color='white',nodelist=node_list, font_weight='bold',
                              pos=pos,with_labels = True,node_size=500)    
    pl.axis('off')
    pl.title(network2,fontsize=20)

    ax = pl.subplot(gs[1, 0])
    # Transform values from dictionary into a list of numbers.
    vals1 = list(dict(G1.degree).values())
    vals2 = list(dict(G2.degree).values())
    
    hist_attribute_distribution(vals1,vals2,'Cumulative Degree Distribution',network1,network2,ax)
    
    ax = pl.subplot(gs[1, 1])
    vals1 = list(nx.eigenvector_centrality_numpy(G1).values())
    vals2 = list(nx.eigenvector_centrality_numpy(G2).values())
     
    hist_attribute_distribution(vals1,vals2,'Cumulative Clustering Centrality Distribution',network1,network2,ax)
    
    ax = pl.subplot(gs[2, 0])
    vals1 = list(nx.betweenness_centrality(G1).values())
    vals2 = list(nx.betweenness_centrality(G2).values())
    
    hist_attribute_distribution(vals1,vals2,'Cumulative Betweenness Centrality Distribution',network1,network2,ax)

    ax = pl.subplot(gs[2, 1])
    vals = list(diffusion_centrality(G1).values())
    vals1 = [float(a) for a in vals]
    vals2 = list(diffusion_centrality(G2).values())
    vals2 = [float(a) for a in vals2]
    
    hist_attribute_distribution(vals1,vals2,'Cumulative Eigenvector Centrality Distribution',network1,network2,ax)
   
    pl.tight_layout()

    pl.subplots_adjust(hspace=.2) 
    pl.show()

    
                      
def compare_interactive(network1,network2,N1=10,N2=10,p1=.1,p2=.1,
         m1=2,m2=2,path1='',path2='',layout=''):
    G1 =  network_selector(network1,N1,p1,m1,path1)
    G2 =  network_selector(network2,N2,p2,m2,path2)
    compare_net_plot(G1,G2,network1,network2,layout)
   

style = {'description_width': 'initial'}


comp_N1=widgets.BoundedIntText(
    value=10,
    min=1,
    max=100,
    step=1,
    description='# Nodes:',style=style,
    disabled=False
)
comp_p1 = widgets.FloatSlider(min=0,max=1,step=.01,
                        value=.3,
                     description='Probability of Edge:', style=style,
                        continuous_update=False,
                        disabled=False)

comp_m1=widgets.BoundedIntText(
    value=0,
    min=0,
    max=comp_N1.value-1,
    step=1,
    description='',style=style,
    disabled=True
)

comp_path1 = widgets.Text(
    placeholder='Path to network file',
    description='File Path:', style=style, 
    disabled=False
)

comp_N2=widgets.BoundedIntText(
    value=26,
    min=1,
    max=100,
    step=1,
    description='Nodes:',
    disabled=False
)
comp_p2 = widgets.FloatSlider(min=0,max=1,step=.01,
                        value=.3,
                     description='Probability:',style=style,
                        continuous_update=False)

comp_m2=widgets.BoundedIntText(
    value=1,
    min=1,
    max=comp_N2.value-1,
    step=1,
    description='Nodes:',
    disabled=False
)

comp_path2 = widgets.Text(
    placeholder='Path to network file',
    description='File Path:',
    disabled=False
)

Network1 = widgets.ToggleButtons(
    options=['Dutch High School','Erdos-Renyi','Barabasi Albert','Powerlaw Cluster','Watts-Strogatz',
            'Newman-Watts-Strogatz','Wheel Network','Regular Network','From File'],
    description='Network Type:', style= style,
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Circular Arrangement', 
              'Algorithm','Spring','Random'],
#     icons=['check'] * 3
)

Network2 = widgets.ToggleButtons(
    options=['Erdos-Renyi','Barabasi Albert','Powerlaw Cluster','Watts-Strogatz',
            'Newman-Watts-Strogatz','Wheel Network','Regular Network','Dutch High School','From File'],
    description='Network2:',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Circular Arrangement', 
              'Algorithm','Spring','Random'],
#     icons=['check'] * 3
)

comp_lays = widgets.ToggleButtons(
    options=['Circle','Kamada Kawai','Spring','Random'],
    value='Circle',
    description='Layout:',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Circular Arrangement', 
              'Algorithm','Spring','Random'],
#     icons=['check'] * 3
)

nets1 = widgets.VBox([Network1,widgets.HBox([comp_N1,comp_p1,comp_m1]),comp_path1])
nets2 = widgets.VBox([Network2,widgets.HBox([comp_N2,comp_p2,comp_m2]),comp_path2])

comp_accordion = widgets.Accordion(children=[nets1,nets2])
comp_accordion.set_title(0,  Network1.value)
comp_accordion.set_title(1,  Network2.value)

    
comp_y= widgets.interactive_output(compare_interactive, {'network1': Network1, 'network2': Network2,
                                     'N1':comp_N1,'p1':comp_p1,'m1':comp_m1,
                                     'N2':comp_N2,'p2':comp_p2,'m2':comp_m2,
                                     'path1':comp_path1,'path2':comp_path2,'layout':comp_lays})


def comp_update1(*args):
    comp_accordion.set_title(0, Network1.value)
    if Network1.value=='Erdos-Renyi':
        comp_N1.disabled=False
        comp_m1.disabled=True
        comp_p1.disabled=False
        comp_p1.min=0
        comp_p1.description='Probability of Edge'
        comp_m1.description=''
        comp_N1.value = comp_N1.value
        comp_N1.value = comp_N1.value
    elif Network1.value=='Barabasi Albert':
        comp_m1.disabled=False
        comp_p1.disabled=True
        comp_p1.description=''
        comp_N1.disabled=False
        comp_m1.description='Degree of New Node'
        comp_p1.value=0
        comp_m1.min=1
        comp_N1.value = comp_N1.value
    elif Network1.value=='Powerlaw Cluster':
        comp_m1.description='# Random Edges'
        comp_m1.min=2
        comp_p1.description='Prob. Adding Triangles'
        comp_p1.min=.01
        comp_p1.disabled=False
        comp_N1.disabled=False
        comp_m1.disabled=False
        comp_N1.value = comp_N1.value
    elif Network1.value=='Watts-Strogatz':
        comp_m1.min=2
        comp_m1.description='Degree'
        comp_p1.description='Prob. Rewiring'
        comp_p1.min=0
        comp_N1.disabled=False
        comp_p1.disabled=False
        comp_m1.disabled=False
        comp_N1.value = comp_N1.value
    elif Network1.value=='Newman-Watts-Strogatz':
        comp_m1.description='Degree'
        comp_p1.description='Prob. Adding New Edge'
        comp_N1.disabled=False
        comp_m1.disabled=False
        comp_p1.min=0
        comp_m1.min=1
        comp_N1.value = comp_N1.value
    elif Network1.value=='Wheel Network':
        comp_m1.description=''
        comp_p1.description=''
        comp_p1.value=0
        comp_p1.disabled=True
        comp_N1.disabled=False
        comp_m1.disabled=True
        comp_m1.min=0
        comp_m1.value=0
        comp_N1.value = comp_N1.value
    elif Network1.value=='Regular Network':
        comp_m1.description='Degree'
        comp_p1.description=''
        comp_N1.disabled=False
        comp_p1.value=0
        comp_p1.disabled=True
        comp_m1.disabled=False
        comp_m1.min=0
        comp_N1.value = comp_N1.value+1 if (comp_N1.value % 2) != 0 and m1.value==1 else comp_N1.value
    elif Network1.value=='Dutch High School':
        comp_m1.description=''
        comp_p1.description=''
        comp_p1.value=0
        comp_m1.value=0
        comp_p1.disabled=True
        comp_m1.disabled=True
        comp_N1.disabled=True
        comp_m1.min=0
        comp_N1.value = 26


def comp_update2(*args):
    comp_accordion.set_title(1, Network2.value)
    if Network2.value=='Erdos-Renyi':
        comp_m2.disabled=True
        comp_p2.disabled=False
        comp_p2.min=0
        comp_p2.description='Probability of Edge'
        comp_m2.description=''
        comp_N2.value = comp_N2.value
    elif Network2.value=='Barabasi Albert':
        comp_m2.disabled=False
        comp_p2.disabled=True
        comp_p2.description=''
        comp_m2.description='Degree of New Node'
        comp_p2.value=0
        comp_m2.min=1
        comp_N2.value = comp_N2.value
    elif Network2.value=='Powerlaw Cluster':
        comp_m2.description='# Random Edges'
        comp_m2.min=2
        comp_p2.description='Prob. Adding Triangles'
        comp_p2.min=.01
        comp_p2.disabled=False
        comp_m2.disabled=False
        comp_N2.value = comp_N2.value
    elif Network2.value=='Watts-Strogatz':
        comp_m2.min=2
        comp_m2.description='Degree'
        comp_p2.description='Prob. Rewiring'
        comp_p2.min=0
        comp_p2.disabled=False
        comp_m2.disabled=False
        comp_N2.value = comp_N2.value
    elif Network2.value=='Newman-Watts-Strogatz':
        comp_m2.description='Degree'
        comp_p2.description='Prob. Adding New Edge'
        comp_m2.disabled=False
        comp_p1.min=0
        comp_m2.min=1
        comp_N2.value = comp_N2.value
    elif Network2.value=='Wheel Network':
        comp_m2.description=''
        comp_p2.description=''
        comp_p2.value=0
        comp_p2.disabled=True
        comp_m2.disabled=True
        comp_m2.min=0
        comp_m2.value=0
        comp_N2.value = comp_N2.value
    elif Network2.value=='Regular Network':
        comp_m2.description='Degree'
        comp_p2.description=''
        comp_p2.value=0
        comp_p2.disabled=True
        comp_m2.disabled=False
        comp_m2.min=0
        comp_N2.value = comp_N2.value+1 if (comp_N2.value % 2) != 0 and m2.value==1 else comp_N2.value
    elif Network2.value=='Dutch High School':
        comp_m2.description=''
        comp_p2.description=''
        comp_p2.value=0
        comp_m2.value=0
        comp_p2.disabled=True
        comp_m2.disabled=True
        comp_N2.disabled=True
        comp_m2.min=0
        comp_N2.value = 26

Network1.observe(comp_update1)
comp_m1.observe(comp_update1)
comp_N1.observe(comp_update1)

Network2.observe(comp_update2)
comp_m2.observe(comp_update2)
comp_N2.observe(comp_update2)        
        
        
        
# Random Homophily interactive
N_h=widgets.BoundedIntText(
    value=10,
    min=1,
    max=100,
    step=1,
    description='Nodes:',
    disabled=False
)

p_h = widgets.FloatSlider(min=0,max=1,step=.01,
                        value=.3,
                     description='Probability:',
                        continuous_update=False)
atts_h = widgets.Dropdown(
    options=['','Degree Centrality','Clustering Centrality',
             'Betweenness Centrality','Eigenvector Centrality',
             'Diffusion Centrality','Relative Homophily','Inbreeding Homophily'],
    value='',
    description='Attribute:',
)

lays_h = widgets.ToggleButtons(
    options=['Spring','Circle','Kamada Kawai','Random'],
    description='Layout:',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Circular Arrangement', 
              'Algorithm','Spring','Random'],
#     icons=['check'] * 3
)


types_h=widgets.ToggleButton(
    value=False,
    description='Have Two Types',
    disabled=False,
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Description',
)

# Homophily paramters

n1_h = widgets.IntSlider(min=0,max=N_h.value,step=1,
                      value=np.ceil(N_h.value/2),
                     description=r'\(\color{#d55e00} {N0}\)',
                      continuous_update=False)
n2_h = widgets.IntSlider(min=0,max=N_h.value,step=1,
                      value=np.floor(N_h.value/2),
                     description='\(\color{#0072b2} {N1}\)',
                      continuous_update=False)

p1_h = widgets.FloatSlider(min=0,max=1,step=.01,
                        value=.1,
                     description='P0:',
                        continuous_update=False)
p2_h = widgets.FloatSlider(min=0,max=1,step=.01,
                        value=.1,
                     description='P1:',
                        continuous_update=False)

px_h = widgets.FloatSlider(min=0,max=1,step=.01,
                        value=0,
                     description='X-Type prob:',
                        continuous_update=False)


prob_ratio1 = widgets.widgets.FloatText(
    value=p1_h.value/px_h.value if px_h.value != 0 else np.nan,
    description='P0/Px:',continuous_update=True,
    disabled=True
)
prob_ratio2 = widgets.widgets.FloatText(
    value=p1_h.value/px_h.value if px_h.value != 0 else np.nan,
    description='P1/Px:',continuous_update=True,
    disabled=True
)




homophily1 = widgets.HBox([types_h,n1_h,n2_h])
homophily2 = widgets.HBox([p1_h,p2_h,px_h])
homophily_h = widgets.VBox([homophily1,homophily2,widgets.HBox([prob_ratio1,prob_ratio2])])

main_h = widgets.HBox([N_h, p_h])
accordion_h = widgets.Accordion(children=[main_h])
accordion_h.set_title(0, 'Global Parameters For Random Network')

accordion_h2 = widgets.Accordion(children=[homophily_h],selected_index=None)
accordion_h2.set_title(0, 'Homophily')

# Block random graph requires parameters for 8 groups
P_h = [[p_h.value,px_h.value,0,0,0,0,0,0],[px_h.value,p1_h.value,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]

y_h= widgets.interactive_output(plot_random_graph, {'n': N_h, 'p': p_h,'Attribute':atts_h,
                                                     'layout':lays_h,'types':types_h,'n1':n1_h,'p1':p1_h,'p2':p2_h,'px':px_h})

# Define a function that updates the content of y based on what we select for x
def update(*args):
    prob_ratio1.value = p1_h.value/px_h.value if px_h.value != 0 else np.nan
    prob_ratio2.value = p2_h.value/px_h.value  if px_h.value != 0 else np.nan
    n1_h.max = N_h.value
    n2_h.max = N_h.value
    N_h.value = n1_h.value + n2_h.value
    p_h.value = p_h.value
    types_h.value = types_h.value 

def update_n1_h(*args):
    n1_h.value = N_h.value-n2_h.value
    
def update_n2_h(*args):
    n2_h.value = N_h.value-n1_h.value
p1_h.observe(update)
p2_h.observe(update)
px_h.observe(update)
N_h.observe(update)

n1_h.observe(update_n2_h)
n2_h.observe(update_n1_h)

types_h.observe(update)
  
    
# Island Model Interactive
n1_island=widgets.BoundedIntText(
    value=5,
    min=1,
    placeholder='N1',
    description='N1:',
    disabled=False
)

n2_island=widgets.BoundedIntText(
    value=5,
    min=1,
    placeholder='N1',
    description='N2:',
    disabled=False
)


n3_island=widgets.BoundedIntText(
    value=5,
    min=0,
    placeholder='N1',
    description='N3:',
    disabled=False
)

n4_island=widgets.BoundedIntText(
    value=0,
    placeholder='N1',
    min=0,
    description='N4:',
    disabled=False
)

n5_island=widgets.BoundedIntText(
    value=0,
    min=0,
    placeholder='N1',
    description='N5:',
    disabled=False
)

n6_island=widgets.BoundedIntText(
    value=0,
    min=0,
    placeholder='N1',
    description='N6:',
    disabled=False
)

n7_island=widgets.BoundedIntText(
    value=0,
    min=0,
    placeholder='N1',
    description='N7:',
    disabled=False
)

n8_island=widgets.BoundedIntText(
    value=0,
    min=0,
    placeholder='N1',
    description='N8:',
    disabled=False
)
top_island = widgets.HBox([n1_island, n2_island, n3_island,n4_island])
bot_island = widgets.HBox([n5_island,n6_island,n7_island,n8_island])


lays_island = widgets.ToggleButtons(
    options=['Circle','Kamada Kawai','Spring','Random'],
    description='Layout:',
    disabled=False,
    button_style='', 
    tooltips=['Circular Arrangement', 
              'Algorithm','Spring','Random'],
)

atts_island = widgets.Dropdown(
    options=['','Degree Centrality','Clustering Centrality',
                         'Betweenness Centrality','Eigenvector Centrality','Diffusion Centrality',
             'Relative Homophily','Inbreeding Homophily'],
    value='',
    description='Attribute:',
    disabled=False,
    tooltips=['Circular Arrangement', 
              'Algorithm','Spring','Random'],
)

def interactive_islands(n1,n2,n3,n4,n5, n6,n7,n8,layout,Attribute):
    
    G, type_id = simple_random_block_graph(n1,n2,n3,n4,n5, n6,n7,n8)

    plot_block_graph(G, type_id,Attribute,layout)



y_island= widgets.interactive_output(interactive_islands, {'n1': n1_island, 'n2': n2_island, 'n3': n3_island,'n4': n4_island,
                                                 'n5': n5_island,'n6': n6_island,'n7': n7_island,'n8': n8_island,
                                                'layout':lays_island,'Attribute':atts_island})


# Peer Effcts, dutch school drinking


def dutch_drink_interact(Period='t=0',Gen=False,Alcohol='None',layout='Kamada Kawai',Attribute=''):
    Alc = np.loadtxt('../Data/dutch_hs/classroom_alcohol.dat')
    Alc = np.where(Alc==1, '1) Never', Alc)
    Alc = np.where(Alc=='2.0', '2) Once', Alc)
    Alc = np.where(Alc=='3.0', '3) 2-4', Alc)
    Alc = np.where(Alc=='4.0', '4) 5-10', Alc)
    Alc = np.where(Alc=='5.0', '5) >10', Alc)
    
    Gender= np.loadtxt('../Data/dutch_hs/classroom_demographics.dat')
    Gender = np.where(Gender==1, 'Female', Gender)
    Gender = np.where(Gender=='2.0', 'Male', Gender)


    drink_1 = {}
    drink_2 = {}
    drink_3 = {}
    m = 8
    ns = 500
    if Period == 't=0':
        G_nodes = nx.from_pandas_edgelist(pd.read_csv('../Data/dutch_hs/dutch_class_t3.csv'))
        G = nx.from_pandas_edgelist(pd.read_csv('../Data/dutch_hs/dutch_class_t0.csv'))
    elif Period == 't=1':
        G_nodes = nx.from_pandas_edgelist(pd.read_csv('../Data/dutch_hs/dutch_class_t3.csv'))
        G = nx.from_pandas_edgelist(pd.read_csv('../Data/dutch_hs/dutch_class_t1.csv'))
    elif Period == 't=2':
        G_nodes = nx.from_pandas_edgelist(pd.read_csv('../Data/dutch_hs/dutch_class_t3.csv'))
        G = nx.from_pandas_edgelist(pd.read_csv('../Data/dutch_hs/dutch_class_t2.csv'))
    elif Period == 't=3':
        G_nodes = nx.from_pandas_edgelist(pd.read_csv('../Data/dutch_hs/dutch_class_t3.csv'))
        G = nx.from_pandas_edgelist(pd.read_csv('../Data/dutch_hs/dutch_class_t3.csv'))
    
    if Gen == 'Reported Gender':
        feml = {}
        for idx, val in enumerate([i for i in Gender]):
            feml[idx] = val
        nx.set_node_attributes(G,feml,name='gender')
        plot_block_graph(G,'gender', Attribute,
                            layout=layout,G_nodes=G_nodes,
                            type_description='Gender')

    else:
        if Alcohol == "No Info":
            plot_simple_graph(G,Attribute,fig_dim=m,node_base_size=ns,
                             layout=layout,G_nodes=G_nodes)
        elif Alcohol =="t=1":
            for idx, drink in enumerate([i[0] for i in Alc]):
                drink_1[idx] = drink  
            nx.set_node_attributes(G,drink_1,name='drink_1')
            plot_block_graph(G,'drink_1', Attribute,
                            layout=layout,G_nodes=G_nodes,
                            type_description='How often did you drink with friends in the last 3 months?')
        elif Alcohol =="t=2":
            for idx, drink in enumerate([i[1] for i in Alc]):
                drink_2[idx] = drink
            nx.set_node_attributes(G,drink_2,name='drink_2')
            plot_block_graph(G,'drink_2', Attribute,
                            layout=layout,G_nodes=G_nodes,
                            type_description='How often did you drink with friends in the last 3 months?')
        elif Alcohol =="t=3":
            for idx, drink in enumerate([i[2] for i in Alc]):
                drink_3[idx] = drink
            nx.set_node_attributes(G,drink_3,name='drink_3')
            plot_block_graph(G,'drink_3', Attribute,
                            layout=layout,G_nodes=G_nodes,
                            type_description='How often did you drink with friends in the last 3 months?')
        
    
        
    
nets_drink = widgets.ToggleButtons(
    options=['t=0', 't=1', 't=2', 't=3'],
    description='Period:',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['High School Classroom in the Netherlands', 
              'Any Relationship in Indian Village 1 from Diffusion of Microfinance'],
#     icons=['check'] * 3
)


types_drink = widgets.ToggleButtons(
    options=['None','Reported Gender'],
    description='Types',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['High School Classroom in the Netherlands', 
              'Any Relationship in Indian Village 1 from Diffusion of Microfinance'],
#     icons=['check'] * 3
)


drinking = widgets.ToggleButtons(
    options=['No Info','t=1', 't=2', 't=3'],
    description='Alcohol Use:',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['High School Classroom in the Netherlands', 
              'Any Relationship in Indian Village 1 from Diffusion of Microfinance'],
#     icons=['check'] * 3
)

lays_drink = widgets.ToggleButtons(
    options=['Kamada Kawai','Circle','Spring','Random'],
    value='Kamada Kawai',
    description='Layout:',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Circular Arrangement', 
              'Algorithm','Spring','Random'],
#     icons=['check'] * 3
)

atts_drink = widgets.Dropdown(
    options=['','Degree Centrality','Clustering Centrality',
                         'Betweenness Centrality','Eigenvector Centrality','Diffusion Centrality'],
    value='',
    description='Attribute:',
    disabled=False,
    tooltips=['Circular Arrangement', 
              'Algorithm','Spring','Random'],
#     icons=['check'] * 3
)
  
accordion_drink = widgets.Accordion(children=[drinking],selected_index=None)
accordion_drink.set_title(0, 'Behavior')

y_drink= widgets.interactive_output(dutch_drink_interact, {'Period': nets_drink, 'Gen': types_drink,'Alcohol':drinking,
                                           'Attribute':atts_drink,'layout':lays_drink})

# Define a function that updates the content of y based on what we select for x
def update_drink(*args):
    if types_drink.value=='Reported Gender':
        drinking.disabled=True
        drinking.value='No Info'
    else:
        drinking.disabled=False


def update_attributes_drink(*args):
    if drinking.value=='No Info' and types_drink.value=='None':
        atts_drink.options=['','Degree Centrality','Clustering Centrality',
                         'Betweenness Centrality','Eigenvector Centrality','Diffusion Centrality']
        atts_drink.value = ''
    else:
        atts_drink.options=['','Degree Centrality','Clustering Centrality',
                         'Betweenness Centrality','Eigenvector Centrality','Diffusion Centrality',
             'Relative Homophily','Inbreeding Homophily']


        
types_drink.observe(update_drink)
types_drink.observe(update_attributes_drink)
drinking.observe(update_attributes_drink)


def real_flu_contagion_with_pval():
    """ This functions loads, cleans and plots flu contagion data form the Reality commons project
        it also estimates the p-value for the share of dyads in which both nodes are sick from 
        all dyads with at least one sick node. It does so by randomly simulating sickness such that the number
        of sick nodes each week corresponds to the ones observed in the actual data.
    """
    # Load and clean flu data/network
    flu_nets = pd.read_csv('../Data/Reality_Commons/RelationshipsFromSurveys.csv')
    
    flu_nets['year']= flu_nets['survey.date'].str[:4]
    #print(pd.crosstab(flu_nets['relationship'],flu_nets['survey.date']))
    
    # Keep only 2008 networks
    flu_nets=flu_nets[flu_nets.year == '2008']
    flu_nets=flu_nets[flu_nets.relationship == 'SocializeTwicePerWeek']
    
    #C = pd.read_csv('Reality_Commons/Proximity.csv')
    #flu_nets = C[C['prob2']>.3] #.3
    
    # Prepare 2009 flu data
    flu = pd.read_csv('../Data/Reality_Commons/FluSymptoms.csv')
    flu['date_column'] = pd.to_datetime(flu['time'])
    flu['year'] = flu.date_column.dt.to_period('Y').astype('str')
    flu['week'] = flu.date_column.dt.to_period('w')
    flu= flu[flu.year=='2009']
    
    node_list = set(flu['user_id'].values)
    
    # Collapse the data to a weekly basis (take the max because we only care
    # about user experiencing symptoms at some point in the week)
    flu = flu.groupby(['user_id','week']).max()
    flu.reset_index(inplace=True)
    names = ['user_id','week']
    
    # expand range to account for isolated nodes and non-responding nodes
    all_user_ids = range(1,85)
    all_weeks = flu['week'].unique()
    
    # complete the panel, assuming no symptoms when nothing is reported
    mind = pd.MultiIndex.from_product(
        [all_user_ids, all_weeks], names=names)
    
    flu.set_index(names).reindex(mind, fill_value=0).reset_index()
    
    
    flu.set_index('user_id',inplace=True)
    
    # Lenient definition for sick: if it has at least two physical symptoms in one week (one could change that)
    flu['sum']=flu[['sore.throat.cough','runnynose.congestion.sneezing','fever', 'nausea.vomiting.diarrhea']].sum(axis=1)
    flu['sick']=flu['sum']>1
    
    
    # Create network 
    G00=nx.from_pandas_edgelist(flu_nets,source='id.A',target= 'id.B')
    G00 = G00.to_undirected()
    G00.remove_edges_from(G00.selfloop_edges()) 
    isolates = node_list-set(G00.nodes())
    for ndx in isolates:
        G00.add_node(ndx)
        
    # Add flu data
    for t in flu['week'].unique():
        for node in G00.nodes(): 
            try:
                G00.nodes[node]['{W}'.format(W=t)] = flu[flu.week==t].loc[node,'sick']
            except:
                continue
    
    # We want to know how more likely is for sick users to have sick friends than non-sick friends
    
    # Make sure that edges appear in both directions on the edge list
    edges = nx.to_pandas_edgelist(G00)
    edges['rid1']=edges[['source','target']].astype(int).min(axis=1)
    edges['rid2']=edges[['source','target']].astype(int).max(axis=1)
    
    edges = edges[['rid1','rid2']]
    edges.columns = ['source','target']
    edges_aux = edges.copy()
    edges_aux.columns = ['target','source']
    
    edges =edges.append(edges_aux, ignore_index = True, sort=False) 
    
    # construct data table with sickness status of both nodes for every pair of friends and every week
    flu['id']=flu.index.values
    
    df2=flu[['id','week','sick']].set_index(['id','week']).unstack(level=1)
    df2.columns = pd.Index(df2.columns)
    
    aux = pd.merge(df2,edges , left_on='id', right_on='source', how='right')
    df_all = pd.merge(df2, aux, left_on='id', right_on='target', how='right')
    
    # one column per week, node and then node ids
    all_names = ['1_x', '2_x','3_x','4_x',
                 '5_x', '6_x','7_x','8_x',
                 '9_x', '10_x','11_x','12_x',
                 '13_x', '14_x','15_x','16_x',
                 '1_y','2_y','3_y','4_y',
                 '5_y','6_y','7_y','8_y',
                 '9_y','10_y','11_y','12_y',
                 '13_y', '14_y','15_y','16_y',
     'source',
     'target']
    
    df_all=df_all.fillna(False)
    
    
    df_all.columns=all_names
    
    
    df_all=df_all.astype(int)
    
    actual = pd.DataFrame()
    for t in list(range(1,17)):
        both_m1 = 0
        both_m2 = 0
        both_p1 = 0
        both_p2 = 0
        df_all=df_all[df_all['source']<df_all['target']]
                         
                         
        both = len(df_all[(df_all['{W}_x'.format(W=t)]==True) & (df_all['{W}_y'.format(W=t)]==True)])
        left = len(df_all[(df_all['{W}_x'.format(W=t)]==True)])
        right = len(df_all[(df_all['{W}_y'.format(W=t)]==True)])
                         
        if left + right - both>0:
            actual.loc[t,'st8']=(both+both_m1+both_m2+both_p1+both_p2)/(left + right - both)  
    
    
    df2 = df2.reset_index()
    actual = actual.transpose()
    
    
    flu['people'] = 1
    stats = flu[['week','sick','people']].groupby(['week']).sum()
    stats['flu_rate']=stats.sick/84
    stats = stats.reset_index()
    
    #Simulations
    
    def shuffle(df, n=1, axis=0):
        df = df.copy()
        for _ in range(n):
            df.apply(np.random.shuffle, axis=axis)
        return df
    
    column_names = list(range(1,17))
    df2.columns=['id']+column_names
    
    
    np.random.seed(seed=22457)
    
    
    simulation = pd.DataFrame()
    print('Running simulation .', end =" ")  
    for n in range(500):
        print('.', end =" ")  
        df2_sim = pd.DataFrame()
        df2_sim['id'] = df2['id']
        
        for t in list(range(1,17)):
            N = 65
            K = int(stats.loc[t-1,'sick'])
            arr = np.array([1] * K + [0] * (N-K))
            np.random.shuffle(arr)
            df2_sim.loc[:,t]=arr
    
        df2_sim.rename(columns={'id':'source'}, inplace=True)
        aux = pd.merge(df2_sim, edges, left_on='source', right_on='source', how='right')
        df2_sim.rename(columns={'source':'target'}, inplace=True)
        df_all_sim = pd.merge(df2_sim, aux, left_on='target', right_on='target', how='right')
        
        
    
        df_all_sim=df_all_sim[df_all_sim['source']<df_all_sim['target']]
        df_all_sim=df_all_sim.fillna(False).astype(int)
    
        for t in list(range(1,16)):                     
            both = len(df_all_sim[(df_all_sim['{W}_x'.format(W=t)]==True) & (df_all_sim['{W}_y'.format(W=t)]==True)])
            left = len(df_all_sim[(df_all_sim['{W}_x'.format(W=t)]==True)])
            right = len(df_all_sim[(df_all_sim['{W}_y'.format(W=t)]==True)])
                         
            if left + right - both>0:
                simulation.loc[t,'sim_{N}'.format(N=n)]=(both)/(left + right - both)  
    
    sims = simulation.transpose()       
    sims.fillna(0, inplace=True)    
       
     
    t = 1
    for wi in flu['week'].unique()[:-2]:
        print(wi)
    
        val_map= nx.get_node_attributes(G00,'{week}'.format(week=wi))
    
        values = [val_map.get(node, 0) for node in G00.nodes()]
    
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        
        pos = nx.kamada_kawai_layout(G00)
        ec = nx.draw_networkx_edges(G00, pos, alpha=.7)
           
        # Assign different color to each group and plot each partition separately:
        partition = nx.get_node_attributes(G00,'{week}'.format(week=wi))
    
        # This is a colorblind friendly palette
        color_list = [ '#d55e00', '#0072b2'] 
    
        node_list = list(G00.nodes())
        color_idx=0
        node_sublist = [node for node in node_list if (node not in partition or partition[node]==True)]
        nc = nx.draw_networkx_nodes(G00, pos, nodelist=node_sublist, node_color=color_list[color_idx], 
                            font_color='black', font_weight='bold', node_size=100)
        color_idx = color_idx+1
    
        node_sublist = [node for node in node_list if (node not in partition or partition[node]==False)]
        nc = nx.draw_networkx_nodes(G00, pos, nodelist=node_sublist, node_color=color_list[color_idx], 
                            font_color='black', font_weight='bold', node_size=100)
        # Add node labels in  white font on top of all.
        #nl = nx.draw_networkx(G00, node_color='none',alpha=1,font_color='white', font_weight='bold',
        #                         pos=pos,with_labels = True,node_size=100)
        
        # Create color-coded legend 
        ax = plt.gca()
        
        for color, label in zip(color_list,['Sick','Non-sick']):
            ax.plot([0],[0],
                    color=color,
                    label=label)
                   
        plt.axis('off')
        # Shrink axis's height by 10% on the bottom to fit the legend below
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width*0.9, box.height * 0.9])
    
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5,title='Flu Symptoms')
        
            
    
        plt.subplot(1, 2, 2)
        sims.loc[:,t].hist(bins=50,cumulative=-1,density=True, histtype='step')
        plt.axvline(actual.loc['st8',t], color='r', linestyle='dashed', linewidth=1.5)
        t = t+1
        plt.tight_layout()
    
        plt.show() 
    
