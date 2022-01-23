import pandas as pd
def get_subgraphs(df1,source='gene1 id',target='gene2 id'):
    import networkx as nx
    g=nx.from_pandas_edgelist(df1,source=source,target=target)
    ug = g.to_undirected()
    sgs = nx.connected_components(ug)
    dn2df={}
    for sg in sgs:
        ns=sorted(sg)
        dn2df['--'.join(ns)]=pd.Series(ns)
    return pd.concat(dn2df,names=['subnetwork name']).reset_index().drop(['level_1'],axis=1).rename(columns={0:'node name'})
