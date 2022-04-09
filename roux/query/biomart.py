from roux.lib.io import *
from roux.query.ensembl import release2prefix

def get_ensembl_dataset_name(x: str) -> str:
    """Get the name of the Ensembl dataset.

    Args:
        x (str): species name.

    Returns:
        str: output.
    """
    l=x.lower().split(' ')
    assert(len(l)==2)
    return f"{l[0][0]}{l[1]}_gene_ensembl"

def query(
          species: str,
          release: int,
          attributes: list=None,
          filters: list=None,
          databasep: str='data/database',
          dataset_name: str=None,
          force: bool=False,
          **kws_query,) -> pd.DataFrame:
    """Query the biomart database.

    Args:
        species (str): species name.
        release (int): Ensembl release. 
        attributes (list, optional): list of attributes. Defaults to None.
        filters (list, optional): list of filters. Defaults to None.
        databasep (str, optional): path to the local database folder. Defaults to 'data/database'.
        dataset_name (str, optional): dataset name. Defaults to None.
        force (bool, optional): overwrite output. Defaults to False.

    Returns:
        pd.DataFrame: output

    Examples: 
        1. Setting filters for the human data:
            filters={
            # REMOVE: mitochondria/sex chr
                'chromosome_name':[str(i) for i in list(range(1,23))],
            # REMOVE: non protein coding
                'biotype':['protein_coding'],
                }
 
    """
    if dataset_name is None:
        dataset_name=get_ensembl_dataset_name(species)
    from pybiomart import Server,Dataset
    serverp=f"http://{release2prefix[release]}.ensembl.org"       
    server = Server(host=serverp)
    assert release==int(server['ENSEMBL_MART_ENSEMBL'].display_name.split(' ')[-1]), server['ENSEMBL_MART_ENSEMBL'].display_name
#     release=server['ENSEMBL_MART_ENSEMBL'].display_name.split(' ')[-1]
    logging.info(f"{dataset_name} version: {release} is used")
    dataset = Dataset(name=dataset_name,host=serverp)
    if attributes is None:
        to_table(dataset.list_attributes(),'test/biomart_attributes.tsv')
        logging.info("choose the attributes from: test/biomart_attributes.tsv")
        attributes = input(f"attributes space separated. e.g. a,b:").split(' ')
    if filters is None:
        to_table(dataset.list_filters(),'test/biomart_filters.tsv')        
        logging.info("choose the attributes from: test/biomart_filters.tsv")
        filters = eval(input(f"filters as python dict."))
    outp=f"{databasep}/www.ensembl.org/biomart/{dataset_name}/{release}/{'_'.join(sorted(attributes))}_{'_'.join(sorted(list(filters.keys())))}.pqt"
    if len(basenamenoext(outp))>50:
        outp=f"{dirname(outp)}/{basenamenoext(outp)[:50]}_plus{len(basenamenoext(outp))-50}{splitext(outp)[1]}"
    if not exists(outp) or force:
        df1=dataset.query(attributes=np.unique(attributes),
                         filters=filters,
                         only_unique=True,
                         **kws_query)
        to_table(df1,outp)
    else:
        df1=read_table(outp)
    return df1