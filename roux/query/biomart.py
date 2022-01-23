from roux.lib.io import *

release2preffix={100:'apr2020.archive',
                101:'aug2020.archive',
                102:'nov2020.archive',
                103:'feb2021.archive',
                104:'may2021.archive',
                 93:'jul2018.archive',
                 75:'feb2014.archive',
                 'grch37':'grch37',
                }
## Ref: https://m.ensembl.org/info/website/archives/index.html 

def query(release,
          attributes=None,
          filters=None,
            databasep='data/database',
            dataset_name='hsapiens_gene_ensembl',
            force=False,
             **kws_query,):
    """
    
        filters={
              # REMOVE: mitochondria/sex chr
                 'chromosome_name':[str(i) for i in list(range(1,23))],
              # REMOVE: non protein coding
                 'biotype':['protein_coding'],
                 }

    TODO: restrict to a ensembl release version
    """    
    from pybiomart import Server,Dataset
    serverp=f"http://{release2preffix[release]}.ensembl.org"       
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