import pandas as pd
# stats 
def perc_label(a,b=None,bracket=True): 
    from roux.lib.str import num2str
    if b is None:
        b=len(a)
        a=sum(a)
    return f"{(a/b)*100:.0f}%"+(f" ({num2str(a)}/{num2str(b)})" if bracket else "")

def pval2annot(pval,
               alternative=None,
               alpha=None,
               fmt='*',
               power=True,
               linebreak=False,
               prefix='P',
               q=False,
              ):
    """
    fmt: *|<|'num'    
    """
    if alternative is None and alpha is None:
        raise ValueError('both alternative and alpha are None')
    if alpha is None:
        alpha=0.025 if alternative=='two-sided' else 0.05
    if pd.isnull(pval):
        annot= ''
    elif pval < 0.0001:
        annot= "****" if fmt=='*' else \
        f"P<\n{0.0001:.0e}" if fmt=='<' else \
        f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else \
        f"P=\n{pval:.1g}"  if not linebreak else \
        f"P={pval:.1g}"
    elif (pval < 0.001):
        annot= "***"  if fmt=='*' else \
        f"P<\n{0.001:.0e}" if fmt=='<' else \
        f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else \
        f"P=\n{pval:.1g}" if not linebreak else \
        f"P={pval:.1g}"
    elif (pval < 0.01):
        annot= "**" if fmt=='*' else \
        f"P<\n{0.01:.0e}" if fmt=='<' else \
        f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else \
        f"P=\n{pval:.1g}" if not linebreak else \
        f"P={pval:.1g}"
    elif (pval < alpha):
        annot= "*" if fmt=='*' else \
        f"P<\n{alpha}" if fmt=='<' else \
        f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else \
        f"P=\n{pval:.1g}" if not linebreak else \
        f"P={pval:.1g}"
    else:
        annot= "ns" if fmt=='*' else \
        f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else \
        f"P=\n{pval:.1g}" if not linebreak else \
        f"P={pval:.1g}"
    annot=annot if linebreak else annot.replace('\n','')
    if prefix!='P':
        annot=annot.replace('P',prefix if not q else 'q')
    if power and 'e' in annot:
        annot=annot.replace('e-0','e-').replace('e','x$10^{')+'}$'
    return annot 