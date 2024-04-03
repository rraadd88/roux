"""For diagrams e.g. flowcharts"""

def diagram_nb(
    graph: str,
    counts: dict=None,
    out: bool=False,
    ):
    """
    Show a diagram in jupyter notebook using mermaid.js.
    
    Parameters:
        graph (str): markdown-formatted graph. Please see https://mermaid.js.org/intro/n00b-syntaxReference.html
        out (bool): Output the URL. Defaults to False.  
    
    References:
        1. https://mermaid.js.org/config/Tutorials.html#jupyter-integration-with-mermaid-js
        
    Examples:
    
        graph LR;
            i1(["input1"]) & d1[("data1")]
            --> 
                p1[["process1"]] 
                    --> o1(["output1"])
                p1 
                    --> o2["output2"]:::ends
        classDef ends fill:#fff,stroke:#fff
    """
    from roux.lib.str import replace_many
    if not counts is None:
        import re
        replaces={}
        for step,ds in counts.items():
            s1=re.split(
                step+r'.*?"',
                graph
                )[1].split('"')[0]
            if isinstance(ds,dict):
                ds=[ds]
            s2=s1+"\n"+"("+(',\n'.join([f"{d['value']} {d['key']}s" for d in ds]))+")"
            replaces[s1]=s2
        graph=replace_many(graph,replaces)
    import base64
    from IPython.display import Image, display
    graphbytes = graph.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    url="https://mermaid.ink/img/" + base64_string
    display(Image(url=url))
    if out:
        return url
    