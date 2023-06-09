"""For diagrams e.g. flowcharts"""

def diagram_nb(
    graph: str,
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
            N1[("Step1")]
            ---> N2["Step2"]
        
    """
    import base64
    from IPython.display import Image, display
    graphbytes = graph.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    url="https://mermaid.ink/img/" + base64_string
    display(Image(url=url))
    if out:
        return url
    