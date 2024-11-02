"""For diagrams e.g. flowcharts"""
import logging 

def diagram_nb(
    graph: str,
    counts: dict = None,
    out: bool = False,
    test: bool = False,
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

    def get_ds(
        ds: dict,
    ):
        if isinstance(ds, dict):
            if sorted(list(ds.keys())) == sorted(["key", "value"]):
                ds = [ds]
            else:
                ds_ = []
                for k, v in ds.items():
                    ds_.append(
                        {
                            "key": k,
                            "value": v,
                        },
                    )
                ds = ds_
                # ds=[{
                #     'key':list(ds.keys())[0],
                #     'value':list(ds.values())[0],
                # }]
        return ds

    from roux.lib.str import replace_many

    if counts is not None:
        import re

        replaces = {}
        for step, ds in counts.items():
            # print(ds)
            # uniform format
            if isinstance(ds, list):
                ds_ = []
                for d in ds:
                    ds_ += get_ds(d)
                ds = ds_
            ds = get_ds(ds)
            if test:
                print("counts", ds)

            try:
                regex = step + r'[\[\(\[].*?"'
                if test:
                    print(f"regex: {regex}")
                s1 = re.split(regex, graph)[1].split('"')[0]
            except:
                logging.warning(f'node and/or label missing for {step}')
                s1=None
            if s1 is not None:
                s2 = (
                    s1
                    + "\n"
                    + "("
                    + (",\n".join([f"{d['value']} {d['key']}s" for d in ds]))
                    + ")"
                )
                # if test:
                #     print('replaces',f"{s1} : {s2}")
                replaces[s1] = s2
                
            if test:
                print(replaces)
        if test:
            print("\n", replaces)
        graph = replace_many(
            graph,
            replaces,
            ignore=True,
        )

    ## diagram
    import base64
    from IPython.display import Image, display

    graphbytes = graph.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    url = "https://mermaid.ink/img/" + base64_string
    display(Image(url=url))

    if isinstance(out, bool):
        if out:
            return url
    elif isinstance(out, str):
        from pathlib import Path
        Path(out).parent.mkdir(parents=True,exist_ok=True)
        open(out, "w").write(graph)
        return out
