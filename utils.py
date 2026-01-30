class TOC:
    def __init__(self, object, md: str | None = None, num_sep=2) -> None:
        self.num_sep = num_sep
        self.object = object
        if md:
            headers = [(x.split(" ")[0], " ".join(x.split(" ")[1:]), " ".join(x.split(" ")[1:]).lower().replace(" ", "-"))  for x in md.split("\n") if x.startswith("#")]
            self.toc = [f"{'&nbsp;'*self.num_sep*len(x)}[{y}](#{z})" for x,y,z in headers]
        else:
            self.toc = []

    def __call__(self, sep: str = "<br>") -> str:
        self.object.markdown(sep.join(self.toc), unsafe_allow_html=True)

    def append_md(self, md: str) -> None:
        headers = [(x.split(" ")[0], " ".join(x.split(" ")[1:]), " ".join(x.split(" ")[1:]).lower().replace(" ", "-"))  for x in md.split("\n") if x.startswith("#")]
        self.toc += [f"{'&nbsp;'*self.num_sep*len(x)}[{y}](#{z})" for x,y,z in headers]

    def append(self, object: Any, name:str, anchor:str, indent:int=0):
        if object:
            if indent==0:
                object.header(name, anchor=anchor)
            else:
                object.subheader(name, anchor=anchor)
        self.toc.append(f"{'&nbsp;'*self.num_sep*indent}[{name}](#{anchor})")