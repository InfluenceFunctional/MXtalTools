from mxtaltools.constants.asymmetric_units import RAW_ASYM_UNITS
from mxtaltools.constants.space_group_info import SPACE_GROUPS

names = []
numbers = []
for ind in RAW_ASYM_UNITS.keys():
    names.append(
        SPACE_GROUPS[int(ind)]
    )
    numbers.append(ind)

use_texttt=True
font_size="\\footnotesize"
ncols = 5
spacegroups = [f"{num}:{name}" for num, name in zip(numbers, names)]
def fmt(s):
    return f"\\texttt{{{s}}}" if use_texttt else s

lines = [fmt(s) + r"\\" for s in spacegroups]

latex = []
latex.append(r"\begin{center}")
latex.append(rf"\begin{{multicols}}{{{ncols}}}")
latex.append(font_size)
latex.extend(lines)
latex.append(r"\end{multicols}")
latex.append(r"\end{center}")

latex_str = "\n".join(latex)


# ---- example usage ----
spacegroups = [
    "P1", "P-1", "P2", "P21", "C2",
    "Pm", "Pc", "Cm", "Cc",
    "P2/m", "P21/m", "C2/m",
    "P2/c", "P21/c", "C2/c",
]

print(latex_str)