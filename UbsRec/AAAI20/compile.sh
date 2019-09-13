# compile the paper
pdflatex  -papersize=letter  -halt-on-error main.tex
bibtex main.aux
pdflatex  -papersize=letter  -halt-on-error main.tex
pdflatex  -papersize=letter  -halt-on-error main.tex
