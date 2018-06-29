DOC=thesis
TEX=pdflatex
BIBTEX=biber
OUT=out

pdf:
	@mkdir -p $(OUT)
	$(TEX) -output-directory=$(OUT) $(DOC)

bib:
	$(BIBTEX) --output-directory=$(OUT) $(DOC)

all: pdf bib pdf
