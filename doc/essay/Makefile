DOC=thesis
TEX=pdflatex
BIBTEX=biber
OUT=out

all: bib pdf

pdf:
	@mkdir -p $(OUT)
	$(TEX) -output-directory=$(OUT) $(DOC)

bib:
	$(BIBTEX) --output-directory=$(OUT) $(DOC)
