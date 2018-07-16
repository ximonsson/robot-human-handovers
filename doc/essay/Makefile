DOC=thesis
TEX=pdflatex
BIBTEX=biber
OUT=out
BCF=$(OUT)/$(DOC).bcf

all: bib pdf

pdf:
	@mkdir -p $(OUT)
	$(TEX) -output-directory=$(OUT) $(DOC)

bib:
	$(BIBTEX) --output-directory=$(OUT) $(DOC)
