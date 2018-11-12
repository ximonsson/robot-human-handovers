DOC=thesis
TEX=pdflatex
BIBTEX=biber
OUT=build/tex
BCF=$(OUT)/$(DOC).bcf


all: bib pdf

pdf:
	@mkdir -p $(OUT)
	$(TEX) -output-directory=$(OUT) $(DOC)

bib:
	$(BIBTEX) --output-directory=$(OUT) $(DOC)

imgs: method_imgs result_imgs

method_imgs:
	montage img/methods/masks/*.jpg -geometry 128x128+2+2 -tile x5 img/methods/masks.jpg
	montage img/methods/objects/*.jpg -geometry 128x128+2+2 -tile x5 img/methods/objects.jpg

result_imgs:
	montage img/results/object_handovers/*.jpg -geometry 128x128+2+2 -tile x5 img/results/object_handovers.jpg
