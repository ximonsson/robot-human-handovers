THESIS=thesis
TEX=pdflatex
BIBTEX=biber
OUT=build/tex
BCF=$(OUT)/$(DOC).bcf
GNUPLOT=gnuplot
TEXSRC=tex

GNUPLOTSRC=clusters.gp scores.gp samples_silhouette_score.gp
GNUTERM="epslatex color"

TEXFLAGS=-output-directory=$(OUT) \
		 --shell-escape

# plot files from classification step
CLS_PLOT_DIR=src/classification/plot
CLS_PLOT_SRC= \
			 results-loss.gp \
			 results-val.gp \
			 results-test.gp \
			 confmat.gp
CLS_PLOT_TEX=$(addprefix $(TEXSRC)/plot_classification__, $(CLS_PLOT_SRC:.gp=.tex))

# plot files from clustering step
CLT_PLOT_DIR=src/clustering/plot
CLT_PLOT_SRC=\
			 sum-sqr-dist.gp \
			 mean-silhouette.gp \
			 clusters-3d.gp \
			 clusters-2d.gp \
			 obj-sampl-assgmnt.gp
CLT_PLOT_TEX=$(addprefix $(TEXSRC)/plot_clustering__, $(CLT_PLOT_SRC:.gp=.tex))


all: ref $(THESIS)


$(THESIS): plot
	@mkdir -p $(OUT)
	TEXINPUTS=.:./$(TEXSRC): $(TEX) $(TEXFLAGS) $@


ref: $(THESIS)
	$(BIBTEX) --output-directory=$(OUT) $(THESIS)


# plots
plot: plot_silhouette plot_classification plot_clustering

$(TEXSRC)/plot_clustering__clusters-%.tex: $(CLT_PLOT_DIR)/clusters-%.gp
	GNUTERM=$(GNUTERM) $(GNUPLOT) -e "outputfile='$@'" -c $< 6


plot_silhouette: tex/plot_silhouette_5.tex tex/plot_silhouette_6.tex tex/plot_silhouette_7.tex

tex/plot_silhouette_5.tex: src/clustering/plot/samples_silhouette_score.gp
	GNUTERM=$(GNUTERM) $(GNUPLOT) -e "outputfile='$@'" -c $< results/clustering/silhouette_sample_values_5.dat

tex/plot_silhouette_6.tex: src/clustering/plot/samples_silhouette_score.gp
	GNUTERM=$(GNUTERM) $(GNUPLOT) -e "outputfile='$@'" -c $< results/clustering/silhouette_sample_values_6.dat

tex/plot_silhouette_7.tex: src/clustering/plot/samples_silhouette_score.gp
	GNUTERM=$(GNUTERM) $(GNUPLOT) -e "outputfile='$@'" -c $< results/clustering/silhouette_sample_values_7.dat


plot_clustering: $(CLT_PLOT_TEX) # results/clustering/%.dat

$(TEXSRC)/plot_clustering__%.tex: $(CLT_PLOT_DIR)/%.gp
	GNUTERM=$(GNUTERM) $(GNUPLOT) -e "outputfile='$@'" -c $<


plot_classification: $(CLS_PLOT_TEX)

$(TEXSRC)/plot_classification__%.tex: $(CLS_PLOT_DIR)/%.gp
	GNUTERM=$(GNUTERM) $(GNUPLOT) -e "outputfile='$@'" -c $<


# images
imgs: method_imgs result_imgs

method_imgs:
	montage img/methods/masks/*.jpg -geometry 128x128+2+2 -tile x5 img/methods/masks.jpg
	montage img/methods/objects/*.jpg -geometry 128x128+2+2 -tile x5 img/methods/objects.jpg

result_imgs:
	montage img/results/objects/*.jpg -geometry 128x128+2+2 -tile x5 img/results/objects.jpg
	montage img/results/bad-images/*.jpg -geometry 128x128+2+2 -tile x5 img/results/bad-images.jpg

clean:
	rm tex/plot_*
