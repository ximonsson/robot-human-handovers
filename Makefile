DOC=thesis
TEX=pdflatex
BIBTEX=biber
OUT=build/tex
BCF=$(OUT)/$(DOC).bcf
GNUPLOT=gnuplot

GNUPLOTSRC=clusters.gp scores.gp samples_silhouette_score.gp
GNUTERM=epslatex

TEXFLAGS=-output-directory=$(OUT) \
		 -shell-escape


all: ref pdf


pdf: plot
	@mkdir -p $(OUT)
	$(TEX) $(TEXFLAGS) $(DOC)


ref:
	$(BIBTEX) --output-directory=$(OUT) $(DOC)


# plots
plot: plot_clusters plot_scores plot_silhouette


plot_clusters: tex/plot_clusters.tex

tex/plot_clusters.tex: src/clustering/plot/clusters.gp
	GNUTERM=$(GNUTERM) $(GNUPLOT) -e "outputfile='$@'" -c $< 6


plot_scores: tex/plot_scores.tex

tex/plot_scores.tex: src/clustering/plot/scores.gp
	GNUTERM=$(GNUTERM) $(GNUPLOT) -e "outputfile='$@'" -c $<


plot_silhouette: tex/plot_silhouette_5.tex tex/plot_silhouette_6.tex tex/plot_silhouette_7.tex

tex/plot_silhouette_5.tex: src/clustering/plot/samples_silhouette_score.gp
	GNUTERM=$(GNUTERM) $(GNUPLOT) -e "outputfile='$@'" -c $< results/clustering/silhouette_sample_values_5.dat

tex/plot_silhouette_6.tex: src/clustering/plot/samples_silhouette_score.gp
	GNUTERM=$(GNUTERM) $(GNUPLOT) -e "outputfile='$@'" -c $< results/clustering/silhouette_sample_values_6.dat

tex/plot_silhouette_7.tex: src/clustering/plot/samples_silhouette_score.gp
	GNUTERM=$(GNUTERM) $(GNUPLOT) -e "outputfile='$@'" -c $< results/clustering/silhouette_sample_values_7.dat


# images
imgs: method_imgs result_imgs

method_imgs:
	montage img/methods/masks/*.jpg -geometry 128x128+2+2 -tile x5 img/methods/masks.jpg
	montage img/methods/objects/*.jpg -geometry 128x128+2+2 -tile x5 img/methods/objects.jpg

result_imgs:
	montage img/results/object_handovers/*.jpg -geometry 128x128+2+2 -tile x5 img/results/object_handovers.jpg
