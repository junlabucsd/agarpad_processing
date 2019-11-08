EXPNAME :=  20191029_SJ1486_wateragar

PARAMS_ND2 := roles/process_nd2.yaml
PARAMS_PRE := roles/preprocess_images.yaml
PARAMS_SEG := roles/segmentation.yaml
PARAMS_COL := roles/collection.yaml
PARAMS_PDIM := roles/dimensions.yaml
PARAMS_PFL := roles/fluorescence.yaml

ND2 := ../data/$(EXPNAME).nd2
CELLS := cells/collection/collection.js

# target variables
T_ND2 := target_nd2
T_PREDEBUG := target_predebug
T_PRE := target_pre
T_OTSU := target_otsu
T_SEGDEBUG := target_segdebug
T_SEG := target_seg
T_COL := target_col $(CELLS)
T_PDIM := plot_dim
T_PFL := plot_flu

# other variables
FOVPREF_DEBUG := f00
FOVPREF := f

default: $(T_OTSU)
process: $(T_COL)
plots: $(T_PDIM) $(T_PFL)
all: plots process

##############################################################################
# IMAGE PROCESSING TARGETS
##############################################################################
# collection_cells.py -- make cell dictionary
$(T_COL): $(T_SEG) $(PARAMS_SEG)
	python code/image_processing/collection_cells.py -f $(PARAMS_COL) -d .
	touch $(T_COL)

# segmentation_cells.py -- segmentation all
$(T_SEG): $(T_SEGDEBUG) $(PARAMS_SEG)
	python code/image_processing/segmentation_cells.py -f $(PARAMS_SEG) -d .  -i TIFFS_preprocessed/$(EXPNAME)_$(FOVPREF)*.tif
	touch $(T_SEG)

# segmentation.py -- segmentation debug
$(T_SEGDEBUG): $(T_OTSU) $(PARAMS_SEG)
	python code/image_processing/segmentation_cells.py -f $(PARAMS_SEG) -d .  --debug -i TIFFS_preprocessed/$(EXPNAME)_$(FOVPREF_DEBUG)*.tif
	touch $(T_SEGDEBUG)

# utils.py -- otsu analysis
$(T_OTSU): $(T_PRE)
	python code/image_processing/utils.py --otsu -d TIFFS -i TIFFS/$(EXPNAME)_$(FOVPREF)*.tif
	python code/image_processing/utils.py --otsu -d TIFFS_preprocessed -i TIFFS_preprocessed/$(EXPNAME)_$(FOVPREF)*.tif
	@echo "NOW STOP A MOMENT AND FILL-IN THE OTSU THRESHOLD IN THE 'segmentation.yaml' file."
	touch $(T_OTSU)

# preprocess_images.py -- all
$(T_PRE): $(T_PREDEBUG) $(PARAMS_PRE)
	python code/image_processing/preprocess_images.py -f $(PARAMS_PRE) -d TIFFS_preprocessed -i TIFFS/$(EXPNAME)_$(FOVPREF)*.tif
	touch $(T_PRE)

# preprocess_images.py -- debug
$(T_PREDEBUG): $(T_ND2) $(PARAMS_PRE)
	python code/image_processing/preprocess_images.py -f $(PARAMS_PRE) -d TIFFS_preprocessed --debug -i TIFFS/$(EXPNAME)_$(FOVPREF_DEBUG)*.tif
	touch $(T_PREDEBUG)

# process_nd2.py
$(T_ND2): $(ND2) $(PARAMS_ND2)
	python code/image_processing/process_nd2.py -f $(PARAMS_ND2) -d . $(ND2)
	touch $(T_ND2)

##############################################################################
# ANALYSIS AND PLOTS
##############################################################################
# analysis.py
$(T_PDIM): $(T_COL) $(PARAMS_PDIM)
	python code/analysis/analysis.py -f $(PARAMS_PDIM) -d . $(CELLS)
	touch $(T_PDIM)

$(T_PFL): $(T_COL) $(PARAMS_PFL)
	python code/analysis/analysis.py -f $(PARAMS_PFL) -d . $(CELLS)
	touch $(T_PFL)
##############################################################################
# UTILS
##############################################################################
# dummy targets to prevent further processing
dummy:
	touch $(T_ND2)
	touch $(T_PREDEBUG)
	touch $(T_PRE)
	touch $(T_SEGDEBUG)
	touch $(T_SEG)
	touch $(T_COL)
	touch $(T_PDIM)
	touch $(T_PFL)

##############################################################################
# HELP AND DOC ON MAKEFILES
##############################################################################
# https://www.gnu.org/software/make/manual/make.html
# https://makefiletutorial.com/

