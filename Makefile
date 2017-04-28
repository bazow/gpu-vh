DIR_MAIN       = ./
DIR_SRC        = $(DIR_MAIN)rhic
DIR_H          = $(DIR_MAIN)include/
DIR_BUILD      = $(DIR_MAIN)build/
DIR_OBJ        = $(DIR_BUILD)rhic

DEBUG =
OPTIMIZATION = -O5 
FLOWTRACE =
OPTIONS = --relocatable-device-code=true -use_fast_math --ptxas-options=-v -lineinfo -Wno-deprecated-gpu-targets
LINK_OPTIONS = --cudart static --relocatable-device-code=true -link -Wno-deprecated-gpu-targets
CFLAGS = $(DEBUG) $(OPTIMIZATION) $(FLOWTRACE) $(OPTIONS)
COMPILER = nvcc
LIBS = -lm -lgsl -lgslcblas -lconfig -lgtest
INCLUDES = -I rhic/rhic-core/src/include -I rhic/rhic-trunk/src/include -I rhic/rhic-harness/src/include  

CPP := $(shell find $(DIR_SRC) -name '*.cpp')
CU := $(shell find $(DIR_SRC) -name '*.cu')
CPP_OBJ  = $(CPP:$(DIR_SRC)%.cpp=$(DIR_OBJ)%.o)
CU_OBJ  = $(CU:$(DIR_SRC)%.cu=$(DIR_OBJ)%.o)
OBJ = $(CPP_OBJ) $(CU_OBJ)

EXE =\
gpu-vh

$(EXE): $(OBJ)
	echo "Linking:   $@ ($(COMPILER))"
	$(COMPILER) $(LINK_OPTIONS) -o $@ $^ $(LIBS) $(INCLUDES)

$(DIR_OBJ)%.o: $(DIR_SRC)%.cpp
	@[ -d $(DIR_OBJ) ] || find rhic/rhic-core rhic/rhic-harness rhic/rhic-trunk -type d -exec mkdir -p ./build/{} \;
	@echo "Compiling: $< ($(COMPILER))"
	$(COMPILER) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(DIR_OBJ)%.o: $(DIR_SRC)%.cu
	@[ -d $(DIR_OBJ) ] || find rhic/rhic-core rhic/rhic-harness rhic/rhic-trunk -type d -exec mkdir -p ./build/{} \;
	@echo "Compiling: $< ($(COMPILER))"
	$(COMPILER) $(CFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	@echo "Object files and executable deleted"
	if [ -d "$(DIR_OBJ)" ]; then rm -rf $(EXE) $(DIR_OBJ)/*; rmdir $(DIR_OBJ); rmdir $(DIR_BUILD); fi

.SILENT:
