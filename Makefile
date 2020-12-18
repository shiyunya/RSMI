CC=g++ -O3 -std=c++14#14
SRCS=$(wildcard *.cpp */*.cpp)
OBJS=$(patsubst %.cpp, %.o, $(SRCS))

# for MacOs
# INCLUDE = -I/usr/local/include/libtorch/include -I/usr/local/include/libtorch/include/torch/csrc/api/include
# LIB +=-L/usr/local/include/libtorch/lib -ltorch -lc10 -lpthread 
# FLAG = -Xlinker -rpath -Xlinker /usr/local/include/libtorch/lib

TYPE = CPU
# TYPE = GPU

# for linux
ifeq ($(TYPE), GPU)
	INCLUDE = -I/mnt/d/libtorch-cxx11-abi-shared-with-deps-1.7.0+cpu/libtorch_gpu/include -I/mnt/d/libtorch-cxx11-abi-shared-with-deps-1.7.0+cpu/libtorch_gpu/include/torch/csrc/api/include
	LIB +=-L/mnt/d/libtorch-cxx11-abi-shared-with-deps-1.7.0+cpu/libtorch_gpu/lib -ltorch -lc10 -lpthread
	FLAG = -Wl,-rpath=/mnt/d/libtorch-cxx11-abi-shared-with-deps-1.7.0+cpu/libtorch_gpu/lib
else
	INCLUDE = -I ../libtorch-cxx11-abi-shared-with-deps-1.4.0+cpu/libtorch/include -I ../libtorch-cxx11-abi-shared-with-deps-1.4.0+cpu/libtorch/include/torch/csrc/api/include -I ../boost_1_73_0
	LIB += -L ../libtorch-cxx11-abi-shared-with-deps-1.4.0+cpu/libtorch/lib -ltorch -lc10 -lpthread
	FLAG = -Wl,-rpath=../libtorch-cxx11-abi-shared-with-deps-1.4.0+cpu/libtorch/lib
endif



# INCLUDE = -I/mnt/d/libtorch-cxx11-abi-shared-with-deps-1.7.0+cpu/libtorch/include -I/mnt/d/libtorch-cxx11-abi-shared-with-deps-1.7.0+cpu/libtorch/include/torch/csrc/api/include
# LIB +=-L/mnt/d/libtorch-cxx11-abi-shared-with-deps-1.7.0+cpu/libtorch/lib -ltorch -lc10 -lpthread
# FLAG = -Wl,-rpath=/mnt/d/libtorch-cxx11-abi-shared-with-deps-1.7.0+cpu/libtorch/lib

NAME=$(wildcard *.cpp)
TARGET=$(patsubst %.cpp, %, $(NAME))


$(TARGET):$(OBJS)
	$(CC) -o $@ $^ $(INCLUDE) $(LIB) $(FLAG)
%.o:%.cpp
	$(CC) -o $@ -c $< -g $(INCLUDE)

clean:
	rm -rf $(TARGET) $(OBJS)

# # g++ -std=c++11 Exp.cpp FileReader.o -ltensorflow -o Exp_tf

# g++ -O3 -std=c++14 -o Exp Exp.o utils/FileReader.o utils/ExpRecorder.o utils/Constants.o utils/FileWriter.o entities/Mbr.o entities/NodeExtend.o entities/NonLeafNode.o entities/LeafNode.o entities/Node.o entities/Point.o curves/hilbert.o curves/z.o curves/hilbert4.o -I/mnt/d/libtorch-cxx11-abi-shared-with-deps-1.7.0+cpu/libtorch/include -I/mnt/d/libtorch-cxx11-abi-shared-with-deps-1.7.0+cpu/libtorch/include/torch/csrc/api/include -I/mnt/d//boost_1_73_0 -L/mnt/d/libtorch-cxx11-abi-shared-with-deps-1.7.0+cpu/libtorch/lib -ltorch -lc10 -lpthread -Wl,-rpath=/mnt/d/libtorch-cxx11-abi-shared-with-deps-1.7.0+cpu/libtorch/lib