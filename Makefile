CC = g++
CFLAGS = -O3 -std=c++11
INCLUDE_DIR = include

all: network

network: src/multi_layer_perception.cc
	$(CC) $(CFLAGS) -c src/multi_layer_perception.cc -o build/network.o -I $(INCLUDE_DIR) 

test: test/test_network_mlp.cc network
	$(CC) $(CFLAGS) -o build/test_network test/test_network_mlp.cc build/network.o -I $(INCLUDE_DIR)

clean:
	rm -f build/*.o build/*~ build/test_network
