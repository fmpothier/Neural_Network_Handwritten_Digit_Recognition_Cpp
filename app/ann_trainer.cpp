#include "num_recognizer.hpp"
#include "ann_trainer.hpp"
#include <fstream>

using namespace ann;

int main(){

	const std::vector<std::size_t> layers{784, 512, 256, 128, 10};
	// Construct a neural network with random weights.
	//neural_network NN(layers); 

	return 0;
}
