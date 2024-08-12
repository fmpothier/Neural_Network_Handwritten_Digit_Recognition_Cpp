#include "ann_trainer.hpp"
#include <filesystem>

using namespace ann;
namespace fs = std::filesystem;

int main(){

	fs::path base_path = fs::current_path() / "lib";

	const std::vector<std::size_t> layers{784, 512, 256, 128, 10};

	neural_network::Matrix W(10,10);
	neural_network::Vector b(10);
	W.setRandom(); b.setRandom();
	std::cout << "Weights: " << W << std::endl;
	std::cout << "Biases: " << b << std::endl;

	std::vector<neural_network::Matrix> W_vec;
	W_vec.push_back(W);
	std::vector<neural_network::Vector> b_vec;
	b_vec.push_back(b);

	
	fs::path weight_path = base_path / "weights.bin";
	fs::path bias_path = base_path / "biases.bin";
	if(write_binary(W_vec, b_vec, weight_path.string(), bias_path.string())){

	}
	else return -1;

	std::vector<neural_network::Matrix> W1_vec = read_weights("lib/weights.bin");
	std::vector<neural_network::Vector> b1_vec = read_biases("lib/biases.bin");

	std::cout << "New weights: " << W1_vec.at(0) << std::endl;
	std::cout << "New biases: " << b1_vec.at(0) << std::endl;


	return 0;
}
