#include "ann_trainer.hpp"
#include <filesystem>

using namespace ann;
namespace fs = std::filesystem;
// learning rate 0.01 is a good starting point

int main(){

	using Matrix = neural_network::Matrix;
	using Vector = neural_network::Vector;

	fs::path base_path = fs::current_path() / "lib";
	fs::path weight_path = base_path / "weights.bin";
	fs::path bias_path = base_path / "biases.bin";
	fs::path training_data_path = base_path / "images" / "training_data";

	const std::vector<std::size_t> layers{784, 512, 256, 128, 10};
	neural_network NN(layers);
	std::vector<Matrix> weights = NN.get_weights();
	std::vector<Vector> biases = NN.get_biases();

	if(!write_binary(weights, biases, weight_path, bias_path))
	 throw std::runtime_error("Failed to write weight and bias files."); 

	for(auto& image : fs::directory_iterator(training_data_path.string())){
		if(image.is_regular_file()){
			std::string path_to_image = image.path().string();	
			cv::Mat input = cv::imread(path_to_image, cv::IMREAD_GRAYSCALE);
			if(!input.data) continue;

			if(input.rows != 28 || input.cols != 28) continue;

			if(!input.isContinuous()) input = input.clone();        

			input.convertTo(input, CV_64F);
			Vector input_vec = Eigen::Map<Vector>(input.ptr<double>(),
			 input.rows * input.cols);
			// Convert to row-major form
			input_vec = input_vec.transpose();
			input_vec = input_vec / 255.0;

			Vector output = NN.feed_forward(input_vec);


			std::string image_name = image.path().filename().string();
			Vector target(output.size());
			target.setZero();
			int expected_value = image_name[0] - '0';
			target(expected_value) = 1.0;
			
			NN.feed_backward(0.01, output, target);
		}
	}	
	weights = NN.get_weights();
	biases = NN.get_biases();

	if(!write_binary(weights, biases, weight_path, bias_path))
	 throw std::runtime_error("Failed to write new weight and bias files.");	

	return 0;
}
