#include "num_recognizer.hpp"
#include "ann_trainer.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>

using namespace ann;
namespace fs = std::filesystem;

int main(int argc, char** argv){

	if(argc < 2) throw std::invalid_argument("Image file name required.");

	// Need to check size of image, if not 28x28, convert to 28x28.
	// Read pixel values of each pixel into vector (row major order) of size 784.
	// Normalize pixel values with sigmoid function in vector so input layer
	// activations are same range as hidden layers. Close file.

	cv::Mat image;
	if(argc == 2){
		// Open file (check file type). 	
		image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);	
		if(!image.data) throw std::runtime_error("Image is empty or could not be read.");
	}
	else{ return -1; }
	if(image.rows != 28 || image.cols != 28){
		cv::Mat resized_image;
		cv::resize(image, resized_image, cv::Size(28, 28));	
		image = resized_image;
		if(!image.data) throw std::runtime_error("Image could not be resized.");
	}

	if(!image.isContinuous()) image = image.clone();	
	image.convertTo(image, CV_64F);

	neural_network::Vector input = Eigen::Map<neural_network::Vector>(image.ptr<double>(),
	 image.rows * image.cols);

	// Convert to row-major form
	input = input.transpose();
	input = input / 255.0;

	fs::path base_path = fs::current_path() / "lib";
        fs::path weight_path = base_path / "weights.bin";
        fs::path bias_path = base_path / "biases.bin";

	std::vector<neural_network::Matrix> weights = read_weights(weight_path);
	std::vector<neural_network::Vector> biases = read_biases(bias_path);

	const std::vector<std::size_t> layers{784, 512, 256, 128, 10};
	neural_network NN(layers, weights, biases);

	neural_network::Vector output = NN.feed_forward(input);
	Eigen::Index digit;
	double maxActivation = output.maxCoeff(&digit);
	std::cout << "Guess: " << digit << " Confidence: " << maxActivation << std::endl;
	std::cout << "Output: " << output << std::endl;

	return 0;
}
