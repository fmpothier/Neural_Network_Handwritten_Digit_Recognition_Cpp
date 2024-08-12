#include "num_recognizer.hpp"
#include "ann_trainer.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>

using namespace ann;
namespace fs = std::filesystem;

void test(){
	fs::path base_path = fs::current_path() / "lib";
        fs::path weight_path = base_path / "weights.bin";
        fs::path bias_path = base_path / "biases.bin";
	fs::path testing_data_path = base_path / "images" / "testing_data";

	std::vector<neural_network::Matrix> weights = read_weights(weight_path);
	std::vector<neural_network::Vector> biases = read_biases(bias_path);

	const std::vector<std::size_t> layers{784, 512, 256, 128, 10};
	neural_network NN(layers, weights, biases);

	long int counter = 0;
	long int acc_counter = 0;

	for(auto& image : fs::directory_iterator(testing_data_path.string())){
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
			Eigen::Index digit;
			double maxActivation = output.maxCoeff(&digit);

                        std::string image_name = image.path().filename().string();
                        int expected_value = image_name[0] - '0';

			if(digit == expected_value) ++acc_counter;

			++counter;
                }
        }
	double acc_counterD = acc_counter;
	double counterD = counter;
	double accuracy = (100.0 * (acc_counterD / counterD));
	std::cout << "Accuracy: " << accuracy << "%" << std::endl;
}

int main(int argc, char** argv){
	try{
	if(argc < 2) throw std::invalid_argument("Image file name required.");

	// Need to check size of image, if not 28x28, convert to 28x28.
	// Read pixel values of each pixel into vector (row major order) of size 784.
	// Normalize pixel values with sigmoid function in vector so input layer
	// activations are same range as hidden layers. Close file.

	cv::Mat image;
	bool debug_flag = false;
	bool test_flag = false;

	if(argc == 2){
		// Check if test mode or normal mode
		if(!strcmp(argv[1], "test")){
			test();	
			return 0;
		}

		// Open file (check file type). 	
		try {
		    image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);	
		    if(!image.data) throw std::runtime_error("Image is empty or could not be read.");
		} catch(...){
	  	    throw std::runtime_error(
		    "File type invalid. Input file of type .pnm .pgm .png or .jpg");
		}
	}
	else if(argc == 3){
		// Open file (check file type). 	
		if(strcmp(argv[1], "debug")) throw std::invalid_argument("Incorrect input format."); 
		// Open file (check file type). 	
		try {
		    image = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);	
		    if(!image.data) throw std::runtime_error("Image is empty or could not be read.");
		} catch(...){
	  	    throw std::runtime_error(
		    "File type invalid. Input file of type .pnm .pgm .png or .jpg");
		}
		debug_flag = true;
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
	std::cout << "Value: " << digit << "\n" << "Confidence: " << maxActivation*100.0 << "%" << "\n";
	if(debug_flag){
		std::cout << "Output layer activation values: "; 
	 	for(int i = 0; i < output.size(); ++i) std::cout << i << "(" << output(i) << ") ";
		std::cout << std::endl;
	}

	// Handling any errors that may be thrown
	} catch(const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	} catch(...){ return -1; }

	return 0;
}
