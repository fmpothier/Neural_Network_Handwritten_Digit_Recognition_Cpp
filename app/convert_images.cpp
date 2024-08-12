#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include <vector>
#include <Eigen/Dense>
#include <string>
#include <stdexcept>

namespace fs = std::filesystem;

// Converts byte from big endian to little endian
int convert_endian(char num[4]){
	
	// Convert to little endian by shifting bits to the left
	int converted_num = (static_cast<unsigned char>(num[0]) << 24) | (static_cast<unsigned char>(num[1]) << 16) |
			 (static_cast<unsigned char>(num[2]) << 8) | static_cast<unsigned char>(num[3]);
	
	return converted_num;
}

// This function gets the labels for images
std::vector<std::vector<unsigned char>> read_labels(const std::string& file_name){
	// Open file as binary type
	std::ifstream file(file_name, std::ios::binary);
	
	char magicNum[4];
	char imagesNum[4];

	// Read the numbers from the file
	file.read(magicNum, 4);
	file.read(imagesNum, 4);
	
	// Convert to little endian
	int num_converted = convert_endian(imagesNum);
	
	// Vector to store the images and pixal values in
	std::vector<std::vector<unsigned char>> labels;
	
	// Loop over all images
	for(int i = 0; i < num_converted; ++i){
		// Create a vector to wholed the labels each label is 1 byte
		std::vector<unsigned char> img(1);

		// Read label and add it to the vector
		file.read((char*)(img.data()), 1);
		labels.push_back(img);
	}
	
	// Close file
	file.close();

	return labels;
}

// Outer vector has all the images, inner vector has all of the pixels per image (28 x 28)
std::vector<std::vector<unsigned char>> read_images(const std::string& file_name){
	//open file as binary type
	std::ifstream file(file_name, std::ios::binary);
	
	//need save magic number, number of images, rows, and collums and each is 4 bytes
	char magicNum[4];
	char imagesNum[4];
	char rowNum[4];
	char colsNum[4];
	
	//read the numbers from the file
	file.read(magicNum, 4);
	file.read(imagesNum, 4);
	file.read(rowNum, 4);
	file.read(colsNum, 4);
	
	//convert to little endian
	int imagesNum_little = convert_endian(imagesNum);
	int rowNum_little = convert_endian(rowNum);
	int colsNum_little = convert_endian(colsNum);

	//vector to store the images and pixal values in
	std::vector<std::vector<unsigned char>> images;
	
	//loop over all images
	for(int i = 0; i < imagesNum_little; ++i){
		//create a vector to wholed pixals for each image
		std::vector<unsigned char> img(rowNum_little * colsNum_little);

		//read pixel values for each image into img vector
		file.read((char*)(img.data()), rowNum_little * colsNum_little);
		images.push_back(img);
	}
	
	//close file
	file.close();

	return images;
}

int main(){
  try{	
	std::vector<std::vector<unsigned char>> image = read_images("lib/images/t10k-images.idx3-ubyte");
	std::vector<std::vector<unsigned char>> label = read_labels("lib/images/t10k-labels.idx1-ubyte");
	fs::path base_path = fs::current_path() / "lib" / "images" / "testing_data";

	// Comment out the previous three lines, and uncomment these
	// three lines below to create the training data images.

	// std::vector<std::vector<unsigned char>> image = read_images("lib/images/train-images.idx3-ubyte");
	// std::vector<std::vector<unsigned char>> label = read_labels("lib/images/train-labels.idx1-ubyte");
	// fs::path base_path = fs::current_path() / "lib" / "images" / "training_data";

	if(!fs::exists(base_path)){
		fs::create_directories(base_path);
	}

	// Loop over images
	for(size_t i = 0; i < image.size(); ++i){
		// Create a temp image
		cv::Mat tmp = cv::Mat::zeros(cv::Size(28, 28), CV_8UC1);
		
		// Make sure the image is of size 28 by 28
		if(image[i].size() != 784){
			std::cerr << "Image " << i << " does not have 784 pixels." << std::endl;
        		continue; // Skip this iteration if the image is not valid
		}
	
		// Variable to count the rows and cols
		int row = 0;
		int cols = 0;

		// Loop over pixels
		for(size_t j = 0; j < image[i].size(); ++j){
			// Write image data to the tmp image
			int row = j / 28;
			int col = j % 28;
			tmp.at<uchar>(cv::Point(col, row)) = static_cast<uchar>(image[i][j]);
		}
		
		std::string label_str = std::to_string(static_cast<int>(label[i][0]));
		std::string filename = label_str + "_t" + std::to_string(i + 1) + ".pgm";
		fs::path full_path = base_path / filename;
		cv::imwrite(full_path.string(), tmp);
	}	
	
        } catch(const std::exception& e) {
                std::cerr << e.what() << std::endl;
                return -1;
        } catch(...){ return -1; }

	return 0;
}
