#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

//converts byte from big endian to little endian
int convert_endian(char num[4]){
	
	//convert to little endian by shifting bits to the left
	int converted_num = (static_cast<unsigned char>(num[0]) << 24) | (static_cast<unsigned char>(num[1]) << 16) |
			 (static_cast<unsigned char>(num[2]) << 8) | static_cast<unsigned char>(num[3]);
	
	return converted_num;
}

//get labels for images
std::vector<std::vector<unsigned char>> read_labels(const std::string& file_name){
	//open file as binary type
	std::ifstream file(file_name, std::ios::binary);
	
	char magicNum[4];
	char imagesNum[4];

	//read the numbers from the file
	file.read(magicNum, 4);
	file.read(imagesNum, 4);
	
	//convert to little endian
	int num_converted = convert_endian(imagesNum);
	
	//vector to store the images and pixal values in
	std::vector<std::vector<unsigned char>> labels;
	
	//loop over all images
	for(int i = 0; i < num_converted; ++i){
		//create a vector to wholed the labels each label is 1 byte
		std::vector<unsigned char> img(1);

		//read label and add it to the vector
		file.read((char*)(img.data()), 1);
		labels.push_back(img);
	}
	
	//close file
	file.close();

	return labels;
	
}

// outer vector has all the iamges, inner vector has all the the pixels per image(28 x 28)
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
	
	std::vector<std::vector<unsigned char>> image = read_images("images/train-images.idx3-ubyte");	
	std::vector<std::vector<unsigned char>> label = read_labels("images/train-labels.idx1-ubyte");
	//view the images
	cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);

	//loop over images
	for(int i = 0; i < (int)image.size(); ++i){
		
		//create a temp image
		cv::Mat tmp = cv::Mat::zeros(cv::Size(28, 28), CV_8UC1);
		
		//make sure the image is of size 28 by 28
		if(image[i].size() != 784){
			std::cerr << "Image " << i << " does not have 784 pixels." << std::endl;
        		continue; // Skip this iteration if the image is not valid
		}
		
	
		//varble to count the rows and cols
		int row = 0;
		int cols = 0;

		//loop over pixels
		for(int j = 0; j < (int)image[i].size(); ++j){
			//write image data to the tmp image
			tmp.at<uchar>(cv::Point(cols, row)) = static_cast<uchar>(image[i][j]);

			//increament colume counter
			++cols;

			//to keep track of rows
			if(j%28 == 0 && j != 0){
				++row;
				
				//at the end of an image reset the colume counter
				cols = 0;
			}	
		}
		std::cout<< "Label: " << (int)label[i][0]<<std::endl; 
		cv::imshow("image", tmp);
		cv::waitKey(0);		
	}	
	
	return 0;
}
