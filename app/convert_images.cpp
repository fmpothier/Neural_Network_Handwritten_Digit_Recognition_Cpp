#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>

//converts byte from big endian to little endian
int convert_endian(char num[4]){
	
	//convert to little endian by shifting bits to the left
	int converted_num = (static_cast<unsigned char>(num[0]) << 24) | (static_cast<unsigned char>(num[1]) << 16) |
			 (static_cast<unsigned char>(num[2]) << 8) | static_cast<unsigned char>(num[3]);
	
	return converted_num;
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

	//print numbers after converted to make sure they are right
	std::cout <<"number of images: " << imagesNum_little << "\nnumber of rows: " << rowNum_little << "\nnumber of collums: " << 			colsNum_little << std::endl;

	//vector to store the images and pixal values in
	std::vector<std::vector<unsigned char>> images;
	
	//loop over all images
	for(int i = 0; i < imagesNum_little; ++i){
		//create a vector to wholed pixals for each image
		std::vector<unsigned char> img(rowNum_little * colsNum_little);

		//read pixel values for each image into img vector
		file.read((char*)(img.data()), rowNum_little * colsNum_little);
		
		//add image to the back of the result vector
		images.push_back(img);
	}
	
	//close file
	file.close();

	return images;
}

int main(){
	
	std::vector<std::vector<unsigned char>> image = read_images("images/train-images.idx3-ubyte");	
	
	//loop over images
	for(int i = 0; i < 5/*(int)image.size()*/; ++i){
		
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
			
			std::cout<<(int)image[i][j]<< " ";
			
			//increament colume counter
			++cols;

			//to keep track of rows
			if(j%28 == 0 && j!=0){
				++row;
				cols = 0;
				std::cout<<std::endl;
			}	
		}	
	}	
	
	return 0;
}
