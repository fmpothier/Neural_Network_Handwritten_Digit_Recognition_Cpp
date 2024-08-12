#ifndef ANN_TRAINER_HPP
#define ANN_TRAINER_HPP

#include "num_recognizer.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>

namespace ann{
	using Vector = neural_network::Vector;
	using Matrix = neural_network::Matrix; 
	
	bool write_binary(const std::vector<Matrix>& W, const std::vector<Vector>& b,
	 const std::string& weights_path, const std::string& biases_path){

		// open the weights file
		std::ofstream weights_file(weights_path, std::ios::binary | std::ios::trunc);

		//check if the file opened
		if(weights_file.is_open()){
			
			// loop over the list of Matrix
			for(const auto& weights : W){

				// get the size of the matrix
				int rows = weights.rows();
				int cols = weights.cols();

				// write the size over
				// these lines are converting the pointers from int* to char* 
				// and telling it how many bytes to write over
				weights_file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
		    		weights_file.write(reinterpret_cast<const char*>(&cols), sizeof(int));

		    		//write the matrix values over to the file
		    		weights_file.write(reinterpret_cast<const char*>(weights.data()),
				 rows*cols*sizeof(double));
			}
			weights_file.close();
		}
		else return false;

		// open biases file
		std::ofstream biases_file(biases_path, std::ios::binary | std::ios::trunc);

		// check if the file opened
		if(biases_file.is_open()){

			for(const auto& biases : b){

				// get the size
				int size = biases.rows();

				// write the size over
				// these lines are converting the pointers from int* to char*
				// and telling it how many bytes to write over
				biases_file.write(reinterpret_cast<const char*>(&size), sizeof(int));

		    		//write the matrix values over to the file
		    		biases_file.write(reinterpret_cast<const char*>(biases.data()),
				 size*sizeof(double));

			}
			biases_file.close();
		}
		else return false;
		
		return true;
	}

	std::vector<Matrix> read_weights(const std::string& file_path){
		// open the weights file
		std::ifstream file(file_path, std::ios::binary);

		// create the container to hold the matrixs
		std::vector<Matrix> weights;

		// check if the file is open
		if(file.is_open()){
	    	    while(!file.eof()){
			int rows;
			int cols;

			// read number of rows and cols. also check if that numbers are there
			// if not leave loop
		    	if(!file.read(reinterpret_cast<char*>(&rows), sizeof(int))) break;
			if(!file.read(reinterpret_cast<char*>(&cols), sizeof(int))) break;
		    
		    	// make a matrix to add to the back of the list
		    	Matrix W(rows, cols);

		    	// fill the Matrix
		    	file.read(reinterpret_cast<char*>(W.data()), rows * cols * sizeof(double));

		    	// add matrix to the back of the vector
		    	weights.push_back(W);
		    }
	            file.close();
		}
		else std::cerr << "Unable to open file." << std::endl;

		return weights;
	}

	std::vector<Vector> read_biases(const std::string& file_path){
		// open the biases file
		std::ifstream file(file_path, std::ios::binary);
		std::vector<Vector> biases;

		if(file.is_open()){
		    while(!file.eof()){
			int size;

			//read size of vector at the same time of checking in its there
			//if its not leave loop because we hit the end of the file
			if (!file.read(reinterpret_cast<char*>(&size), sizeof(int))) break;

			//make Vector to add to
			Vector b(size);

			//fill vector
			file.read(reinterpret_cast<char*>(b.data()), size * sizeof(double));

			//add Vector to the back to the std::vector
			biases.push_back(b);
		    }
	            file.close();
		}
		else std::cerr << "Unable to open file." << std::endl;

		return biases;
	}
}

#endif
