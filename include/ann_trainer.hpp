#ifndef NUM_RECOGNIZER_HPP
#define NUM_RECOGNIZER_HPP

#include "num_recognizer.hpp"
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>

namespace ann{
	class trainer{
	    public:
		using Vector = Eigen::VectorXd;
		using Matrix = Eigen::MatrixXd;
		
		trainer(neural_network& nn);
		
		//the propagate_forward will taken in a image in matrx form for input
		//and calc the output will call feed_forward to calc the next layer of activations values
		//maybe this function should return the last layer?
		//the image will be put in to the vector using row major form?
		void propagate_forward(const Vector& input);
		
		//maybe takens in the label for the image we are looking at?
		void propagate_backward();
		
		//function that takens in a matcix or/and vecotor and with write
		//to the weights and bieses .txt files 
		//maybe if it sucssefully writes it will return true and else false
		//should we do this in two functions? one fore weights and one for bieses
		// or do both in the same functio
		bool write(const std::vector<Matrix>& W, const std::<Vector>& b);
		
		//this function will read from the Weight.txt and/or bieses.txt files to
		// fill in the weights matrixses and the bieses vectors 
		// if we do it in the same function we can take in empty vectors of matrix
		// vectors and just fill then up and return true if that all gose good
		// or we can have two differetn fuctions that will create the vector of matrix
		// and vector of vecotrs and return them	
		//first way:
		bool read(std::vector<Matrix>& W, std::vector<Vector>& b);

		//second way:
		std::vector<Matrix> read_weights();
		std::vector<Vector> read_bieses();


		void calc_error();
		void update_weights();	
	
	};


}

#endif
