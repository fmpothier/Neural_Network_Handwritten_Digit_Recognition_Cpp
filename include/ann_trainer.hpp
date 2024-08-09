#ifndef NUM_RECOGNIZER_HPP
#define NUM_RECOGNIZER_HPP
#include "num_recognizer.hpp"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>

namespace ann{
	class trainer{
		public:
			
			//maybe the neural_network and trainer class 
			//should be put in to one class to make things simplayer

 
			using Vector = Eigen::VectorXd;
			using Matrix = Eigen::MatrixXd;
			
			//i think it should take in a instance of the nn class? not sure tho
			trainer(neural_network& nn);
			
			//the propagate_forward will taken in a image in matrx form for input
			//and calc the output will call feed_forward to calc the next layer of activations values
			//maybe this function should return the last layer?
			//the image will be put in to the vector using row major form?
			void propagate_forward(const Vector& input);
			
			//maybe takens in the label for the image we are looking at?
			void propagate_backward();
			
			//
			void calc_error();
			void update_weights();	
		
	};


}

#endif
