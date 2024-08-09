#ifndef NUM_RECOGNIZER_HPP
#define NUM_RECOGNIZER_HPP
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>

namespace ann{
	class neural_network{
	public:
		using Vector = Eigen::VectorXd;
		using Matrix = Eigen::MatrixXd;

		//constucts a NN object
		//takens in a vector with the sizes of each layer, this vectors size 
		//equals the amont of layers that are in the NN
		//creates n - 1 maticx for the weights and n - 1 vectors for the bias
		//the matrix are of size row: lenght of prev layer, col: lenght of next layer
		//the bias vectors are size of the lenght of the prev layer
		neural_network(const std::vector<int>& layers) : layers_(layers){
			//get the amount of layers in the NN
			int num_layers = layers.size();

			//randomly grenerate the weights and bieses
			for(int i = 0; i < num_layers - 1; ++i){
				weights_.emplace_back(Matrix::Random(layers[i], layers[i - 1]));
				biases_.emplace_back(Vector::Random(layers[i]));
			}
			
		}
		
		//the sigmoid function takens in a vector 
		//and maps the input values to an output range between 0 and 1
		// sigmoid function = 1 / (1 + (e^-x))

		//notes: .array() turns the eigen vactor in to an array type allowing
		// elemnt-wise calucations to be done
		// .exp() is also from the eigen liabry and it takes each elemnt in the array
		// as an expoint to e	
		//has not been tested yet
		Vector sigmoid(const Vector& x){
			return 1.0 / ( 1.0 + (( -x.array() ).exp() ) );
		}
		
		//the feed_forward function takes in a input layer 
		//mutiblys the weights and input layers together
		//then addeds the bias
		//then calls the sigmoid function 
		//resaltion in a vector of the next layers activation values
		Vector feed_forward(const Vector& input);
		
		//returns the current activations in the NN
		//this might be unnescessary as i don't think we have to save the
		// activations but we might want to for understating what it happen in are NN
		// but it dose make extra work that we dont really need to do
		const std::vector<Vector>& get_activations() const{
			return activations_;
		}
		
		//retruns the vector with the current matrix of weights on it
		const std::vector<Matrix>& get_weights() const{
			return weights_;
		}
		
		//retunrs the Vector with the current vectors of biases
		const std::vector<Vector>& get_biases() const{
			return biases_;
		}

	private:
		std::vector<int> layers_;
		std::vector<Matrix> weights_;
		std::vector<Vector> biases_;
		
		//might not be nescessary
		std::vector<Vector> activations_;
	};
}

#endif
