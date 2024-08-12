#ifndef NUM_RECOGNIZER_HPP
#define NUM_RECOGNIZER_HPP

#include <stdexcept>
#include <vector>
#include <Eigen/Dense>
#include <iostream>

namespace ann{

	class neural_network{
	public:
		using Vector = Eigen::VectorXd;
		using Matrix = Eigen::MatrixXd;
		using size_type = std::size_t;

		// Constructs a NN object.
		// Takes in a vector of layer sizes. The size of the vector is the number of layers (n). 
		// Creates n-1 matrices for the weights, and n-1 vectors for the biases.
		// Weight matrix: number of rows = size of output vector.
		// Number of columns = size of input vector.
		// The bias vectors are size of the length of the prev layer.
		neural_network(const std::vector<size_type>& layers) : layers_(layers) {

			// Gets the amount of layers in the NN.
			const size_type num_layers = layers.size();

			// Randomly generate the weights and biases.
			for(size_type i = 0; i < num_layers - 1; ++i){ 
				weights_.emplace_back(Matrix::Random(layers.at(i+1), layers.at(i)));
				biases_.push_back(Vector::Random(layers.at(i+1)));
			} 
		}

		neural_network(const std::vector<size_type>& layers, const std::vector<Matrix>& W,
		 const std::vector<Vector>& b) : layers_(layers), weights_(W), biases_(b) {}

		
		// The sigmoid function takes in a vector 
		// and maps the values to an range between 0 and 1.
		// Sigmoid function = 1 / (1 + (e^-x))

		// Notes: .array() turns the input vector into an array type allowing
		// element-wise calucations to be done.
		// .exp() is also from the eigen library, and it takes each elemnet in the array
		// and exponentiates it.
		static Vector sigmoid(const Vector& x){
			Vector sigmoid_x = -x;	
			sigmoid_x = 1.0 / (1.0 + sigmoid_x.array().exp());
			return sigmoid_x;
		}
		
		// The feed_forward function takes in an input layer 
		// mutiplies it by the weights, then adds the biases.
		// Then it calls the sigmoid function on the result,
		// resulting in a vector of the next layer's activation values.
		Vector feed_forward(const Vector& input){
			// Input is of size 784. Output is of size 10.
			Vector output;

			// Clear old activations so that we can replace with new 
			// activations for each level of the NN
//			activations_.clear();

			//first layer of activations 
//			activations_.push_back(input);

			// This variable will hold our vector of node activations for forward
			// propagation.
			Vector cur_layer = input;
			for(size_type index = 0; index < weights_.size(); ++index){	
				cur_layer = sigmoid((weights_.at(index)*cur_layer) + biases_.at(index));
//				activations_.push_back(cur_layer);
			}
			output = cur_layer;
			return output;
		}
	
		//returns the current activations in the NN
		//this might be unnescessary as i don't think we have to save the
		// activations but we might want to for understating what it happen in are NN
		// but it dose make extra work that we dont really need to do
/*		const std::vector<Vector>& get_activations() const{
			return activations_;
		}
*/		
		//retruns the vector with the current matrix of weights on it
		const std::vector<Matrix>& get_weights() const{
			return weights_;
		}
		
		//retunrs the Vector with the current vectors of biases
		const std::vector<Vector>& get_biases() const{
			return biases_;
		}

	private:
		std::vector<size_type> layers_;
		std::vector<Matrix> weights_;
		std::vector<Vector> biases_;
		
		//might not be nescessary
//		std::vector<Vector> activations_;
	};
}

#endif
