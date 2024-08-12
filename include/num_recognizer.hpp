#ifndef NUM_RECOGNIZER_HPP
#define NUM_RECOGNIZER_HPP

#include <stdexcept>
#include <vector>
#include <Eigen/Dense>

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
			activations_.clear();

			//first layer of activations 
			activations_.push_back(input);

			// This variable will hold our vector of node activations for forward
			// propagation.
			Vector cur_layer = input;
			for(size_type index = 0; index < weights_.size(); ++index){	
				cur_layer = sigmoid((weights_.at(index)*cur_layer) + biases_.at(index));
				activations_.push_back(cur_layer);
			}
			output = cur_layer;
			return output;
		}

		// This function reads the errors propagated from calculate_error, and 
		// calculates and writes the modified weights and biases.
		void feed_backward(const double& learning_rate, const Vector& output, const Vector& expected){
			std::vector<Vector> errors = calculate_error(output, expected);

			for(size_type layer = 0; layer < weights_.size(); ++layer){
				Matrix dW = errors[layer] * activations_[layer].transpose();
				Vector dB = errors[layer];

				weights_[layer] -= learning_rate * dW;
				biases_[layer] -= learning_rate * dB;
			}
		}
	
		// This function calculates the error between the expected output layer and the 
		// actual output layer.
		std::vector<Vector> calculate_error(const Vector& output, const Vector& expected){
			std::vector<Vector> errors(weights_.size());

			Vector delta = (output - expected).array() * output.array() * (1-output.array());
			errors.back() = delta;

			for(size_type layer = weights_.size() - 1; layer > 0; --layer){
				const Matrix& current_weights = weights_[layer];
				const Vector& current_activations = activations_[layer];
				delta = (current_weights.transpose() * delta).array() *
				 current_activations.array() * (1-current_activations.array());
				errors[layer - 1] = delta;
			}

			return errors;			
		}

		const std::vector<Vector>& get_activations() const{
			return activations_;
		}
		
		void set_weights(const std::vector<Matrix>& weights){
			weights_.clear();
			weights_ = weights;
		}		

		void set_biases(const std::vector<Vector>& biases){
			biases_.clear();
			biases_ = biases;
		}

		const std::vector<Matrix>& get_weights() const{
			return weights_;
		}
		
		const std::vector<Vector>& get_biases() const{
			return biases_;
		}


	private:
		std::vector<size_type> layers_;
		std::vector<Matrix> weights_;
		std::vector<Vector> biases_;
		std::vector<Vector> activations_;
	};
}

#endif
