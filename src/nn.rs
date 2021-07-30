use crate::matrix::Matrix;

pub struct NeuralNetwork {
    topology: Vec<usize>,
    weights: Vec<Matrix>,
    bias: Vec<Matrix>,
}

impl NeuralNetwork {
    pub fn new(topology: Vec<usize>) -> Result<NeuralNetwork, &'static str> {
        if topology.len() < 2 {
            return Err("Topology must have an input layer and an output layer.");
        }

        let mut nn = NeuralNetwork {
            topology: topology,
            weights: Vec::new(),
            bias: Vec::new(),
        };

        nn.init_weights();
        nn.randomize_weights();

        nn.init_bias();
        nn.randomize_bias();

        Ok(nn)
    }

    pub fn print_network(&self) {
        print!("Topology: ");
        for i in 0..self.topology.len() {
            print!("{}, ", self.topology[i]);
        }
        println!("");
        println!("");

        println!("Weights:");
        for i in 0..self.weights.len() {
            self.weights[i].print_matrix();
        }
        println!("");

        println!("Bias:");
        for i in 0..self.bias.len() {
            self.bias[i].print_matrix();
        }
        println!("");
    }

    fn init_weights(&mut self) {
        for i in 0..self.topology.len() {
            if i + 1 < self.topology.len() {
                self.weights.push(Matrix::new(self.topology[i], self.topology[i+1]));
            }
        }
    }

    fn randomize_weights(&mut self) {
        for i in 0..self.weights.len() {
            self.weights[i].randomize();
        }
    }

    fn init_bias(&mut self) {
        for i in 0..self.topology.len() {
            if i > 0 {
                self.bias.push(Matrix::new(1, self.topology[i]));
            }
        }
    }

    fn randomize_bias(&mut self) {
        for i in 0..self.bias.len() {
            self.bias[i].randomize();
        }
    }

    pub fn feedforward(&self, input: &Matrix) -> Result<Matrix, &str> {
        if input.rows() != 1 || input.columns() != self.topology[0] {
            return Err("Input does not match the topology.");
        }

        let mut layer: Matrix = Matrix::new(0, 0);
        for i in 0..self.topology.len() - 1 {
            if i == 0 {
                layer = Matrix::multiply(input, &self.weights[i]).unwrap();
            } else {
                layer = Matrix::multiply(&layer, &self.weights[i]).unwrap();
            }
            layer = Matrix::add(&layer, &self.bias[i]).unwrap();
            layer.activate();
        }

        Ok(layer)
    }
}
