use crate::matrix::Matrix;

pub struct NeuralNetwork {
    topology: Vec<usize>,
    weights: Vec<Matrix>,
    bias: Vec<Matrix>,
    z: Vec<Matrix>,
    a: Vec<Matrix>,
    error_l: Matrix,
    learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(topology: Vec<usize>, learning_rate: f64) -> Result<NeuralNetwork, &'static str> {
        if topology.len() < 2 {
            return Err("Topology must have an input layer and an output layer.");
        }

        let mut nn = NeuralNetwork {
            topology: topology,
            weights: Vec::new(),
            bias: Vec::new(),
            z: Vec::new(),
            a: Vec::new(),
            error_l: Matrix::new(0, 0),
            learning_rate: learning_rate,
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

    pub fn feedforward(&mut self, input: &Matrix) -> Result<Matrix, &str> {
        if input.rows() != 1 || input.columns() != self.topology[0] {
            return Err("Input does not match the topology.");
        }

        self.z.clear();
        self.a.clear();

        self.z.push(input.clone());
        self.a.push(input.clone());

        let mut layer: Matrix = Matrix::new(0, 0);
        for i in 0..self.topology.len() - 1 {
            if i == 0 {
                layer = Matrix::multiply(input, &self.weights[i]).unwrap();
            } else {
                layer = Matrix::multiply(&layer, &self.weights[i]).unwrap();
            }
            layer = Matrix::add(&layer, &self.bias[i]).unwrap();
            self.z.push(layer.clone());
            layer = Matrix::activate(&layer);
            self.a.push(layer.clone());
        }

        Ok(layer)
    }

    pub fn backpropagation(&mut self, input: &Matrix, target: &Matrix) -> Result<(), &str> {
        if input.rows() != 1 || input.columns() != self.topology[0] {
            return Err("Input does not match the topology.");
        }

        self.feedforward(&input).unwrap();

        let mut i: i64 = self.topology.len() as i64 - 2;
        while i >= 0 {
            // error_l for last layer
            if i == self.topology.len() as i64 - 2 {
                let loss_l = Matrix::subtract(target, &self.a[i as usize + 1]).unwrap();
                let z_prime = Matrix::activate_prime(&self.z[i as usize + 1]);
                self.error_l = Matrix::hadamard_product(&loss_l, &z_prime).unwrap().clone();
            // error_l for all other layers
            } else {
                let weights_t = Matrix::transpose(&self.weights[i as usize + 1]);
                let z_prime = Matrix::activate_prime(&self.z[i as usize + 1]);
                let wt_error_l_product = Matrix::multiply(&self.error_l, &weights_t).unwrap();
                self.error_l = Matrix::hadamard_product(&wt_error_l_product, &z_prime).unwrap();
            }

            // new layer bias
            let bias_gradient = Matrix::scalar(&self.error_l, self.learning_rate);
            self.bias[i as usize] = Matrix::add(&self.bias[i as usize], &bias_gradient).unwrap();

            // new layer weights
            for k in 0..self.weights[i as usize].rows() {
                for j in 0..self.weights[i as usize].columns() {
                    let q: f64;
                    let e: f64;
                    match self.a[i as usize].get_value(0, k) {
                        Some(x) => q = x,
                        None => return Err("Could not get a value."),
                    }
                    match self.error_l.get_value(0, j) {
                        Some(x) => e = x,
                        None => return Err("Could not get e value."),
                    }
                    match self.weights[i as usize].get_value(k, j) {
                        Some(x) => {
                            let value = x + self.learning_rate * q * e;
                            self.weights[i as usize].set_value(k, j, value).unwrap();
                        },
                        None => return Err("Could not compute new weight values."),
                    }
                }
            }
            i -= 1;
        }

        Ok(())
    }
}
