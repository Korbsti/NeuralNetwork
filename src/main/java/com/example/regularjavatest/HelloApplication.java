package com.example.regularjavatest;


import java.util.Arrays;

public class HelloApplication {

    public static void main(String[] args) {
        /*
        int size = 100000;
        NeuralNetworkManager neuralNetworkManager = new NeuralNetworkManager();
        NeuralNetwork network = neuralNetworkManager.loadNetwork(new File(System.getProperty("user.home") + File.separator + "Downloads" + File.separator + "network240.11.txt"));
        double[] inputs = new double[17];
        // generate a random number of 0 or 1
        for (int x = 0; x != 17; x++) {
            if (x < 15) {
                if (Math.random() > 0.5) {
                    inputs[x] = 1;
                } else {
                    inputs[x] = 0;
                }
            } else {
                inputs[x] = Math.random();
            }
        }


        for (int i = 0; i != size; i++) {

            inputs = new double[17];

            for (int x = 0; x != 17; x++) {
                if (x < 15) {
                    if (Math.random() > 0.5) {
                        inputs[x] = 0.9;
                    } else {
                        inputs[x] = 0.1;
                    }
                } else {
                    inputs[x] = Math.random();
                }
            }

            network.train(inputs, new double[]{0.99, 0.1});

        }
        */


        NeuralNetwork network = new NeuralNetwork(3, 2, 4, 1, 0.1, 0.00001);


        //NeuralNetwork network = new NeuralNetwork(new int[]{6, 6, 6, 6, 2}, 0.1);
        int size = 100000;
        //print inputs
        int first = 0;
        int second = 0;
        for (int i = 0; i != size; i++) {
            network.train(new double[]{0.1, 0.478, 0.9}, new double[]{0.99});
            network.train(new double[]{0.2, 0.49, 0.9}, new double[]{0.99});
            network.train(new double[]{0.3, 0.445, 0.9}, new double[]{0.99});
            network.train(new double[]{0.74, 0.33, 0.1}, new double[]{0.01});
            network.train(new double[]{0.345, 0.15, 0.1}, new double[]{0.01});
            network.train(new double[]{0.53, 0.23, 0.1}, new double[]{0.01});

            }

            /*
        double[] inputs = new double[76];

            for (int x = 0; x != 76; x++) {
                if (x < 74) {
                    if (Math.random() > 0.5) {
                        inputs[x] = 0.9;
                    } else {
                        inputs[x] = 0.1;
                    }
                } else {
                    inputs[x] = Math.random();
                }
            }
            if (inputs[74] > 0.5 || inputs[75] > 0.5) {
                network.train(inputs, new double[]{0.99});
                first++;
            } else {
                network.train(inputs, new double[]{0.01});
                second++;
            }*/

        }

}

/*    private void backpropagation(double[] expected) {
        for (int i = 0; i != outputNodes; i++) {
            outputLayer.get(i).error = expected[i] - outputLayer.get(i).getValue();
        }

        for (int i = hiddenLayersCount - 1; i >= 0; i--) {
            for (int j = 0; j != hiddenLayersNeuronCount; j++) {
                if (i == hiddenLayersCount - 1) {
                    double error = 0;
                    for (int k = 0; k != outputNodes; k++) {
                        error += outputLayer.get(k).error * weights[i + 1][k][j];
                    }
                    hiddenLayers.get(i).get(j).error = error;
                } else {
                    double error = 0;
                    for (int k = 0; k != hiddenLayersNeuronCount; k++) {
                        error += hiddenLayers.get(i + 1).get(k).error * weights[i + 1][k][j];
                    }
                    hiddenLayers.get(i).get(j).error = error;
                }
            }
        }

        for (int i = 0; i != hiddenLayersCount; i++) {
            for (int j = 0; j != hiddenLayersNeuronCount; j++) {
                double error = hiddenLayers.get(i).get(j).error;
                if (i == 0) {
                    for (int k = 0; k != inputNodes; k++) {
                        // Calculate the gradient of the loss function with respect to the weight
                        double grad = learningRate * error * inputLayer.get(k).getValue();
                        // output grad
                        // Update the first and second moments of the gradient
                        m[i][j][k] = beta1 * m[i][j][k] + (1 - beta1) * grad;
                        // print out m above
                        v[i][j][k] = beta2 * v[i][j][k] + (1 - beta2) * grad * grad;
                        // Update the moving averages of the first and second moments
                        double mHat = m[i][j][k] / (1 - Math.pow(beta1, t));
                        double vHat = v[i][j][k] / (1 - Math.pow(beta2, t));
                        // Update the weight using the Adam update rule
                        // print vHat
                        weights[i][j][k] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
                    }
                    // Calculate the gradient of the loss function with respect to the bias
                    double grad = learningRate * error;
                    // Update the first and second moments of the gradient
                    mBias[i][j] = beta1 * mBias[i][j] + (1 - beta1) * grad;
                    vBias[i][j] = beta2 * vBias[i][j] + (1 - beta2) * grad * grad;
                    // Update the moving averages of the first and second moments
                    double mHat = mBias[i][j] / (1 - Math.pow(beta1, t));
                    double vHat = vBias[i][j] / (1 - Math.pow(beta2, t));
                    // Update the bias using the Adam update rule
                    hiddenLayers.get(i).get(j).bias -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
                } else {
                    for (int k = 0; k != hiddenLayersNeuronCount; k++) {
                        // Calculate the gradient of the loss function with respect to the weight
                        double grad = learningRate * error * hiddenLayers.get(i - 1).get(k).getValue();
                        // Update
                        // Update the first and second moments of the gradient
                        m[i][j][k] = beta1 * m[i][j][k] + (1 - beta1) * grad;
                        v[i][j][k] = beta2 * v[i][j][k] + (1 - beta2) * grad * grad;

                        // print out v
                        // Update the moving averages of the first and second moments
                        double mHat = m[i][j][k] / (1 - Math.pow(beta1, t));
                        double vHat = v[i][j][k] / (1 - Math.pow(beta2, t));
                        // Update the weight using the Adam update rule
                        weights[i][j][k] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
                    }
                    // Calculate the gradient of the loss function with respect to the bias
                    double grad = learningRate * error;
                    // Update the first and second moments of the gradient
                    mBias[i][j] = beta1 * mBias[i][j] + (1 - beta1) * grad;
                    vBias[i][j] = beta2 * vBias[i][j] + (1 - beta2) * grad * grad;
                    // Update the moving averages of the first and second moments
                    double mHat = mBias[i][j] / (1 - Math.pow(beta1, t));
                    double vHat = vBias[i][j] / (1 - Math.pow(beta2, t));
                    // Update the bias using the Adam update rule
                    hiddenLayers.get(i).get(j).bias -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
                }
            }
        }

        for (int i = 0; i != outputNodes; i++) {
            double error = outputLayer.get(i).error;
            for (int j = 0; j != hiddenLayersNeuronCount; j++) {
                // Calculate the gradient of the loss function with respect to the weight
                double grad = learningRate * error; // Update the first and second moments of the gradient
                m[hiddenLayersCount][i][j] = beta1 * m[hiddenLayersCount][i][j] + (1 - beta1) * grad;
                v[hiddenLayersCount][i][j] = beta2 * v[hiddenLayersCount][i][j] + (1 - beta2) * grad * grad;
                // Update the moving averages of the first and second moments
                double mHat = m[hiddenLayersCount][i][j] / (1 - Math.pow(beta1, t));
                double vHat = v[hiddenLayersCount][i][j] / (1 - Math.pow(beta2, t));
                // Update the weight using the Adam update rule
                weights[hiddenLayersCount][i][j] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
            }
            // Calculate the gradient of the loss function with respect to the bias
            double grad = learningRate * error;
            // Update the first and second moments of the gradient
            mBias[hiddenLayersCount][i] = beta1 * mBias[hiddenLayersCount][i] + (1 - beta1) * grad;
            vBias[hiddenLayersCount][i] = beta2 * vBias[hiddenLayersCount][i] + (1 - beta2) * grad * grad;
            // Update the moving averages of the first and second moments
            double mHat = mBias[hiddenLayersCount][i] / (1 - Math.pow(beta1, t));
            double vHat = vBias[hiddenLayersCount][i] / (1 - Math.pow(beta2, t));// Update the bias using the Adam update rule
            outputLayer.get(i).bias -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
        }

        // Increment the time step
        t++;
    }

*/
