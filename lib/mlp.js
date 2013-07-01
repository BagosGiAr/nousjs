var events = require('events');

module.exports = (function (options) {

	var layers = [3, 3, 1];

	var defaults = {
		layers      : options.layers || layers,
		learningRate: options.learningRate || 0.3,
		step        : options.step || 0.1,
		threshold   : options.threshold || 0.0001,
		epochs      : options.epochs || 200000,
		activate    : options.activate || __sigmoid,
		rand        : options.rand || __rand
	};

	/**
	 *
	 * @returns {*}
	 * @constructor
	 */
	function NeuralNet(options) {
		this.name = options.name || "";
		this.defaults = defaults;
		this.layers = options.layers || this.defaults.layers;
		this.learningRate = options.learningRate || this.defaults.learningRate;
		this.step = options.step || this.defaults.step;
		this.threshold = options.threshold || this.defaults.threshold;
		this.epochs = options.epochs || this.defaults.epochs;
		this.activate = options.activate || this.defaults.activate;
		this.rand = options.rand || this.defaults.rand;
		this.delta = [];
		this.weight = [];
		this.pwStage = []; // weight change
		this.out = [];
		this.error = [];

		// Seed and assign random weights & initialize previous weights to 0
		for (var i = 1; i < this.layers.length; i++) {
			this.weight[i] = new Array(this.layers[i]);
			this.pwStage[i] = new Array(this.layers[i]);

			for (var j = 0; j < this.layers[i]; j++) {
				this.weight[i][j] = new Array(this.layers[i - 1]);
				this.pwStage[i][j] = new Array(this.layers[i - 1] + 1);

				for (var k = 0; k < this.layers[i - 1] + 1; k++) {
					this.weight[i][j][k] = this.rand(i, j);
					this.pwStage[i][j][k] = 0.0;
				}
				// bias in the last neuron
				this.weight[i][j][this.layers[i - 1]] = 0;
			}
		}
		// initialize delta, error & out
		for (var i = 0; i < this.layers.length; i++) {
			this.delta[i] = new Array(this.layers[i]);
			this.error[i] = new Array(this.layers[i]);
			this.out[i] = new Array(this.layers[i]);

			for (var j = 0; j < this.layers[i]; j++) {
				this.delta[i][j] = 0.0;
				this.error[i][j] = 0.0;
				this.out[i][j] = 0.0;
			}
		}
		return this;
	}

	/**
	 *
	 * @param data
	 * @returns {events.EventEmitter}
	 */
	NeuralNet.prototype.run = function (data) {
		var bubble = new events.EventEmitter();
		var loops = data.length;
		var __this = this;
		process.nextTick(function () {
			for (var i = 0; i < loops; i++) {
				__this.ffwd(data[i]);
				bubble.emit("step", i, data[i], __this.out[__this.layers.length - 1]);
			}

			bubble.emit("complete", __this.out[__this.layers.length - 1]);
		});
		return bubble;
	};

	/**
	 *
	 * @param data
	 * @param targetArr
	 * @param options
	 * @returns {events.EventEmitter}
	 */
	NeuralNet.prototype.train = function (data, targetArr, options) {
		var MSE = Infinity;
		var opts = options || {};
		var errors = opts.errors || [];
		var bubble = new events.EventEmitter();
		var __this = this;
		this.layers = opts.layers || this.layers;
		this.learningRate = opts.learningRate || this.learningRate;
		this.step = opts.step || this.step;
		this.threshold = opts.threshold || this.threshold;
		this.epochs = opts.epochs || this.epochs;

		process.nextTick(function () {
			for (var i = 0; i < __this.epochs && MSE > __this.threshold; i++) {
				// Backpropagate
				__this.bpgt(data[i % data.length], targetArr[i % targetArr.length]);

				MSE = __this.mse(targetArr[i % targetArr.length]);
				errors.push(MSE);

				bubble.emit("step", MSE, i, data[i % data.length], targetArr[i % targetArr.length]);
			}
			bubble.emit("complete", MSE, __this.weight);
			return 0;
		});

		return bubble;
	};

	/**
	 *
	 * @param s
	 * @returns {*}
	 */
	NeuralNet.prototype.activate = function (s) {
		return this.defaults.activate.call(this, s);
	};

	/**
	 *
	 * @param layer
	 * @param neuron
	 * @returns {*}
	 */
	NeuralNet.prototype.rand = function (layer, neuron) {
		return this.defaults.rand.call(this, layer, neuron);
	};

	/**
	 *
	 * @param arr
	 * @param target
	 * @returns {*}
	 */
	NeuralNet.prototype.bpgt = function (arr, target) {
		var __this = this;
		var sum = 0.0;
		var nOut = 0;

		// Update the output values for each neuron
		this.ffwd(arr);

		// Delta for output layer
		for (var i = 0; i < this.layers[this.layers.length - 1]; i++) {
			this.error[this.layers.length - 1][i] = (target - this.out[this.layers.length - 1][i]);
			this.delta[this.layers.length - 1][i] =
				this.out[this.layers.length - 1][i] * (1 - this.out[this.layers.length - 1][i])
					* this.error[this.layers.length - 1][i];
		}

		// Delta for hidden layers
		for (var i = this.layers.length - 2; i > 0; i--) {
			for (var j = 0; j < this.layers[i]; j++) {
				sum = 0.0;

				for (var k = 0; k < this.layers[i + 1]; k++) {
					sum += this.delta[i + 1][k] * this.weight[i + 1][k][j];
				}
				this.error[i][j] = sum;
				this.delta[i][j] = this.out[i][j] * (1 - this.out[i][j]) * sum;
			}
		}

//		// Step function
//		for (var i = 1; i < this.layers.length; i++) {
//			for (var j = 0; j < this.layers[i]; j++) {
//				for (var k = 0; k < this.layers[i - 1]; k++) {
//					this.weight[i][j][k] += this.step * this.pwStage[i][j][k];
//				}
//				this.weight[i][j][this.layers[i - 1]] += this.step * this.pwStage[i][j][this.layers[i - 1]];
//			}
//		}

		// Adjust weights
		for (var i = 1; i < this.layers.length; i++) {
			for (var j = 0; j < this.layers[i]; j++) {
				for (var k = 0; k < this.layers[i - 1]; k++) {
					this.pwStage[i][j][k] = this.learningRate * this.delta[i][j] * this.out[i - 1][k] + (this.step * this.pwStage[i][j][k]);
					this.weight[i][j][k] += this.pwStage[i][j][k];
				}
				// Apply corrections
				this.pwStage[i][j][this.layers[i - 1]] = this.learningRate * this.delta[i][j];
				this.weight[i][j][this.layers[i - 1]] += this.pwStage[i][j][this.layers[i - 1]];
			}
		}

		return this;
	};

	/**
	 * feed forwards activations for one set of inputs
	 * @param arr
	 * @returns {*}
	 */
	NeuralNet.prototype.ffwd = function (arr) {
		var __this = this;
		var sum = 0.0;
		var ret = 0;

		// assign content to input layer
		for (var i = 0; i < this.layers[0]; i++) {
			this.out[0][i] = arr[i] || 0;
		}

		// assign output (activation) value to each neuron usng activate func
		for (var i = 1; i < this.layers.length; i++) {
			// For each neuron in current layer
			for (var j = 0; j < this.layers[i]; j++) {
				sum = 0.0;
				// For each input from each neuron in proceeding layer
				for (var k = 0; k < this.layers[i - 1]; k++) {
					// Apply weight to inputs and add to sum
					sum += this.out[i - 1][k] * this.weight[i][j][k];
				}
				// Apply bias
				sum += this.weight[i][j][this.layers[i - 1]];
				// Apply activate function
				this.out[i][j] = this.activate(sum);
			}
			ret = this.out[i];
		}
		return ret;
	};

	/**
	 *
	 * @param target
	 * @returns {number}
	 */
	NeuralNet.prototype.mse = function (target) {
		var __this = this;
		var mse = 0;
		for (var i = 0; i < this.layers[this.layers.length - 1]; i++) {
			mse += (target - this.out[this.layers.length - 1][i]) * (target - this.out[this.layers.length - 1][i]);
		}
		return mse / 2;
	};

	return NeuralNet;
})({});

function __sigmoid(s) {
	return (1.0 / (1.0 + Math.exp(-s)));
}

function __rand() {
	return ((Math.random() * 100) / 16383.5) - 1;
	//return (Math.random() * 0.4) - 0.3;
}