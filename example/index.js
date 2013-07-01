
var nnet = require("../lib/mlp.js");
var xorNNet = new nnet({
	name        : "xor",
	layers      : [3, 3, 3, 1],
	learningRate: 0.3,
	step        : 0.1,
	threshold   : 0.00001,
	epochs      : 200000
});

var data = [];
var target = [];
var test = [];

data.push([0, 0, 0]); target.push(0);
data.push([0, 0, 1]); target.push(1);
data.push([0, 1, 0]); target.push(1);
data.push([0, 1, 1]); target.push(0);
data.push([1, 0, 0]); target.push(1);
data.push([1, 0, 1]); target.push(0);
data.push([1, 1, 0]); target.push(0);
data.push([1, 1, 1]); target.push(1);

//data.push([1, 1, 1]); target.push(0);
//data.push([255/255, 230/255, 230/255]); target.push(10/100);
//data.push([255/255, 204/255, 205/255]); target.push(20/100);
//data.push([255/255, 179/255, 180/255]); target.push(30/100);
//data.push([255/255, 153/255, 155/255]); target.push(40/100);
//data.push([255/255, 128/255, 130/255]); target.push(50/100);
//data.push([255/255, 102/255, 105/255]); target.push(60/100);
//data.push([255/255, 77/255, 79/255]); target.push(70/100);
//data.push([255/255, 51/255, 54/255]); target.push(80/100);
//data.push([255/255, 25/255, 29/255]); target.push(90/100);
//data.push([255/255, 0, 4/255]); target.push(1);
//data.push([1, 1, 1]); target.push(0);

xorNNet.train(data, target)
	//.on("step", console.log.bind(console, "train step"))
	.on("complete", function() {
		console.log("train complete");

		test.push([0, 0, 0]);
		test.push([0, 0, 1]);
		test.push([0, 1, 0]);
		test.push([0, 1, 1]);
		test.push([1, 0, 0]);
		test.push([1, 0, 1]);
		test.push([1, 1, 0]);
		test.push([1, 1, 1]);

//		test.push([1, 1, 1]);
//		test.push([255 / 255, 230 / 255, 230 / 255]);
//		test.push([255 / 255, 204 / 255, 205 / 255]);
//		test.push([255 / 255, 179 / 255, 180 / 255]);
//		test.push([255 / 255, 153 / 255, 155 / 255]);
//		test.push([255 / 255, 128 / 255, 130 / 255]);
//		test.push([255 / 255, 102 / 255, 105 / 255]);
//		test.push([255 / 255, 77 / 255, 79 / 255]);
//		test.push([255 / 255, 51 / 255, 54 / 255]);
//		test.push([255 / 255, 25 / 255, 29 / 255]);
//		test.push([255 / 255, 0, 4 / 255]);

		xorNNet.run(test)
			.on("step", function(index, data, out) {
				console.log(index, ":", data[0], data[1], data[2], " -> ", out);
				//console.log(index, "r:", data[0] * 255, "g:", data[1] * 255, "b:", data[2] * 255, " -> ", out[0] * 100);
			})
			.on("complete", console.log.bind(console, "run complete"));
	});

