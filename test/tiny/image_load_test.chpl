use NDArray;

ndarray.setGlobalRandomSeed(0);

config const imagePath: string;

var a = ndarray.loadImage(imagePath);

writeln(a.shape);

a.saveImage("new_sun.png");

for i in 0..<10 {
    var randomSun = (1:real/i:real) * ndarray.random((...a.shape));
    var b = a * randomSun;
    b.saveImage("new_sun" + i:string + ".png");
}

var randomSun = 0.001 * ndarray.random((...a.shape));


var b = a + randomSun;
b.saveImage("new_sun_hello.png");