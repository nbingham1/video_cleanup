#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	if(argc < 2) {
		cout << "./cleanup <video>" << endl;
		return 0;
	}

	VideoCapture cap(argv[1]);
}
