#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "FileIO.h"

#include <time.h>

using namespace cv;
using namespace cv::ml;
using namespace std;



void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector)
{
	// get the support vectors
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);
	hog_detector.clear();

	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
}


/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData)
{
	//--Convert data
	const int rows = (int)train_samples.size();
	const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
	cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = cv::Mat(rows, cols, CV_32FC1);
	vector< Mat >::const_iterator itr = train_samples.begin();
	vector< Mat >::const_iterator end = train_samples.end();
	for (int i = 0; itr != end; ++itr, ++i)
	{
		CV_Assert(itr->cols == 1 ||
			itr->rows == 1);
		if (itr->cols == 1)
		{
			transpose(*(itr), tmp);
			tmp.copyTo(trainData.row(i));
		}
		else if (itr->rows == 1)
		{
			itr->copyTo(trainData.row(i));
		}
	}
}


void load_pos_images(const string & filename, vector< Mat > & img_lst, bool debug)
{
	vector<string> imagelist;
	vector<vector<cv::Rect>> rectlists;
	if(!LoadAnnotationFile(filename, imagelist, rectlists))	
	{
		cerr << "Unable to open the list of images from " << filename << " filename." << endl;
		exit(-1);
	}

	for (int i = 0; i < imagelist.size(); i++) {
		Mat img = imread(imagelist[i]);
		if (img.empty())
			continue;
		for (int j = 0;j < rectlists[i].size();j++) {
			Mat part_img = img(rectlists[i][j]);
			if (debug) {
				imshow("image", part_img);
				waitKey(10);
			}
			img_lst.push_back(part_img.clone());
		}
	}
}



void load_images(const string & filename, vector< Mat > & img_lst, bool debug)
{
	vector<string> imagelist;
	if (!ReadList(filename, imagelist))
	{
		cerr << "Unable to open the list of images from " << filename << " filename." << endl;
		exit(-1);
	}

	for (int i = 0; i < imagelist.size(); i++) {
		Mat img = imread(imagelist[i]);
		if (img.empty())
			continue;
			
		if (debug) {
			imshow("image", img);
			waitKey(10);
		}
		img_lst.push_back(img);
	}
}


void sample_neg(const std::vector< cv::Mat > & full_neg_lst, std::vector< cv::Mat > & neg_lst, const cv::Size & size, bool debug)
{
	Rect box;
	box.width = size.width;
	box.height = size.height;

	const int size_x = box.width;
	const int size_y = box.height;

	srand((unsigned int)time(NULL));

	vector< Mat >::const_iterator img = full_neg_lst.begin();
	vector< Mat >::const_iterator end = full_neg_lst.end();
	for (; img != end; ++img)
	{
		box.x = rand() % (img->cols - size_x);
		box.y = rand() % (img->rows - size_y);
		Mat roi = (*img)(box);
		neg_lst.push_back(roi.clone());
		if (debug) {
			imshow("img", roi.clone());
			waitKey(10);
		}
	}
}


// From http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size)
{
	const int DIMX = size.width;
	const int DIMY = size.height;
	float zoomFac = 3;
	Mat visu;
	resize(color_origImg, visu, Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac)));

	int cellSize = 8;
	int gradientBinSize = 9;
	float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?

																	   // prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				cellx = blockx;
				celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				  // note: overlapping blocks lead to multiple updates of this sum!
				  // we therefore keep track how often a cell was updated,
				  // to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	  // compute average gradient strengths
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			int mx = drawX + cellSize / 2;
			int my = drawY + cellSize / 2;

			rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), Scalar(100, 100, 100), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)(cellSize / 2.f);
				float scale = 2.5; // just a visualization scale, to see the lines better

								   // compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visualization
				line(visu, Point((int)(x1*zoomFac), (int)(y1*zoomFac)), Point((int)(x2*zoomFac), (int)(y2*zoomFac)), Scalar(0, 255, 0), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	  // don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;

} // get_hogdescriptor_visu

void compute_hog(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size, bool debug)
{
	HOGDescriptor hog;
	hog.winSize = size;
	Mat gray;
	vector< Point > location;
	vector< float > descriptors;

	vector< Mat >::const_iterator img = img_lst.begin();
	vector< Mat >::const_iterator end = img_lst.end();
	for (; img != end; ++img)
	{
		cvtColor(*img, gray, COLOR_BGR2GRAY);
		hog.compute(gray, descriptors, Size(8, 8), Size(0, 0), location);
		gradient_lst.push_back(Mat(descriptors).clone());
		if (debug) {
			imshow("gradient", get_hogdescriptor_visu(img->clone(), descriptors, size));
			waitKey(10);
		}
	}
}

cv::Ptr<cv::ml::SVM> train_svm(const vector< Mat > & gradient_lst, const vector< int > & labels)
{

	Mat train_data;
	convert_to_ml(gradient_lst, train_data);

	clog << "Start training...";
	Ptr<SVM> svm = SVM::create();
	/* Default values to train SVM */
	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	svm->setGamma(0);
	svm->setKernel(SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
	svm->setC(0.01); // From paper, soft classifier
	svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
	svm->train(train_data, ROW_SAMPLE, Mat(labels));
	clog << "...[done]" << endl;

	//svm->save("my_people_detector.yml");
	return svm;
}


void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color)
{
	if (!locations.empty())
	{
		vector< Rect >::const_iterator loc = locations.begin();
		vector< Rect >::const_iterator end = locations.end();
		for (; loc != end; ++loc)
		{
			rectangle(img, *loc, color, 2);
		}
	}
}


void get_hard_negatives(const std::vector<std::string>& test_imgs, const cv::Ptr<cv::ml::SVM>& svm, const cv::Size& size,
	std::vector<cv::Mat>& hard_negs, bool debug)
{
	HOGDescriptor my_hog;
	my_hog.winSize = size;

	// Set the trained svm to my_hog
	vector< float > hog_detector;
	get_svm_detector(svm, hog_detector);
	my_hog.setSVMDetector(hog_detector);
	//my_hog.setSVMDetector(my_hog.getDefaultPeopleDetector());

	vector<string>::const_iterator imf;
	vector<Rect>::const_iterator rect_i;
	hard_negs.clear();
	for (imf = test_imgs.begin(); imf != test_imgs.end(); imf++) {
		vector< Rect > locations;
		Mat im = imread(*imf);
		if (im.empty())
			continue;
		my_hog.detectMultiScale(im, locations);
		printf("%d hard negatives in %s.\n", locations.size(), imf->c_str());
		for (rect_i = locations.begin();rect_i != locations.end();rect_i++) {
			Mat resize_im;
			cv::resize(im(*rect_i), resize_im, size);
			hard_negs.push_back(resize_im);
		}
		if (debug) {
			draw_locations(im, locations, cv::Scalar(255, 0, 0));
			imshow("hard_negatives", im);
			waitKey(10);
		}
	}
}



void TrainHOGdetector(const std::string& pos, const std::string& neg, const std::string& val,
	const cv::Size& train_size, const std::string& save_svm, const std::string& save_old_svm, bool debug, bool restart_hns)
{
	vector< Mat > pos_lst;
	vector< Mat > neg_lst;
	vector< Mat > gradient_lst;
	vector< int > labels;

	load_pos_images(pos, pos_lst, debug);
	printf("Load %d positive images.\n", pos_lst.size());
	labels.assign(pos_lst.size(), +1);
	const unsigned int old = (unsigned int)labels.size();

	load_images(neg, neg_lst, debug);
	printf("Load %d negative images.\n", neg_lst.size());
	labels.insert(labels.end(), neg_lst.size(), -1);
	CV_Assert(old < labels.size());

	compute_hog(pos_lst, gradient_lst, train_size, debug);
	pos_lst.clear();
	printf("Compute positive samples' HOG.\n");
	compute_hog(neg_lst, gradient_lst, train_size, debug);
	neg_lst.clear();
	printf("Compute negative samples' HOG.\n");

	Ptr<SVM> svm;
	if (restart_hns) {
		printf("load %s\n", save_old_svm.c_str());
		svm = SVM::load<SVM>(save_old_svm);
	}
	else {
		svm = train_svm(gradient_lst, labels);
	}

	printf("Train support vector machine.\n");
	if (val.empty()) {
		svm->save(save_svm);
		printf("save %s as trained file.", save_svm.c_str());
		return;
	}
	else {
		svm->save(save_old_svm);
		printf("save %s as trained file.", save_old_svm.c_str());
	}

	std::vector<string> val_lst;
	if (!ReadList(val, val_lst)) {
		printf("Fail to read %s.\n", val.c_str());
		exit(-1);
	}

	std::vector<cv::Mat> false_pos_lst;
	get_hard_negatives(val_lst, svm, train_size, false_pos_lst, debug);

	val_lst.clear();
	printf("Got %d hard negative samples.\n", false_pos_lst.size());

	compute_hog(false_pos_lst, gradient_lst, train_size, debug);
	labels.insert(labels.end(), false_pos_lst.size(), -1);
	printf("Compute hard negative samples' HOG.\n");

	false_pos_lst.clear();
	svm = train_svm(gradient_lst, labels);
	svm->save(save_svm);
	printf("Train SVM and save as %s.\n", save_svm.c_str());
}


int main(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv,
		"{help h|| show help message}"
		"{positive p|| positive image file}"
		"{negative n|| negative image list file}"
		"{validate v|| validate image list file for hard negative sampling}"
		"{debug d|| display train images and HOGS}"
		"{start_hns s|| restart from hard negative sampling}"
		"{width W|64| width of training image}"
		"{height H|128| height of training image}"
		"{@output|| output trained parameter file}");
	if (parser.has("help"))
	{
		parser.printMessage();
		exit(0);
	}
	string pos = parser.get<string>("p");
	string neg = parser.get<string>("n");
	string val = parser.get<string>("v");
	string outputf = parser.get<string>("@output");
	if (pos.empty() || neg.empty() || outputf.empty())
	{
		cout << "Wrong number of parameters." << endl;
		parser.printMessage();
		exit(-1);
	}

	int w = parser.get<int>("W");
	int h = parser.get<int>("H");
	bool debug = parser.has("d");
	bool restart_hns = parser.has("s");
	TrainHOGdetector(pos, neg, val, cv::Size(w, h), outputf, outputf + ".tmp", debug, restart_hns);

	//TestHOGdetector("testlist.txt", "test2", cv::Size(w, h), "my_people_detector.yml");
	//test_it(Size(96, 160)); // change with your parameters

	return 0;
}
