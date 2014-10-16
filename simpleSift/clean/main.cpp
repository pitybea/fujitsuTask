#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>


#include "..\FileIO\FileInOut.h"

using namespace std;
using namespace cv;






pair<vector<vector<double> >,vector<vector<double> > > imageToFeatures(string s)
{
	
	Mat iptimg=imread(s);

	SiftFeatureDetector detector;
	vector<KeyPoint> temkpts;
	detector.detect(iptimg,temkpts);
	SiftDescriptorExtractor extractor;
	Mat descriptor;
	extractor.compute(iptimg,temkpts,descriptor);
	
	
	vector<vector<double> > features;
	vector<vector<double> > locations;
	features.resize(descriptor.rows,vector<double>(128,0.0));
	locations.resize(descriptor.rows,vector<double>(2,0.0));
	for (int i=0;i<descriptor.rows;i++)
	{

		//temfes.pos.x=(int)temkpts[i].pt.x;
		//	temfes.pos.y=(int)temkpts[i].pt.y;

		locations[i][0]=temkpts[i].pt.x;
		locations[i][1]=temkpts[i].pt.y;
		double temsum(0.000001);
		for (int j=0;j<128;j++)
		{
			temsum+=descriptor.at<float>(i,j);
				
		}
		for (int j=0;j<128;j++)
		{
			
			features[i][j]=descriptor.at<float>(i,j)/temsum;
		}


	}
	return pair<vector<vector<double> >,vector<vector<double> > >(features,locations);
}







int main(int argc,char* argv[])
{
	assert(argc>1);
	string s=argv[1];

	pair<vector<vector<double> >,vector<vector<double>> > fealoc=imageToFeatures(s);
	vector<vector<double> > features=fealoc.first;
	vector<vector<double> > locations=fealoc.second;

	fileIOclass::OutVectorSDouble(s+".sift",features);
	fileIOclass::OutVectorSDouble(s+".siftloc",locations);


	return 0;
}




