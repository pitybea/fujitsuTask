#include "utinity.h"


vector<vector<double> > imageToFeaturesQuick(const string& s)
{
	return fileIOclass::InVectorSDouble(s);
}

class trainTestTask
{
public:
	string taskName;
	string trainFolder;
	string testFolder;
	string trainImages;
	string trainLabels;
	string testImages;


	string feaFileName;//=taskName+"features.txt";
	string labelFieName;//=taskName+"classlabel.txt";
	string centerFileName;//=taskName+"featureCenters.txt";
	string clusterLabelFileName;//=taskName+"clusterlabels";

	vector<vector<double> > features;
	vector<string> labels;

	trainTestTask(string _taskName,string _trainFolder,string _testFolder,string _trainImages,string _trainLabels,string _testImages)
		:taskName(_taskName),trainFolder(_trainFolder),testFolder(_testFolder),trainImages(_trainImages),trainLabels(_trainLabels),testImages(_testImages)
	{
		feaFileName=taskName+"features.txt";
		labelFieName=taskName+"classlabel.txt";
		centerFileName=taskName+"featureCenters.txt";
		clusterLabelFileName=taskName+"clusterlabels";
	}

	void train()
	{
		_chdir(trainFolder.c_str());
		if(!( fileExists(feaFileName)&& fileExists(labelFieName)) )
		{
			features.clear();
			labels.clear();
			
			auto imgNames=fileIOclass::InVectorString(trainImages);
			auto labelNames=fileIOclass::InVectorString(trainLabels);

			for(int i=0;i<imgNames.size();i++)
			{
				if(i%3==0) cout<<i<<endl;

				auto tfeatures=imageToFeaturesQuick(imgNames[i]+".jpg.sift");
				vector<string> tlabels=vector<string>(tfeatures.size(),labelNames[i]);
				features.insert(features.end(),tfeatures.begin(),tfeatures.end());
				labels.insert(labels.end(),tlabels.begin(),tlabels.end());
			}
			fileIOclass::OutVectorSDouble(feaFileName,features);
			fileIOclass::OutVectorString(labelFieName,labels);
			cout<<"codebook generated<<"<<endl;
		}
	}

	void doKmeans()
	{
		_chdir(trainFolder.c_str());
		if(!fileExists(centerFileName))
		{
			if(features.size()==0)
			{
				features=fileIOclass::InVectorSDouble(feaFileName);
				labels=fileIOclass::InVectorString(labelFieName);
			}
			auto clusters=parallelKMeans(features);
			fileIOclass::OutVectorSDouble(centerFileName,clusters.first);
			fileIOclass::OutVectorInt(clusterLabelFileName,clusters.second);
		}
	}

};

