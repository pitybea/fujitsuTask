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

	bool eachCategoryMode;

	trainTestTask(const string& _taskName,const string& _trainFolder,const string& _testFolder,const string& _trainImages,const string& _trainLabels,const string& _testImages,bool _eachCategoryMode=true)
		:taskName(_taskName),trainFolder(_trainFolder),testFolder(_testFolder),trainImages(_trainImages),trainLabels(_trainLabels),testImages(_testImages),eachCategoryMode(_eachCategoryMode)
	{
		feaFileName=taskName+"features.txt";
		labelFieName=taskName+"classlabel.txt";
		centerFileName=taskName+"featureCenters.txt";
		clusterLabelFileName=taskName+"clusterlabels";
	}

	void generateGroundTrueth(const string& totalList,const string& totalLabel,string classlist="",string categoryList="")
	{
		_chdir(trainFolder.c_str());
		auto tlist=fileIOclass::InVectorString(totalList);
		auto tlables=fileIOclass::InVectorString(totalLabel);
		
		assert(tlist.size()==tlables.size());

		auto imgNames=fileIOclass::InVectorString(trainImages);
		auto labelNames=fileIOclass::InVectorString(trainLabels);

		unordered_set<string> _trainImgs;
		unordered_set<string> _trainLables;

		vector<string> groundThruthNames;
		vector<string> groundThrthLabels;

		for(auto s:imgNames)
			_trainImgs.insert(s);


		if(eachCategoryMode)
		{
			auto classNames=fileIOclass::InVectorString(classlist);
			auto categoryNames=fileIOclass::InVectorString(categoryList);
			assert(classNames.size()==categoryNames.size());
			unordered_map<string,string> classCategoryCorrespondences;
			for (int i = 0; i < classNames.size(); i++)
			{
				classCategoryCorrespondences[classNames[i]]=categoryNames[i];
			}
			for(auto s:labelNames)
				_trainLables.insert(classCategoryCorrespondences[s]);

			for(int i=0;i<tlist.size();++i)
			{
				if( (!_trainImgs.count(tlist[i]) )&& _trainLables.count(classCategoryCorrespondences[tlables[i]]))
				{
					groundThruthNames.push_back(tlist[i]);
					groundThrthLabels.push_back(tlables[i]);
				}
			}

		}
		else
		{
			for(auto s:labelNames)
				_trainLables.insert(s);

			for(int i=0;i<tlist.size();++i)
			{
				if( (!_trainImgs.count(tlist[i]) )&& _trainLables.count(tlables[i]))
				{
					groundThruthNames.push_back(tlist[i]);
					groundThrthLabels.push_back(tlables[i]);
				}
			}
		}

		fileIOclass::OutVectorString(taskName+"groundTruth"+"_names.txt",groundThruthNames);
		fileIOclass::OutVectorString(taskName+"groundTruth"+"_labels.txt",groundThrthLabels);


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
			auto clusters=parallelKMeans2(features);
			fileIOclass::OutVectorSDouble(centerFileName,clusters.first);
			fileIOclass::OutVectorInt(clusterLabelFileName,clusters.second);
		}
	}

	void testForList()
	{
		if (features.size()==0)
		{
			if( !fileExists(feaFileName)||
				!fileExists(labelFieName)||
				!fileExists(centerFileName)||
				!fileExists(clusterLabelFileName)
				)
			{
				cout<<"training not finished for "<<taskName<<endl;
			}
			else
			{
				features=fileIOclass::InVectorSDouble(feaFileName);
				labels=fileIOclass::InVectorString(labelFieName);
				pair<vector<vector<double> >,vector<int> > clusters;
				clusters.first=fileIOclass::InVectorSDouble(centerFileName);
				clusters.second=fileIOclass::InVectorInt(clusterLabelFileName);


			}
		}
	}
	private:
		string testForOne(string _testImage)
		{
		}
};

